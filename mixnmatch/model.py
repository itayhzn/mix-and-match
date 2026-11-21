"""
PixelDiT text-to-image transformer, copied from NVlabs/PixelDiT (commit 41f7300) and trimmed to inference.

Source files: pixdit_core/modules.py, pixdit_core/pixeldit_c2i.py, pixdit_core/pixeldit_t2i.py.
License: NVIDIA Source Code License (NSCLv1), see licenses/PixelDiT_LICENSE.txt.

Class, attribute and weight names are unchanged, so the released checkpoint loads as-is. Two stages:
    1. Patch stage: 16x16 pixel patches become tokens and go through MM-DiT blocks together with the text tokens.
    2. Pixel stage: PiT blocks refine every pixel, conditioned on its patch token; patches attend each other.

Mix-and-Match changes (marked "Mix-and-Match:"): the forward takes a stack of S images and P prompts and runs them
as ONE sequence. All images reuse the same RoPE positions, so every tile sees itself at its region's place in the
image. Attention is masked by an attention.TileAttentionLayout, and the region keys/values of a cross-region rule
are appended here. A layout sorts every image's patches by region (undone before the output) and keeps the shared
region's patches once for all images.
"""

import json
import math

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from .attention import attention, keep_positions, keep_shared_once, restore_all_images
from .paths import WEIGHTS_DIR

CHECKPOINT_REPO_ID = "nvidia/PixelDiT-1300M-1024px"
CHECKPOINT_FILENAME = "pixeldit_t2i_v1.pth"
MODEL_CONFIG_FILENAME = "config.json"
ARCHITECTURE_KEYS = (
    "in_channels", "patch_size", "num_groups", "hidden_size", "pixel_hidden_size", "pixel_attn_hidden_size",
    "pixel_num_groups", "patch_depth", "pixel_depth", "txt_embed_dim", "txt_max_length", "use_text_rope",
    "text_rope_theta", "use_pixel_abs_pos",
)
CHECKPOINT_WRAPPER_PREFIX = "core."  # added by PixelDiT's training wrapper
CHECKPOINT_TRAINING_ONLY_PREFIX = "_repa_projector."


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_adaln(x, shift, scale):
    return x * (1 + scale) + shift


def precompute_freqs_cis_2d(dim, height, width, theta=10000.0, scale=16.0):
    """[height * width, dim // 2] complex 2D RoPE frequencies; positions span [0, scale] whatever the grid size."""
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def apply_rotary_emb(xq, xk, freqs_cis):
    freqs_cis = freqs_cis[None, :, None, :]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def region_keys_values(keys, values, layout):
    """
    Mix-and-Match: the region keys and values a tile reads outside its own region. Copy s carries tile s of every
    region and all copies share one patch order, so a patch's alternatives sit tile_patch_count tokens apart.

    Args:
        keys, values: [batch, image tokens, heads, head dim], after norm and RoPE.
        layout: TileAttentionLayout with region keys.

    Returns:
        (region keys, region values), each [batch, tile patches, heads, head dim]: the average over the k copies
        ("average"), or the patch of one drawn alternative per region ("sampled_tile").
    """
    count = layout.tile_patch_count
    if layout.region_keys == "sampled_tile":
        drawn = layout.sampling.draw(layout.num_regions, layout.num_tiles)
        positions = drawn[layout.tile_patch_regions] * count + torch.arange(count, device=keys.device)
        return keys[:, positions], values[:, positions]

    def average(tokens):
        tiles = tokens[:, :layout.num_tiles * count]
        return tiles.view(tiles.shape[0], layout.num_tiles, count, *tokens.shape[2:]).mean(dim=1)

    return average(keys), average(values)


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half)
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq.to(next(self.mlp.parameters()).dtype))


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class FeedForward(nn.Module):
    """SwiGLU feed-forward block of the patch stage."""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    """GELU MLP of the pixel stage."""

    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x):
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans=3, embed_dim=768, norm_layer=None, bias=True):
        super().__init__()
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        return self.norm(self.proj(x))


class PixelTokenEmbedder(nn.Module):
    """Embeds every pixel and groups the pixels by patch for the pixel stage."""

    def __init__(self, in_channels, hidden_size_output, use_pixel_abs_pos=True):
        super().__init__()
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(int(in_channels), self.hidden_size_output, bias=True)
        self._pos_cache = dict()

    def _fetch_pixel_pos_image(self, height, width, device, dtype):
        key = ("image", height, width)
        if key not in self._pos_cache:
            grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            self._pos_cache[key] = torch.from_numpy(get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid))
        return self._pos_cache[key].to(device=device, dtype=dtype)

    def forward(self, inputs, img_height, img_width, patch_size):
        """[N, channels, H, W] images -> [N * patches, patch_size ** 2, hidden] pixel features, grouped by patch."""
        B, C, H, W = inputs.shape
        Hs, Ws = H // patch_size, W // patch_size
        x = self.proj(inputs.permute(0, 2, 3, 1).contiguous())
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(H, W, inputs.device, inputs.dtype)
            x = x + pos_full.view(H, W, self.hidden_size_output).unsqueeze(0)
        x = x.view(B, Hs, patch_size, Ws, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B * Hs * Ws, patch_size * patch_size, self.hidden_size_output)


class RotaryAttention(nn.Module):
    """Self-attention with RoPE and query/key RMS norms, between compressed patches in the pixel stage."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, pos, layout=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = self.q_norm(qkv[0]), self.k_norm(qkv[1]), qkv[2]
        q, k = apply_rotary_emb(q, k, freqs_cis=pos)
        # Mix-and-Match: region keys/values and the tile mask.
        if layout is not None and layout.region_keys is not None:
            region_k, region_v = region_keys_values(k, v, layout)
            k, v = torch.cat([k, region_k], dim=1), torch.cat([v, region_v], dim=1)
        block_mask = layout.pixel_block_mask if layout is not None else None
        x = attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class PiTBlock(nn.Module):
    """Pixel-stage block: per-pixel modulation from the patch token, and attention between compressed patches."""

    def __init__(self, pixel_hidden_size, patch_hidden_size, patch_size, num_heads, mlp_ratio=4.0,
                 attn_hidden_size=None, attn_num_heads=None, rope_fn=None):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else int(patch_hidden_size)
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        p2 = int(patch_size) ** 2
        self.compress_to_attn = nn.Linear(p2 * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, p2 * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio)
        self.adaLN_modulation = nn.Sequential(nn.Linear(int(patch_hidden_size), 6 * self.pixel_dim * p2, bias=True))
        self._pos_cache = dict()
        self._rope_fn = rope_fn if rope_fn is not None else precompute_freqs_cis_2d

    def _fetch_pos(self, height, width, device):
        key = (height, width)
        if key not in self._pos_cache:
            self._pos_cache[key] = self._rope_fn(self.attn_dim // self.num_heads, height, width)
        return self._pos_cache[key].to(device)

    def forward(self, x, s_cond, image_height, image_width, patch_size, num_images=1, layout=None):
        BL, P2, C = x.shape
        Hs, Ws = image_height // patch_size, image_width // patch_size
        # Mix-and-Match: all images form one sequence, with the shared region's patches kept once.
        pos_comp = self._fetch_pos(Hs, Ws, x.device)
        if layout is None:
            sequence_length = num_images * Hs * Ws
            pos_comp = pos_comp.repeat(num_images, 1)
        else:
            if layout.patch_order is not None:
                pos_comp = pos_comp[layout.patch_order]
            pos_comp = keep_positions(pos_comp, layout, num_images)
            sequence_length = pos_comp.shape[0]
        B = BL // sequence_length
        cond_params = self.adaLN_modulation(s_cond).view(BL, P2, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_comp = self.compress_to_attn(x_norm.view(BL, P2 * self.pixel_dim)).view(B, sequence_length, self.attn_dim)
        attn_out = self.attn(x_comp, pos_comp, layout)
        attn_exp = self.expand_from_attn(attn_out.view(BL, self.attn_dim)).view(BL, P2, self.pixel_dim)
        x = x + gate_msa * attn_exp
        return x + gate_mlp * self.mlp(apply_adaln(self.norm2(x), shift_mlp, scale_mlp))


class MMDiTJointAttention(nn.Module):
    """Joint attention over text and image tokens with separate projections per modality."""

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def forward(self, x, y, pos_img, pos_txt=None, layout=None):
        B, Nx, C = x.shape
        Ny = y.shape[1]
        qkv_x = self.qkv_x(x).reshape(B, Nx, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qx, kx, vx = self.q_norm_x(qkv_x[0]), self.k_norm_x(qkv_x[1]), qkv_x[2]
        qkv_y = self.qkv_y(y).reshape(B, Ny, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        qy, ky, vy = self.q_norm_y(qkv_y[0]), self.k_norm_y(qkv_y[1]), qkv_y[2]
        qx, kx = apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        # Token order is [text, image]; Mix-and-Match: region keys/values go last, and the tile mask is used.
        q_joint = torch.cat([qy, qx], dim=1)
        k_joint = torch.cat([ky, kx], dim=1)
        v_joint = torch.cat([vy, vx], dim=1)
        if layout is not None and layout.region_keys is not None:
            region_k, region_v = region_keys_values(kx, vx, layout)
            k_joint, v_joint = torch.cat([k_joint, region_k], dim=1), torch.cat([v_joint, region_v], dim=1)
        block_mask = layout.joint_block_mask if layout is not None else None
        out_joint = attention(q_joint.transpose(1, 2), k_joint.transpose(1, 2), v_joint.transpose(1, 2), block_mask)
        out_joint = out_joint.transpose(1, 2)
        out_y = out_joint[:, :Ny].reshape(B, Ny, C)
        out_x = out_joint[:, Ny:].reshape(B, Nx, C)
        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    """Patch-stage MM-DiT block: joint attention and per-modality MLPs, modulated by the timestep."""

    def __init__(self, hidden_size, groups, mlp_ratio=4.0):
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, layout=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(6, dim=-1)
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(6, dim=-1)
        x_norm = apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, layout)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y
        x = x + gate_mlp_x * self.mlp_x(apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixDiT_T2I(nn.Module):
    """PixelDiT text-to-image model: predicts the flow velocity (noise - image) of pixel-space images."""

    def __init__(self, in_channels=3, num_groups=16, hidden_size=1152, pixel_hidden_size=64,
                 pixel_attn_hidden_size=None, pixel_num_groups=None, patch_depth=26, pixel_depth=2, patch_size=16,
                 txt_embed_dim=4096, txt_max_length=1024, use_text_rope=True, text_rope_theta=10000.0,
                 use_pixel_abs_pos=True):
        super().__init__()
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_size = int(patch_size)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)

        self.pixel_embedder = PixelTokenEmbedder(in_channels, pixel_hidden_size, use_pixel_abs_pos=use_pixel_abs_pos)
        self.s_embedder = PatchTokenEmbedder(in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = PatchTokenEmbedder(int(txt_embed_dim), self.hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, self.hidden_size))
        self.patch_blocks = nn.ModuleList([MMDiTBlockT2I(self.hidden_size, self.num_groups) for _ in range(int(patch_depth))])
        pixel_attn_hidden_size = int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList([
            PiTBlock(int(pixel_hidden_size), self.hidden_size, patch_size=self.patch_size, num_heads=self.num_groups,
                     mlp_ratio=4.0, attn_hidden_size=pixel_attn_hidden_size, attn_num_heads=pixel_num_groups,
                     rope_fn=precompute_freqs_cis_2d)
            for _ in range(int(pixel_depth))
        ])
        self.final_layer = FinalLayer(int(pixel_hidden_size), self.out_channels)
        self.precompute_pos = dict()
        self.precompute_pos_txt = dict()

    def fetch_pos(self, height, width, device):
        if (height, width) not in self.precompute_pos:
            self.precompute_pos[(height, width)] = precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width)
        return self.precompute_pos[(height, width)].to(device)

    def fetch_pos_text(self, length, device):
        if length not in self.precompute_pos_txt:
            head_dim = self.hidden_size // self.num_groups
            freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
            angles = torch.arange(0, length).float().unsqueeze(1) * freqs.unsqueeze(0)
            self.precompute_pos_txt[length] = torch.polar(torch.ones_like(angles), angles)
        return self.precompute_pos_txt[length].to(device)

    def forward(self, images, timestep, text_embeds, layout=None):
        """
        Mix-and-Match: runs a stack of S images and P prompts as ONE sequence per batch element.

        Args:
            images: [batch, S, 3, H, W] noisy images; copy s carries tile s of every region.
            timestep: [batch] timesteps in [0, 1000].
            text_embeds: [batch, P, prompt length, text dim]; prompt 0 is the global prompt, prompt 1 + region * k +
                tile that tile's regional prompt.
            layout: TileAttentionLayout, or None for the original, unmasked model.

        Returns:
            [batch, S, 3, H, W] predicted flow velocity.
        """
        B, S, _, H, W = images.shape
        P = text_embeds.shape[1]
        Hs, Ws = H // self.patch_size, W // self.patch_size
        L = Hs * Ws
        patch_order = layout.patch_order if layout is not None else None

        # Mix-and-Match: patchify every image and join them into one sequence; all images reuse the same positions.
        flat_images = images.reshape(B * S, *images.shape[2:])
        x_patches = torch.nn.functional.unfold(flat_images, kernel_size=self.patch_size, stride=self.patch_size)
        x_patches = x_patches.transpose(1, 2)
        pos = self.fetch_pos(Hs, Ws, images.device)
        if patch_order is not None:
            x_patches, pos = x_patches[:, patch_order], pos[patch_order]
        if layout is None:
            x_patches, pos = x_patches.reshape(B, S * L, -1), pos.repeat(S, 1)
        else:
            x_patches = keep_shared_once(x_patches, layout, S)
            pos = keep_positions(pos, layout, S)

        t_emb = self.t_embedder(timestep.view(-1)).view(B, -1, self.hidden_size)
        condition = torch.nn.functional.silu(t_emb)

        # Mix-and-Match: every prompt gets its own learned positions and text RoPE, as if it were alone.
        text_length = min(text_embeds.shape[2], self.txt_max_length)
        y_emb = self.y_embedder(text_embeds[:, :, :text_length]) + self.y_pos_embedding[:, :text_length].to(images.dtype)
        y_emb = y_emb.reshape(B, P * text_length, self.hidden_size)
        pos_txt = self.fetch_pos_text(text_length, images.device).repeat(P, 1) if self.use_text_rope else None

        s = self.s_embedder(x_patches)
        for block in self.patch_blocks:
            s, y_emb = block(s, y_emb, condition, pos, pos_txt, layout)
        s = torch.nn.functional.silu(t_emb + s)

        sequence_length = s.shape[1]
        s_cond = s.reshape(B * sequence_length, self.hidden_size)
        P2 = self.patch_size * self.patch_size
        x_pixels = self.pixel_embedder(flat_images, img_height=H, img_width=W, patch_size=self.patch_size)
        x_pixels = x_pixels.view(B * S, L, P2, -1)
        if patch_order is not None:
            x_pixels = x_pixels[:, patch_order]
        if layout is not None:
            x_pixels = keep_shared_once(x_pixels, layout, S)
        x_pixels = x_pixels.reshape(B * sequence_length, P2, -1)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, H, W, self.patch_size, num_images=S, layout=layout)

        # Mix-and-Match: every image gets its tile patches back and a copy of the shared region.
        x_pixels = self.final_layer(x_pixels).view(B, sequence_length, P2, self.out_channels)
        x_pixels = (restore_all_images(x_pixels, layout, S) if layout is not None
                    else x_pixels.view(B * S, L, P2, self.out_channels))
        if patch_order is not None:
            x_pixels = x_pixels[:, layout.patch_restore]
        x_pixels = x_pixels.permute(0, 3, 2, 1).reshape(B * S, self.out_channels * P2, L)
        x_img = torch.nn.functional.fold(x_pixels, (H, W), kernel_size=self.patch_size, stride=self.patch_size)
        return x_img.view(B, S, self.out_channels, H, W)


def load_pixeldit(device, dtype):
    """
    Builds PixDiT_T2I from the released checkpoint (downloaded into WEIGHTS_DIR once). The model is built on the
    meta device and takes the memory-mapped checkpoint's tensors directly, so the weights are never held twice.

    Returns:
        (model in eval mode, the repository's config dict; its "scheduler" entry holds the flow shift).
    """
    for filename in (MODEL_CONFIG_FILENAME, CHECKPOINT_FILENAME):
        if not (WEIGHTS_DIR / filename).exists():
            print(f"Downloading {CHECKPOINT_REPO_ID}/{filename} to {WEIGHTS_DIR} (once)...")
            hf_hub_download(CHECKPOINT_REPO_ID, filename, local_dir=WEIGHTS_DIR)
    model_config = json.loads((WEIGHTS_DIR / MODEL_CONFIG_FILENAME).read_text())
    with torch.device("meta"):
        model = PixDiT_T2I(**{key: model_config[key] for key in ARCHITECTURE_KEYS})
    checkpoint = torch.load(WEIGHTS_DIR / CHECKPOINT_FILENAME, map_location="cpu", weights_only=False, mmap=True)
    state_dict = {key.removeprefix(CHECKPOINT_WRAPPER_PREFIX): value
                  for key, value in checkpoint["state_dict"].items()
                  if not key.startswith(CHECKPOINT_TRAINING_ONLY_PREFIX)}
    model.load_state_dict(state_dict, strict=True, assign=True)
    return model.to(device=device, dtype=dtype).eval(), model_config
