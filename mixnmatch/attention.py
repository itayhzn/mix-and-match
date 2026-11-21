"""
The attention rules of the joint denoising (paper Sec. 3.6 and Tab. 6).

After the split, one transformer sequence holds, in this order:

    [global prompt | regional prompts (region-major) | tile tokens (k copies x tile patches) | shared-region patches
     | region keys]

Copy s of the image carries tile s of every region. The shared region's patches appear ONCE (taken from copy 0)
and the model copies their result back into every copy, so every combination has the same shared region.

"attention" sets what a tile sees of the regions it does not belong to:

    "average"       the region average: the keys and values of the k alternatives, averaged (the paper's rule);
    "sampled_tile"  one alternative, drawn uniformly and redrawn at every layer (ablation);
    "all_tiles"     all k alternatives at once (ablation);
    "none"          nothing (ablation).

The first two read the region keys, which exist only for them and are appended to the keys and values (never to
the queries). Who may attend whom (query -> key):

    query \\ key        | global | own regional | own  | sibling | tile of      | region key | shared
                        | prompt | prompt       | tile | tile    | other region | of other   | region
    global prompt       |  yes   |     no       |  no  |   no    |     no       |    no      |  yes
    regional prompt     |  no    |     yes      |  yes |   no    |     no       |    no      |  no
    tile                |  yes   |     yes      |  yes |   no    | "all_tiles"  | avg/sampled|  yes
    shared region       |  yes   |     no       |  no  |   no    | "all_tiles"  | avg/sampled|  yes

The pixel stage has no text and uses the same rule on its image tokens and region keys.

Every image's patches are sorted by region once at the model input (and put back at its output), so that every
(region, copy) group is contiguous and flex attention can skip whole blocks; the shared region's patches sort last.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention

GLOBAL_PROMPT_TOKEN = 0
REGIONAL_PROMPT_TOKEN = 1
TILE_TOKEN = 2
REGION_KEY_TOKEN = 3  # key-only: one region's alternatives at one patch, averaged or drawn
SHARED_TOKEN = 4  # a patch of the shared region, kept once for all k copies
NO_INDEX = -1
NO_TOKEN_KIND = -1  # a kind no key carries: the "none" rule

REGION_KEY_RULES = ("average", "sampled_tile")

compiled_flex_attention = torch.compile(flex_attention)
compiled_create_block_mask = torch.compile(create_block_mask)

BLOCK_SIZE = 128  # flex attention's block size
CHUNK_QUERY_BLOCKS = 8  # query blocks chunked_block_mask evaluates at once


@dataclass
class RegionSampling:
    """
    The draws of the "sampled_tile" rule. Both guidance passes of a step see the same draws: the pipeline takes one
    seed per step (next_step_seed) and restarts the per-layer draws from it before each pass (begin_pass).
    """

    generator: torch.Generator
    base_seed: int
    step: int = 0

    def next_step_seed(self):
        self.step += 1
        return self.base_seed + self.step

    def begin_pass(self, step_seed):
        self.generator.manual_seed(step_seed)

    def draw(self, num_regions, num_tiles):
        return torch.randint(num_tiles, (num_regions,), generator=self.generator, device=self.generator.device)


@dataclass
class TileAttentionLayout:
    """Everything the transformer needs to run one sequence of several images under a mask."""

    num_tiles: int  # k, the number of copies (images) in the sample
    joint_block_mask: BlockMask  # patch stage: prompts + image tokens (+ region keys)
    pixel_block_mask: BlockMask  # pixel stage: image tokens (+ region keys)
    patch_order: torch.Tensor | None = None  # raster index of the patch at every position; None = raster order
    patch_restore: torch.Tensor | None = None  # inverse of patch_order
    tile_patch_count: int = 0  # patches per image outside the shared region; they come first in patch order
    shared_patch_count: int = 0  # patches of the shared region, kept once for all images
    region_keys: str | None = None  # "average" or "sampled_tile" when the sequence carries region keys
    num_regions: int = 0
    tile_patch_regions: torch.Tensor | None = None  # [tile patches] region of every tile patch, in patch order
    sampling: RegionSampling | None = None  # "sampled_tile" only


def tile_mask_mod(labels, other_region_kind):
    """
    The rule table above as a flex-attention mask function. Queries are the first positions of the key sequence
    (region keys exist only at its end), so one label tensor serves both.

    Args:
        labels: [3, key length] (kind, region, tile) of every key position.
        other_region_kind: the key kind a tile may read outside its own region: TILE_TOKEN ("all_tiles"),
            REGION_KEY_TOKEN ("average", "sampled_tile") or NO_TOKEN_KIND ("none").
    """
    kinds, regions, tiles = labels

    def mask_mod(batch, head, query_index, key_index):
        query_kind, key_kind = kinds[query_index], kinds[key_index]
        same_region = regions[query_index] == regions[key_index]
        same_tile = same_region & (tiles[query_index] == tiles[key_index])
        own_tile = same_tile & ((key_kind == REGIONAL_PROMPT_TOKEN) | (key_kind == TILE_TOKEN))
        global_query = (query_kind == GLOBAL_PROMPT_TOKEN) & (
            (key_kind == GLOBAL_PROMPT_TOKEN) | (key_kind == SHARED_TOKEN))
        regional_query = (query_kind == REGIONAL_PROMPT_TOKEN) & own_tile
        tile_query = (query_kind == TILE_TOKEN) & (
            (key_kind == GLOBAL_PROMPT_TOKEN) | own_tile | (key_kind == SHARED_TOKEN)
            | (~same_region & (key_kind == other_region_kind)))
        # The shared region sees the tile regions the way a tile sees another region.
        shared_query = (query_kind == SHARED_TOKEN) & (
            (key_kind == GLOBAL_PROMPT_TOKEN) | (key_kind == SHARED_TOKEN) | (key_kind == other_region_kind))
        return global_query | regional_query | tile_query | shared_query

    return mask_mod


def chunked_block_mask(mask_mod, query_length, key_length, device):
    """
    A BlockMask built without compiling the mask function and without the full mask in memory: the function runs
    on CHUNK_QUERY_BLOCKS query blocks at a time and only which blocks are full, partial or empty is kept. Used by
    the extended-attention baseline, whose sequence shapes change from library to library.
    """
    num_query_blocks = math.ceil(query_length / BLOCK_SIZE)
    num_key_blocks = math.ceil(key_length / BLOCK_SIZE)
    keys = torch.arange(key_length, device=device)[None, :]
    no_index = torch.zeros((), dtype=torch.long, device=device)
    any_allowed = torch.zeros((num_query_blocks, num_key_blocks), dtype=torch.bool, device=device)
    all_allowed = torch.zeros_like(any_allowed)
    for first_block in range(0, num_query_blocks, CHUNK_QUERY_BLOCKS):
        blocks = min(CHUNK_QUERY_BLOCKS, num_query_blocks - first_block)
        start = first_block * BLOCK_SIZE
        stop = min(start + blocks * BLOCK_SIZE, query_length)
        queries = torch.arange(start, stop, device=device)[:, None]
        allowed = mask_mod(no_index, no_index, queries, keys).expand(stop - start, key_length)
        padded = torch.zeros((blocks * BLOCK_SIZE, num_key_blocks * BLOCK_SIZE), dtype=torch.bool, device=device)
        padded[:stop - start, :key_length] = allowed
        tiles = padded.view(blocks, BLOCK_SIZE, num_key_blocks, BLOCK_SIZE)
        any_allowed[first_block:first_block + blocks] = tiles.any(dim=3).any(dim=1)
        all_allowed[first_block:first_block + blocks] = tiles.all(dim=3).all(dim=1)
    partial = any_allowed & ~all_allowed

    def as_indices(selected):
        counts = selected.sum(dim=1).to(torch.int32)
        order = torch.argsort((~selected).to(torch.int8), dim=1, stable=True).to(torch.int32)
        return counts[None, None], order[None, None]

    return BlockMask.from_kv_blocks(*as_indices(partial), *as_indices(all_allowed), BLOCK_SIZE=BLOCK_SIZE,
                                    mask_mod=mask_mod, seq_lengths=(query_length, key_length))


def build_token_labels(tile_patch_regions, shared_patch_count, num_regions, num_tiles, prompt_length,
                       has_region_keys):
    """
    Labels every position of the patch-stage (joint) and pixel-stage sequences with (kind, region, tile).

    Returns:
        ([3, joint key length], [3, pixel key length]) integer tensors.
    """
    device = tile_patch_regions.device
    tile_patch_count = tile_patch_regions.numel()
    # Prompt 0 is the global prompt, prompt 1 + region * k + tile that tile's regional prompt.
    prompt_indices = torch.arange(1 + num_regions * num_tiles, device=device).repeat_interleave(prompt_length)
    regional = prompt_indices > 0
    text_labels = torch.stack([
        torch.where(regional, REGIONAL_PROMPT_TOKEN, GLOBAL_PROMPT_TOKEN),
        torch.where(regional, (prompt_indices - 1) // num_tiles, NO_INDEX),
        torch.where(regional, (prompt_indices - 1) % num_tiles, NO_INDEX),
    ])
    image_labels = torch.stack([
        torch.full((num_tiles * tile_patch_count,), TILE_TOKEN, device=device),
        tile_patch_regions.repeat(num_tiles),
        torch.arange(num_tiles, device=device).repeat_interleave(tile_patch_count),
    ])
    if shared_patch_count:
        image_labels = torch.cat([image_labels, torch.stack([
            torch.full((shared_patch_count,), SHARED_TOKEN, device=device),
            torch.full((shared_patch_count,), NO_INDEX, device=device),
            torch.full((shared_patch_count,), NO_INDEX, device=device),
        ])], dim=1)
    if has_region_keys:
        image_labels = torch.cat([image_labels, torch.stack([
            torch.full((tile_patch_count,), REGION_KEY_TOKEN, device=device),
            tile_patch_regions,
            torch.full((tile_patch_count,), NO_INDEX, device=device),
        ])], dim=1)
    return torch.cat([text_labels, image_labels], dim=1), image_labels


def build_tile_attention_layout(region_map, num_regions, num_tiles, prompt_length, rule, seed, device):
    """
    Args:
        region_map: [patch rows, patch cols] region index of every patch; num_regions marks the shared region.
        num_regions: n.
        num_tiles: k.
        prompt_length: tokens per prompt.
        rule: one of config.ATTENTION_RULES.
        seed: the library's seed, which the "sampled_tile" draws are counted from.
        device: torch device.

    Returns:
        TileAttentionLayout.
    """
    patch_regions = torch.as_tensor(region_map, device=device).flatten()
    shared_patch_count = int((patch_regions == num_regions).sum())
    patch_order = torch.argsort(patch_regions, stable=True)
    if torch.equal(patch_order, torch.arange(len(patch_order), device=device)):
        patch_order, patch_restore = None, None
    else:
        patch_restore = torch.argsort(patch_order)
        patch_regions = patch_regions[patch_order]
    tile_patch_count = patch_regions.numel() - shared_patch_count

    region_keys = rule if rule in REGION_KEY_RULES else None
    joint_labels, pixel_labels = build_token_labels(patch_regions[:tile_patch_count], shared_patch_count,
                                                    num_regions, num_tiles, prompt_length, region_keys is not None)
    other_region_kind = {"average": REGION_KEY_TOKEN, "sampled_tile": REGION_KEY_TOKEN, "all_tiles": TILE_TOKEN,
                         "none": NO_TOKEN_KIND}[rule]
    num_region_keys = tile_patch_count if region_keys is not None else 0
    joint_mask, pixel_mask = (
        compiled_create_block_mask(tile_mask_mod(labels, other_region_kind), None, None,
                                   labels.shape[1] - num_region_keys, labels.shape[1], device=device)
        for labels in (joint_labels, pixel_labels))
    sampling = RegionSampling(torch.Generator(device=device), seed) if rule == "sampled_tile" else None
    return TileAttentionLayout(num_tiles, joint_mask, pixel_mask, patch_order, patch_restore, tile_patch_count,
                               shared_patch_count, region_keys, num_regions, patch_regions[:tile_patch_count],
                               sampling)


def keep_shared_once(tokens, layout, num_images):
    """[batch * images, patches, ...] tokens in patch order -> [batch, k * tile patches (+ shared patches), ...]."""
    per_image = tokens.view(tokens.shape[0] // num_images, num_images, *tokens.shape[1:])
    tiles = per_image[:, :, :layout.tile_patch_count].flatten(1, 2)
    if not layout.shared_patch_count:
        return tiles
    return torch.cat([tiles, per_image[:, 0, layout.tile_patch_count:]], dim=1)


def restore_all_images(tokens, layout, num_images):
    """Undoes keep_shared_once: every image gets its tile patches and a copy of the shared region."""
    batch, tile_count = tokens.shape[0], layout.tile_patch_count
    images = tokens[:, :num_images * tile_count].view(batch, num_images, tile_count, *tokens.shape[2:])
    if layout.shared_patch_count:
        shared = tokens[:, num_images * tile_count:]
        images = torch.cat([images, shared.unsqueeze(1).expand(batch, num_images, *shared.shape[1:])], dim=2)
    return images.flatten(0, 1)


def keep_positions(positions, layout, num_images):
    """The RoPE positions of keep_shared_once's sequence (positions in patch order)."""
    tiles = positions[:layout.tile_patch_count].repeat(num_images, 1)
    return torch.cat([tiles, positions[layout.tile_patch_count:]]) if layout.shared_patch_count else tiles


def attention(query, key, value, block_mask):
    """Multi-head attention, masked by a layout's block mask, or full attention (the original model) without one."""
    if block_mask is None:
        return F.scaled_dot_product_attention(query, key, value)
    return compiled_flex_attention(query, key, value, block_mask=block_mask)
