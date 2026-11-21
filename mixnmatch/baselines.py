"""
The baselines of the paper (Sec. 4.1), on the same backbone, sampler, noise, prompts and guidance, and on the
layout of a finished Mix-and-Match run of the same specification.

    independent         one plain image per regional prompt (and one of the global prompt for the shared
                        region); tile (i, j) is cut from regional prompt (i, j)'s image.
    single              one plain image of the global prompt, every tile cut from it.
    per_combination     one plain image per combination, of the global prompt expanded with its alternatives.
    multidiffusion      MultiDiffusion (Bar-Tal et al., 2023): every region's prompt (the global prompt expanded with
                        the region's alternative; the global prompt on the shared region) predicts the velocity of
                        the whole image, and the predictions are joined through the region masks. For the first
                        MULTIDIFFUSION_BOOTSTRAP of the steps a region sees the image only inside itself, with a
                        noised random constant color outside (MultiDiffusion's bootstrapping).
    extended_attention  shared self-attention as in StyleAligned and ConsiStory: a reference image of the global
                        prompt expanded with the category words, and one image per combination, all from the same
                        noise and run as one sequence in which every target image's patches also attend the
                        reference's patches (patch stage; text and pixel stage stay per image). At most k targets
                        share a sequence.

A velocity predicted from a regional prompt is guided with tile_cfg_scale, every other with cfg_scale. The
neighbor negative prompts are part of Mix-and-Match and are not given to the baselines.
"""

import numpy as np
import torch
from tqdm.auto import tqdm

from .attention import TileAttentionLayout, chunked_block_mask
from .config import PATCH_SIZE
from .pipeline import MODEL_DTYPE, NUM_TRAIN_TIMESTEPS, MixNMatchPipeline, to_display_image
from .prompts import PROMPT_LENGTH

METHODS = ("independent", "single", "per_combination", "multidiffusion", "extended_attention")
EQUAL_COST_ONLY = ("independent", "single")  # their combinations are their tiles assembled
MULTIDIFFUSION_BOOTSTRAP = 0.2  # share of the steps MultiDiffusion bootstraps its regions for
MAX_BATCH = 8  # images per transformer call when images are denoised independently


# Every prompt a baseline reads is a (positive, negative) pair; the negative prompt is always empty.


def global_prompt(config):
    return config.styled(config.prompt), ""


def tile_prompt(config, region, tile):
    """A regional prompt on its own (the independent baseline's image of one tile)."""
    return config.styled(config.regions[region].tiles[tile]), ""


def composed_prompt(config, choice):
    """The global prompt expanded with one alternative per region (one combination's description)."""
    return config.styled(config.variant_prompt(choice)), ""


def reference_prompt(config):
    """The global prompt expanded with the category words (extended attention's reference image)."""
    return config.styled(config.category_prompt()), ""


def region_prompt(config, region, tile):
    """The global prompt expanded with one region's alternative alone (MultiDiffusion's regional prompt)."""
    return config.styled(config.region_prompt(region, tile)), ""


def region_prompts(config, region_map, choice):
    """(labels present in the layout, a prompt per label): the global prompt for the shared region (label n)."""
    labels = [int(label) for label in np.unique(region_map)]
    return labels, [global_prompt(config) if label == config.num_regions else region_prompt(config, label, choice[label])
                    for label in labels]


def method_prompts(config, method, choices, region_map):
    """Every (positive, negative) prompt pair a method reads."""
    if method == "independent":
        return [tile_prompt(config, region, tile) for region in range(config.num_regions)
                for tile in range(config.num_tiles)] + [global_prompt(config)]
    if method == "single":
        return [global_prompt(config)]
    if method == "multidiffusion":
        return [prompt for choice in choices for prompt in region_prompts(config, region_map, choice)[1]]
    composed = [composed_prompt(config, choice) for choice in choices]
    return composed if method == "per_combination" else [reference_prompt(config)] + composed


def extended_attention_layout(num_images, prompt_length, patch_count, device):
    """num_images prompts then num_images images (image 0 the reference): every token attends its own prompt and
    image, and a target image's patches also attend the reference's patches in the patch stage."""
    image_owners = torch.arange(num_images, device=device).repeat_interleave(patch_count)
    owners = torch.cat([torch.arange(num_images, device=device).repeat_interleave(prompt_length), image_owners])
    is_image = torch.arange(len(owners), device=device) >= num_images * prompt_length

    def joint_mask(batch, head, query_index, key_index):
        query_owner, key_owner = owners[query_index], owners[key_index]
        sharing = is_image[query_index] & is_image[key_index] & (key_owner == 0) & (query_owner != 0)
        return (query_owner == key_owner) | sharing

    def pixel_mask(batch, head, query_index, key_index):
        return image_owners[query_index] == image_owners[key_index]

    return TileAttentionLayout(num_images, chunked_block_mask(joint_mask, len(owners), len(owners), device),
                               chunked_block_mask(pixel_mask, len(image_owners), len(image_owners), device),
                               tile_patch_count=patch_count)


class BaselinePipeline(MixNMatchPipeline):
    """Runs the five baselines on Mix-and-Match's backbone; `run` dispatches on the method name."""

    def __init__(self, transformer, scheduler, device):
        super().__init__(transformer, scheduler, device)
        self.image_passes = 0  # images the transformer ran on (both guidance passes count): the methods' cost

    def denoise(self, config, num_images, velocity):
        """num_images images from the seed noise; velocity(sample, sigma, step) -> guided velocity."""
        generator = torch.Generator(device=self.device).manual_seed(config.seed)
        noise = torch.randn((1, 3, config.height, config.width), generator=generator, device=self.device)
        sample = noise.repeat(num_images, 1, 1, 1)
        self.reset_scheduler(config)
        for step, timestep in enumerate(tqdm(self.scheduler.timesteps)):
            sigma = self.scheduler.sigmas[step].item()
            sample = self.scheduler.step(velocity(sample, sigma, step), timestep, sample).prev_sample
        return torch.stack([to_display_image(image) for image in sample])

    def independent_velocity(self, images, sigma, text, scales):
        """Guided velocity of images denoised independently, image m with prompt m and scale scales[m]."""
        velocities = []
        for start in range(0, len(images), MAX_BATCH):
            part = images[start:start + MAX_BATCH].unsqueeze(1).to(MODEL_DTYPE)
            embeds = text[:, start:start + MAX_BATCH].unsqueeze(2)
            model_time = torch.full((len(part),), sigma * NUM_TRAIN_TIMESTEPS, device=images.device, dtype=MODEL_DTYPE)
            negative = self.transformer(part, model_time, embeds[0])[:, 0].float()
            positive = self.transformer(part, model_time, embeds[1])[:, 0].float()
            velocities.append(negative + scales[start:start + MAX_BATCH].view(-1, 1, 1, 1) * (positive - negative))
            self.image_passes += 2 * len(part)
        return torch.cat(velocities)

    def stack(self, embeds, prompts):
        """[2, len(prompts), PROMPT_LENGTH, text dim] (negative, positive) text of the given prompt pairs."""
        return torch.cat([embeds[prompt] for prompt in prompts], dim=1).to(device=self.device, dtype=MODEL_DTYPE)

    @torch.inference_mode()
    def plain(self, config, prompts, embeds):
        """One plain image per prompt pair (independent, single, per_combination)."""
        text = self.stack(embeds, prompts)
        tile_prompts = {tile_prompt(config, region, tile) for region in range(config.num_regions)
                        for tile in range(config.num_tiles)}
        scales = torch.tensor([config.tile_cfg_scale if prompt in tile_prompts else config.cfg_scale
                               for prompt in prompts], device=self.device)
        return self.denoise(config, len(prompts),
                            lambda sample, sigma, step: self.independent_velocity(sample, sigma, text, scales))

    @torch.inference_mode()
    def multidiffusion(self, config, region_map, choices, embeds):
        """One MultiDiffusion image per combination."""
        device = self.device
        pixels = torch.as_tensor(np.asarray(region_map).repeat(PATCH_SIZE, 0).repeat(PATCH_SIZE, 1), device=device)
        labels = region_prompts(config, region_map, choices[0])[0]
        masks = torch.stack([pixels == label for label in labels]).float()[None, :, None]  # [1, R, 1, H, W]
        bootstrapped = torch.tensor([label != config.num_regions for label in labels], device=device)
        text = self.stack(embeds, [prompt for choice in choices
                                   for prompt in region_prompts(config, region_map, choice)[1]])
        scales = torch.tensor([config.tile_cfg_scale if label != config.num_regions else config.cfg_scale
                               for label in labels], device=device).repeat(len(choices))
        bootstrap_steps = round(MULTIDIFFUSION_BOOTSTRAP * config.steps)
        generator = torch.Generator(device=device).manual_seed(config.seed)
        num_labels = len(labels)

        def velocity(sample, sigma, step):
            inputs = sample[:, None].expand(-1, num_labels, -1, -1, -1)
            if step < bootstrap_steps:
                color = torch.rand((*inputs.shape[:3], 1, 1), generator=generator, device=device) * 2 - 1
                filler = (1 - sigma) * color + sigma * torch.randn(inputs.shape, generator=generator, device=device)
                inside = masks + (1 - masks) * (~bootstrapped).float().view(1, -1, 1, 1, 1)
                inputs = inside * inputs + (1 - inside) * filler
            region_velocities = self.independent_velocity(inputs.flatten(0, 1), sigma, text, scales)
            return (region_velocities.view(*inputs.shape) * masks).sum(dim=1)

        return self.denoise(config, len(choices), velocity)

    @torch.inference_mode()
    def extended_attention(self, config, choices, embeds):
        """
        Returns:
            ([len(choices), 3, H, W] target images, [3, H, W] reference image), in [0, 1].
        """
        patch_count = (config.height // PATCH_SIZE) * (config.width // PATCH_SIZE)
        targets, reference = [], None
        for start in range(0, len(choices), config.num_tiles):
            prompts = [reference_prompt(config)] + [composed_prompt(config, choice)
                                                    for choice in choices[start:start + config.num_tiles]]
            text = self.stack(embeds, prompts)
            layout = extended_attention_layout(len(prompts), PROMPT_LENGTH, patch_count, self.device)

            def velocity(sample, sigma, step, text=text, layout=layout):
                self.image_passes += 2 * len(sample)
                return self.velocity(sample, sigma, text, config.cfg_scale, layout)

            images = self.denoise(config, len(prompts), velocity)
            reference = images[0]
            targets.append(images[1:])
        return torch.cat(targets), reference

    def run(self, config, method, choices, region_map, embeds):
        """
        Args:
            embeds: encode_pairs output holding method_prompts(config, method, choices, region_map).

        Returns:
            (images, source, reference, generated): [N, 3, H, W] images; source(region, tile) -> the index of the
            image that tile is cut from (the shared region asks for (n, 0)); extended attention's reference (else
            None); the number of images generated.
        """
        n, k = config.num_regions, config.num_tiles
        if method == "independent":
            images = self.plain(config, method_prompts(config, method, choices, region_map), embeds)
            return images, lambda region, tile: len(images) - 1 if region == n else region * k + tile, None, len(images)
        reference = None
        if method in ("single", "per_combination"):
            images = self.plain(config, method_prompts(config, method, choices, region_map), embeds)
        elif method == "multidiffusion":
            images = self.multidiffusion(config, region_map, choices, embeds)
        else:
            images, reference = self.extended_attention(config, choices, embeds)
        generated = len(images) + (-(-len(images) // k) if reference is not None else 0)
        # Image j chose alternative j everywhere (equal-cost protocol); the shared region comes from image 0.
        return images, lambda region, tile: 0 if region == n or len(images) == 1 else tile, reference, generated
