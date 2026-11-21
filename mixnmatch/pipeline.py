"""
Mix-and-Match: one run of PixelDiT that generates a whole tile library (paper Sec. 3 and Alg. 1).

    1. Shared prefix (steps 0 .. tau-1): one image, denoised under the global prompt expanded with the regions'
       category words. At step tau_b it forks into k branches (branch j: alternative j of every region, h_b steps
       ahead), SAM 3 segments every branch, and each region chooses its composition variant; the prefix resumes
       under the global prompt expanded with the variants.
    2. Layout (step tau-1): the prefix is run h steps ahead on a fork, SAM 3 segments the predicted clean image
       with every region's variant and category word, and the fork is undone; only the layout is kept.
    3. Joint denoising (steps tau .. T-1): the image is copied k times, copy j carrying alternative j of every
       region, and the copies run as one sequence under the attention rule of attention.py. Inside the tile
       regions the velocity is guided with w_t, and every tile's negative prompt holds its neighbors' regional
       prompts.

Sampling is PixelDiT's: DPM-Solver++ (order 2) on flow sigmas, classifier-free guidance with the negative and
positive passes run one after the other, a bfloat16 model and a float32 sample.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler
from tqdm.auto import tqdm

from . import segmentation
from .attention import build_tile_attention_layout
from .config import PATCH_SIZE
from .model import load_pixeldit
from .prompts import PROMPT_LENGTH, cache_path, encode

MODEL_DTYPE = torch.bfloat16
NUM_TRAIN_TIMESTEPS = 1000  # the model's time input is sigma * 1000


def to_display_image(image):
    """[3, H, W] model-space image -> [3, H, W] float CPU tensor in [0, 1]."""
    return ((image.float().clamp(-1, 1) + 1) / 2).cpu()


def pixel_map(patch_map, device=None):
    """[patch rows, patch cols] map -> [H, W] tensor, every patch repeated over its pixels."""
    return torch.from_numpy(np.kron(patch_map, np.ones((PATCH_SIZE, PATCH_SIZE), dtype=np.int64))).to(device)


@dataclass
class Library:
    """The result of one run."""

    images: torch.Tensor  # [k, 3, H, W] in [0, 1]: copy j holds tile j of every region
    region_map: np.ndarray  # [patch rows, patch cols] region of every patch, n for the shared region
    choice: list[int]  # every region's composition variant (all zeros without branches)
    segmented: torch.Tensor | None = None  # [3, H, W] the look-ahead image the layout was segmented on


class MixNMatchPipeline:
    """
    Runs specifications (config.Config) into libraries. Build one with `from_weights` and call it with a config.

    PixelDiT stays on the GPU; SAM 3 and Gemma are loaded when first needed and swapped in while PixelDiT waits on
    the CPU, so a single model occupies the GPU at any time.
    """

    def __init__(self, transformer, scheduler, device):
        self.transformer = transformer
        self.scheduler_config = scheduler.config  # every run gets a fresh scheduler from it
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.sam = None  # (Sam3Model, Sam3Processor), loaded on first use and parked on the CPU

    @classmethod
    def from_weights(cls, device="cuda"):
        """Downloads PixelDiT if needed and builds the pipeline with its scheduler settings."""
        transformer, model_config = load_pixeldit(device, MODEL_DTYPE)
        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN_TIMESTEPS, solver_order=2, algorithm_type="dpmsolver++",
            prediction_type="flow_prediction", use_flow_sigmas=True,
            flow_shift=model_config["scheduler"]["flow_shift"])
        return cls(transformer, scheduler, device)

    def reset_scheduler(self, config):
        """A fresh scheduler set to the config's number of steps."""
        self.scheduler = DPMSolverMultistepScheduler.from_config(self.scheduler_config)
        self.scheduler.set_timesteps(config.steps, device=self.device)

    # ----------------------------------------------------------------------------------------------------------
    # Model calls

    def velocity(self, sample, sigma, text, scale, layout=None):
        """
        Guided velocity v_- + w (v_+ - v_-).

        Args:
            sample: [S, 3, H, W] float32 images (one sequence).
            sigma: noise level.
            text: [2, prompts, PROMPT_LENGTH, text dim] (negative, positive).
            scale: guidance scale, a number or a [1, 1, H, W] map.
            layout: TileAttentionLayout after the split, None before it.
        """
        images = sample.unsqueeze(0).to(MODEL_DTYPE)
        # The exact time sigma * 1000 (scheduler.timesteps are rounded), in the model dtype.
        model_time = torch.full((1,), sigma * NUM_TRAIN_TIMESTEPS, device=sample.device, dtype=MODEL_DTYPE)
        # "sampled_tile": both passes of a step draw the same alternatives.
        sampling = None if layout is None else layout.sampling
        step_seed = None if sampling is None else sampling.next_step_seed()
        passes = []
        for half in (text[:1], text[1:]):
            if sampling is not None:
                sampling.begin_pass(step_seed)
            passes.append(self.transformer(images, model_time, half, layout)[0].float())
        negative, positive = passes
        return negative + scale * (positive - negative)

    def look_ahead(self, sample, velocity, step, text, steps, scale):
        """
        The predicted clean image `steps` steps later, on a fork that is undone (the run continues on its own path).

        Args:
            sample, velocity: [1, 3, H, W] image at `step` and its guided velocity.
            text: [2, 1, PROMPT_LENGTH, text dim] the prompt of the fork.
        """
        state = ([None if output is None else output.clone() for output in self.scheduler.model_outputs],
                 self.scheduler.lower_order_nums, self.scheduler._step_index, self.scheduler._begin_index)
        sigma = self.scheduler.sigmas[step].item()
        for offset in range(steps):
            sample = self.scheduler.step(velocity, self.scheduler.timesteps[step + offset], sample).prev_sample
            sigma = self.scheduler.sigmas[step + offset + 1].item()
            velocity = self.velocity(sample, sigma, text, scale)
        outputs, lower_order_nums, step_index, begin_index = state
        # Cloned again: scheduler.step writes into the list it is given.
        self.scheduler.model_outputs = [None if output is None else output.clone() for output in outputs]
        self.scheduler.lower_order_nums, self.scheduler._step_index = lower_order_nums, step_index
        self.scheduler._begin_index = begin_index
        return sample - sigma * velocity

    @contextmanager
    def transformer_offloaded(self):
        """PixelDiT waits on the CPU while another model (SAM 3, Gemma) uses the GPU."""
        self.transformer.to("cpu")
        torch.cuda.empty_cache()
        try:
            yield
        finally:
            self.transformer.to(self.device)

    def encode(self, prompts):
        """{(prompt, positive): [PROMPT_LENGTH, text dim]}, with PixelDiT off the GPU if Gemma has to run."""
        if all(cache_path(*key).exists() for key in prompts):
            return encode(prompts, self.device)
        with self.transformer_offloaded():
            return encode(prompts, self.device)

    def text(self, encoded, positives, negatives):
        """[2, prompts, PROMPT_LENGTH, text dim] on the device, negatives first."""
        return torch.stack([torch.stack([encoded[(prompt, False)] for prompt in negatives]),
                            torch.stack([encoded[(prompt, True)] for prompt in positives])]
                           ).to(device=self.device, dtype=MODEL_DTYPE)

    def segment(self, images, phrases):
        """SAM 3 maps of several images, each with its own phrases ([phrases, H, W] per image)."""
        if self.sam is None:
            self.sam = segmentation.load_sam3()
        model, processor = self.sam
        with self.transformer_offloaded():
            model.to(self.device)
            try:
                return [segmentation.segment_phrases(model, processor, image, image_phrases)
                        for image, image_phrases in zip(images, phrases, strict=True)]
            finally:
                model.to("cpu")
                torch.cuda.empty_cache()

    # ----------------------------------------------------------------------------------------------------------
    # The steps of a run

    def branch_and_choose(self, config, sample, step, branch_text):
        """
        Forks the prefix at step tau_b into k branches, segments each and chooses every region's composition variant.

        Returns:
            (choice, [n, k, patch rows, patch cols] branch maps: alternative j's mask measured in branch j).
        """
        n, k = config.num_regions, config.num_tiles
        sigma = self.scheduler.sigmas[step].item()
        images = []
        for branch in range(k):
            text = branch_text[:, branch:branch + 1]
            velocity = self.velocity(sample, sigma, text, config.cfg_scale)
            clean = self.look_ahead(sample, velocity, step, text, config.branch_lookahead, config.cfg_scale)
            images.append(to_display_image(clean[0]))
        # Branch j drew alternative j of every region: it is searched for those alternatives' prompts.
        phrases = [[config.regions[region].tiles[branch] for region in range(n)] for branch in range(k)]
        phrase_maps = torch.cat(self.segment(images, phrases))
        phrase_tiles = [region * k + branch for branch in range(k) for region in range(n)]
        variant_maps = segmentation.region_maps(phrase_maps, phrase_tiles, n, k, rescale=False)
        variant_maps = variant_maps.numpy().reshape(n, k, *variant_maps.shape[1:])
        for branch in range(k):  # branch j's regions are its alternatives j, checked against each other
            variant_maps[:, branch] = segmentation.keep_attached(variant_maps[:, branch])
        choice, scores = segmentation.choose_variants(variant_maps)
        for region in range(n):
            print(f"  region {region}: " + ", ".join(f"{tile!r} {score:.3g}" for tile, score in
                                                     zip(config.regions[region].tiles, scores[region]))
                  + f" -> {config.regions[region].tiles[choice[region]]!r}")
        return choice, variant_maps

    def sam_layout(self, config, clean, choice, variant_maps):
        """
        The layout of the look-ahead image: every region searched for its composition variant (and, with branches,
        its category word), then the assignment and clean-up rules of segmentation.py.

        Returns:
            [patch rows, patch cols] labels, n for the shared region.
        """
        n, k = config.num_regions, config.num_tiles
        phrases = [config.regions[region].tiles[choice[region]] for region in range(n)]
        phrase_tiles = [region * k + choice[region] for region in range(n)]
        if variant_maps is not None:  # with branches the category word is unioned in
            phrases = [phrase for region in range(n) for phrase in (phrases[region], config.regions[region].category)]
            phrase_tiles = [tile for tile in phrase_tiles for _ in range(2)]
        [phrase_maps] = self.segment([to_display_image(clean[0])], [phrases])
        tile_maps = segmentation.region_maps(phrase_maps, phrase_tiles, n, k, rescale=True)
        region_maps = tile_maps.reshape(n, k, *tile_maps.shape[1:]).amax(dim=1).numpy()
        labels = segmentation.assign(segmentation.keep_attached(region_maps))
        if variant_maps is not None:
            labels = segmentation.grow_into_variants(labels, variant_maps, choice)
        labels = segmentation.remove_small_components(labels)
        return segmentation.grow_by_margin(labels, n)

    def rectangles(self, config):
        """The "rectangles" ablation: one horizontal band per region, top to bottom, on whole patches."""
        rows, cols = config.height // PATCH_SIZE, config.width // PATCH_SIZE
        edges = [round(i * rows / config.num_regions) for i in range(config.num_regions + 1)]
        labels = np.zeros((rows, cols), dtype=np.int64)
        for region, (top, bottom) in enumerate(zip(edges, edges[1:])):
            labels[top:bottom] = region
        return labels

    def neighbor_negatives(self, config, text, labels):
        """Every tile's negative prompt becomes the regional prompts of the same alternative in the regions it
        touches (the shared region left out)."""
        n, k = config.num_regions, config.num_tiles
        pairs = np.concatenate([np.stack([labels[:, :-1].ravel(), labels[:, 1:].ravel()], axis=1),
                                np.stack([labels[:-1].ravel(), labels[1:].ravel()], axis=1)])
        neighbors = [set() for _ in range(n)]
        for first, second in pairs.tolist():
            if first != second and first < n and second < n:
                neighbors[first].add(second)
                neighbors[second].add(first)
        negatives = {(region, tile): ", ".join(config.regions[other].tiles[tile].strip()
                                               for other in sorted(neighbors[region]))
                     for region in range(n) for tile in range(k) if neighbors[region]}
        encoded = self.encode([(negative, False) for negative in negatives.values()])
        text = text.clone()
        for (region, tile), negative in negatives.items():
            text[0, 1 + region * k + tile] = encoded[(negative, False)].to(text)
        return text

    @torch.inference_mode()
    def __call__(self, config):
        """Generates the library of one specification (the three stages in the module docstring)."""
        device, n, k = self.device, config.num_regions, config.num_tiles
        with_branches = config.branches and config.layout == "sam"
        styled = config.styled
        tile_prompts = [styled(tile) for region in config.regions for tile in region.tiles]
        prefix_prompt = styled(config.category_prompt() if with_branches else config.variant_prompt([0] * n))
        branch_prompts = [styled(config.variant_prompt([branch] * n)) for branch in range(k)] if with_branches else []
        encoded = self.encode([(prompt, True) for prompt in [styled(config.prompt), *tile_prompts, prefix_prompt,
                                                             *branch_prompts]] + [("", False)])
        text = self.text(encoded, [styled(config.prompt), *tile_prompts], [""] * (1 + n * k))
        prefix_text = self.text(encoded, [prefix_prompt], [""])

        generator = torch.Generator(device=device).manual_seed(config.seed)
        sample = torch.randn((1, 3, config.height, config.width), generator=generator, device=device)
        self.reset_scheduler(config)
        timesteps = self.scheduler.timesteps
        library = Library(images=None, region_map=None, choice=[0] * n)
        variant_maps = None  # the branches' masks, None without branches

        # 1-2. The shared prefix, the branches and the layout.
        print(f"Prefix prompt: {prefix_prompt!r}")
        for step in tqdm(range(config.prefix_steps), desc="prefix"):
            if with_branches and step == config.branch_steps:
                print(f"Branches at step {step}:")
                library.choice, variant_maps = self.branch_and_choose(
                    config, sample, step, self.text(encoded, branch_prompts, [""] * k))
                prefix_prompt = styled(config.variant_prompt(library.choice))
                prefix_text = self.text(self.encode([(prefix_prompt, True), ("", False)]), [prefix_prompt], [""])
                print(f"Prefix prompt from step {step}: {prefix_prompt!r}")
            sigma = self.scheduler.sigmas[step].item()
            velocity = self.velocity(sample, sigma, prefix_text, config.cfg_scale)
            if step == config.prefix_steps - 1 and config.layout == "sam":
                print(f"Layout from step {step + config.lookahead}:")
                clean = self.look_ahead(sample, velocity, step, prefix_text, config.lookahead, config.cfg_scale)
                library.region_map = self.sam_layout(config, clean, library.choice, variant_maps)
                library.segmented = to_display_image(clean[0])
            sample = self.scheduler.step(velocity, timesteps[step], sample).prev_sample
        if config.layout == "rectangles":
            library.region_map = self.rectangles(config)
        counts = np.bincount(library.region_map.ravel(), minlength=n + 1)
        print("Patches per region: " + ", ".join(map(str, counts[:n])) + f"; shared region: {counts[n]}")

        # 3. The split and the joint denoising.
        sample = sample.repeat(k, 1, 1, 1)
        self.scheduler.model_outputs = [None if output is None else output.repeat(k, 1, 1, 1)
                                        for output in self.scheduler.model_outputs]
        layout = build_tile_attention_layout(library.region_map, n, k, PROMPT_LENGTH, config.attention, config.seed,
                                             device)
        if config.neighbor_negatives:
            text = self.neighbor_negatives(config, text, library.region_map)
        in_regions = (pixel_map(library.region_map, device) < n)[None, None]
        scale = torch.where(in_regions, float(config.tile_cfg_scale), float(config.cfg_scale))
        for step in tqdm(range(config.prefix_steps, config.steps), desc="tiles"):
            sigma = self.scheduler.sigmas[step].item()
            velocity = self.velocity(sample, sigma, text, scale, layout)
            sample = self.scheduler.step(velocity, timesteps[step], sample).prev_sample
        library.images = torch.stack([to_display_image(image) for image in sample])
        return library
