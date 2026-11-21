"""
SAM 3 masks and the rules that turn them into a layout (paper Sec. 3.5 and its appendix).

    phrase map   per pixel, the max over SAM 3's candidate instances of (query score x presence x mask probability);
                 a phrase whose presence is below PRESENCE_FLOOR votes "not here" everywhere. Averaged per patch
                 and blurred (SIGMA pixels).
    region map   the union (max) of a region's searched phrases, each rescaled to a maximum of 1 first.
    assignment   a patch goes to a region whose map reaches TAU there, preferring the smallest such mask (a hat
                 inside a whole-body mask keeps its patches); patches no region reaches form the shared region.
    clean-up     a region keeps only the parts of its mask that touch another region's (the staff in the hand, not
                 the clover across the garden); it grows over the branch masks of its other drawn alternatives where
                 they connect to it; islands smaller than MIN_COMPONENT patches join their surroundings; every
                 region grows into the shared region by MARGIN pixels.
    branches     each region's composition variant is the alternative whose branch mask best covers those of its
                 other drawn alternatives (choose_variants, Eq. 5).
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from scipy.ndimage import binary_dilation, label
from torchvision.transforms.functional import gaussian_blur
from transformers import Sam3Model, Sam3Processor

from .config import PATCH_SIZE
from .paths import WEIGHTS_DIR

SAM3_REPO_ID = "facebook/sam3"  # gated on the Hub: accept its license and log in once
SAM3_DIR = WEIGHTS_DIR / "sam3"
SAM3_FILES = ["*.json", "*.txt", "model.safetensors"]
SAM3_DTYPE = torch.bfloat16

TAU = 0.2  # a region's map must reach this for the region to take a patch
MARGIN = 48  # pixels every region grows into the shared region
MIN_COMPONENT = 8  # patches; smaller islands of the layout join the label around them
PRESENCE_FLOOR = 0.3  # a phrase SAM 3 thinks absent below this presence gets an empty map
SIGMA = 16  # pixels of gaussian blur on the patch maps
# A branch drew an alternative when its peak reaches DRAWN_FRACTION of the region's strongest alternative's peak,
# and at least DRAWN_FLOOR (relative, since SAM 3 scores small subjects low even when they are clearly drawn).
DRAWN_FRACTION = 0.3
DRAWN_FLOOR = 0.05


def load_sam3():
    """(Sam3Model, Sam3Processor) on the CPU, downloaded into WEIGHTS_DIR once."""
    if not (SAM3_DIR / "model.safetensors").exists():
        print(f"Downloading {SAM3_REPO_ID} to {SAM3_DIR} (once)...")
        snapshot_download(SAM3_REPO_ID, local_dir=SAM3_DIR, allow_patterns=SAM3_FILES)
    return Sam3Model.from_pretrained(SAM3_DIR, dtype=SAM3_DTYPE).eval(), Sam3Processor.from_pretrained(SAM3_DIR)


@torch.inference_mode()
def segment_phrases(model, processor, image, phrases):
    """
    Args:
        model, processor: SAM 3, the model on the device it runs on.
        image: [3, H, W] image in [0, 1].
        phrases: the phrases to find (the image is encoded once; a repeated phrase is run once).

    Returns:
        [phrases, H, W] float CPU maps in [0, 1].
    """
    height, width = image.shape[-2:]
    inputs = processor(images=image.float().cpu(), do_rescale=False, return_tensors="pt")
    vision_embeds = model.get_vision_features(pixel_values=inputs.pixel_values.to(model.device, SAM3_DTYPE))
    maps = {}
    for phrase in dict.fromkeys(phrases):
        text = processor(text=phrase, return_tensors="pt").to(model.device)
        outputs = model(vision_embeds=vision_embeds, input_ids=text.input_ids, attention_mask=text.attention_mask)
        scores = outputs.pred_logits[0].float().sigmoid()
        presence = 1.0 if outputs.presence_logits is None else outputs.presence_logits[0].float().sigmoid().item()
        scores = scores * presence
        masks = F.interpolate(outputs.pred_masks.float(), size=(height, width), mode="bilinear",
                              align_corners=False)[0].sigmoid()
        maps[phrase] = (scores[:, None, None] * masks).amax(dim=0).cpu()
        print(f"  SAM 3 {phrase!r}: presence {presence:.3f}, peak {maps[phrase].max().item():.3f}")
        if presence < PRESENCE_FLOOR:
            maps[phrase] = torch.zeros_like(maps[phrase])
    return torch.stack([maps[phrase] for phrase in phrases])


def rescale_to_max(maps):
    """Every map divided by its own maximum (an empty map stays empty)."""
    return maps / np.maximum(maps.max(axis=(1, 2), keepdims=True), 1e-12)


def region_maps(phrase_maps, phrase_tiles, num_regions, num_tiles, rescale):
    """
    Phrase maps -> per-tile maps (the union of a tile's phrases) on the patch grid.

    Args:
        phrase_maps: [phrases, H, W] maps from segment_phrases.
        phrase_tiles: the tile (region * k + alternative) every phrase belongs to.
        num_regions, num_tiles: n and k.
        rescale: rescale every phrase map to a maximum of 1 before the union.

    Returns:
        [n * k, patch rows, patch cols] float tensor (zeros for tiles without phrases).
    """
    patch_maps = F.avg_pool2d(phrase_maps[:, None], PATCH_SIZE)[:, 0]
    sigma = SIGMA / PATCH_SIZE
    kernel = 2 * math.ceil(3 * sigma) + 1
    patch_maps = gaussian_blur(patch_maps[:, None], [kernel, kernel], [sigma, sigma])[:, 0]
    if rescale:
        patch_maps = torch.from_numpy(rescale_to_max(patch_maps.numpy())).float()
    tile_maps = torch.zeros(num_regions * num_tiles, *patch_maps.shape[1:])
    for phrase_map, tile in zip(patch_maps, phrase_tiles, strict=True):
        tile_maps[tile] = torch.maximum(tile_maps[tile], phrase_map)
    return tile_maps


def keep_attached(maps):
    """
    Every map keeps only its connected parts (above TAU) that touch another map's; a map nothing touches is kept.

    Args:
        maps: [n, patch rows, patch cols] numpy maps.
    """
    masks = maps >= TAU
    kept = maps.copy()
    for region in range(len(maps)):
        others = masks[np.arange(len(maps)) != region].any(axis=0)
        parts, count = label(masks[region])
        touching = [part for part in range(1, count + 1) if (binary_dilation(parts == part) & others).any()]
        if touching:
            kept[region][masks[region] & ~np.isin(parts, touching)] = 0
    return kept


def assign(maps):
    """
    [n, patch rows, patch cols] region maps -> [patch rows, patch cols] labels: every patch where some map reaches TAU
    goes to the region with the smallest such mask, the rest to the shared region (label n).
    """
    num_regions = len(maps)
    masks = maps >= TAU
    labels = np.full(maps.shape[1:], num_regions)
    for region in np.argsort(-masks.sum(axis=(1, 2)), kind="stable"):  # largest first, so smaller ones go on top
        labels[masks[region]] = region
    return labels


def drawn_variants(variant_maps):
    """[n, k] bool: the alternatives a branch drew (see DRAWN_FRACTION)."""
    peaks = variant_maps.max(axis=(2, 3))
    return peaks >= np.maximum(DRAWN_FRACTION * peaks.max(axis=1, keepdims=True), DRAWN_FLOOR)


def choose_variants(variant_maps):
    """
    Every region's composition variant: the drawn alternative whose branch mask covers the most of the others'
    (soft |M_j n M_j'| / |M_j'|, averaged over the region's other drawn alternatives j').

    Args:
        variant_maps: [n, k, patch rows, patch cols] numpy maps, alternative j measured in branch j.

    Returns:
        (the chosen alternative of every region, [n, k] scores).
    """
    areas = variant_maps.sum(axis=(2, 3))
    overlap = np.minimum(variant_maps[:, :, None], variant_maps[:, None]).sum(axis=(3, 4))
    covered = overlap / np.maximum(areas[:, None, :], 1e-6)
    drawn = drawn_variants(variant_maps)
    others = drawn[:, None, :] & ~np.eye(variant_maps.shape[1], dtype=bool)[None]
    scores = (covered * others).sum(axis=2) / np.maximum(others.sum(axis=2), 1)
    scores[~drawn] = -1.0
    return scores.argmax(axis=1).tolist(), scores


def grow_into_variants(labels, variant_maps, choice):
    """Every region also takes the shared patches its other drawn alternatives' branch masks cover, where they
    connect to it; no region takes another's patches."""
    num_regions = len(variant_maps)
    drawn = drawn_variants(variant_maps)
    grown = labels.copy()
    for region in range(num_regions):
        others = [tile for tile in range(variant_maps.shape[1]) if tile != choice[region] and drawn[region, tile]]
        inside = labels == region
        if not others or not inside.any():
            continue
        parts, _ = label((variant_maps[region, others] >= TAU).any(axis=0) | inside)
        attached = np.isin(parts, np.unique(parts[inside])) & (parts > 0)
        grown[attached & (grown == num_regions)] = region
    return grown


def remove_small_components(labels):
    """Every island (a connected run of one label) under MIN_COMPONENT patches takes the label most common around
    it."""
    cleaned = labels.copy()
    for value in np.unique(labels):
        parts, count = label(labels == value)
        for part in range(1, count + 1):
            island = parts == part
            if island.sum() >= MIN_COMPONENT:
                continue
            around = cleaned[binary_dilation(island) & ~island]
            if around.size:
                cleaned[island] = np.bincount(around).argmax()
    return cleaned


def grow_by_margin(labels, num_regions):
    """A shared patch whose center lies within MARGIN pixels of a region patch's center joins the nearest region."""
    rows, cols = np.indices(labels.shape)
    centers = torch.from_numpy(np.stack([rows.ravel(), cols.ravel()], axis=1) * PATCH_SIZE).float()
    flat = labels.ravel()
    is_region = flat < num_regions
    if not is_region.any():
        return labels
    nearest_distance, nearest = torch.cdist(centers[~is_region], centers[is_region]).min(dim=1)
    grown = flat.copy()
    within = (nearest_distance <= MARGIN).numpy()
    grown[np.nonzero(~is_region)[0][within]] = flat[is_region][nearest.numpy()[within]]
    return grown.reshape(labels.shape)
