"""
Libraries on disk, and assembling their combinations.

    outputs/<name>/
        config.json                   the specification, every setting filled in
        tiles/layout.json             {"width", "height", "num_regions", "num_tiles", "region_map"} (patch grid;
                                      num_regions marks the shared region)
        tiles/region<i>/tile<j>.png   tile j of region i: its bounding box, transparent outside the region
        tiles/shared.png              the shared region (the same in every combination)
        segmentation.png              the layout over the image it was segmented on (with the "sam" layout)
        run_stats.json                seconds and peak GPU memory of the run

A combination is one tile index per region; assembling it pastes every chosen tile through its alpha.
"""

import dataclasses
import itertools
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from .config import PATCH_SIZE

LAYOUT_FILENAME = "layout.json"
TILES_DIRNAME = "tiles"
SHARED_FILENAME = "shared.png"
COLORS = [(230, 75, 60), (60, 140, 230), (60, 180, 90), (240, 170, 30), (170, 90, 210), (40, 190, 190),
          (240, 110, 170), (150, 110, 60)]
SHARED_COLOR = (90, 90, 90)


def regions_of(region_map, num_regions):
    """
    Every region's pixel bounding box and alpha mask, the shared region last.

    Returns:
        List of n + 1 entries, each ((left, top, right, bottom), PIL "L" alpha of the box) or None for an empty one.
    """
    pixels = np.asarray(region_map).repeat(PATCH_SIZE, axis=0).repeat(PATCH_SIZE, axis=1)
    boxes = []
    for region in range(num_regions + 1):
        inside = pixels == region
        rows, cols = np.nonzero(inside)
        if len(rows) == 0:
            boxes.append(None)
            continue
        left, top, right, bottom = int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1
        boxes.append(((left, top, right, bottom),
                      Image.fromarray(inside[top:bottom, left:right].astype(np.uint8) * 255)))
    return boxes


def cut(image, region):
    """The tile of `region` ((box, alpha) from regions_of) cut from a PIL image: an RGBA crop of the box."""
    box, alpha = region
    tile = image.crop(box).convert("RGBA")
    tile.putalpha(alpha)
    return tile


def save_tiles(image_of, num_tiles, region_map, num_regions, folder):
    """
    Cuts full images into tiles along a layout.

    Args:
        image_of: image_of(region, tile) -> the PIL RGB image tile `tile` of region `region` is cut from; the shared
            region asks for (num_regions, 0).
        num_tiles: k.
        region_map: [patch rows, patch cols] labels.
        num_regions: n.
        folder: the tiles folder to write.
    """
    folder.mkdir(parents=True, exist_ok=True)
    regions = regions_of(region_map, num_regions)
    for region_index, region in enumerate(regions[:num_regions]):
        if region is None:
            continue
        (folder / f"region{region_index}").mkdir(exist_ok=True)
        for tile_index in range(num_tiles):
            cut(image_of(region_index, tile_index), region).save(
                folder / f"region{region_index}" / f"tile{tile_index}.png")
    if regions[num_regions] is not None:
        cut(image_of(num_regions, 0), regions[num_regions]).save(folder / SHARED_FILENAME)
    width, height = image_of(num_regions, 0).size
    layout = {"width": width, "height": height, "num_regions": num_regions, "num_tiles": num_tiles,
              "region_map": np.asarray(region_map).tolist()}
    (folder / LAYOUT_FILENAME).write_text(json.dumps(layout))


def segmentation_image(image, region_map, num_regions, opacity=0.45):
    """The layout drawn over an image, one color per region, the shared region gray."""
    colors = np.array([COLORS[i % len(COLORS)] for i in range(num_regions)] + [SHARED_COLOR], dtype=np.uint8)
    overlay = Image.fromarray(colors[np.asarray(region_map)]).resize(image.size, Image.NEAREST)
    return Image.blend(image.convert("RGB"), overlay, opacity)


def save_library(library, config, folder):
    """Writes a pipeline.Library as described at the top of this file."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    settings = {field.name: getattr(config, field.name) for field in dataclasses.fields(config)
                if field.name != "source"}
    settings["regions"] = [dataclasses.asdict(region) for region in config.regions]
    settings["composition_variants"] = library.choice
    (folder / "config.json").write_text(json.dumps(settings, indent=2))
    # Copy j holds tile j of every region; the shared region is the same in every copy.
    images = [to_pil_image(image) for image in library.images]
    save_tiles(lambda region, tile: images[tile], config.num_tiles, library.region_map, config.num_regions,
               folder / TILES_DIRNAME)
    if library.segmented is not None:
        segmentation_image(to_pil_image(library.segmented), library.region_map, config.num_regions).save(
            folder / "segmentation.png")


class TileSet:
    """The tiles of a saved library (or of a baseline's equal-cost run), ready to assemble."""

    def __init__(self, folder):
        folder = Path(folder)
        tiles = folder / TILES_DIRNAME if (folder / TILES_DIRNAME).is_dir() else folder
        self.layout = json.loads((tiles / LAYOUT_FILENAME).read_text())
        self.num_regions, self.num_tiles = self.layout["num_regions"], self.layout["num_tiles"]
        self.size = (self.layout["width"], self.layout["height"])
        self.boxes = regions_of(self.layout["region_map"], self.num_regions)
        self.tiles = [[Image.open(tiles / f"region{region}" / f"tile{tile}.png").convert("RGBA")
                       for tile in range(self.num_tiles)] if box is not None else []
                      for region, box in enumerate(self.boxes[:self.num_regions])]
        self.shared = (Image.open(tiles / SHARED_FILENAME).convert("RGBA")
                       if self.boxes[self.num_regions] is not None else None)

    def present_regions(self):
        """The regions that own at least one patch of the layout."""
        return [region for region in range(self.num_regions) if self.tiles[region]]

    def all_combinations(self):
        """Every combination of the regions present in the layout (an empty region stays at tile 0)."""
        present = self.present_regions()
        for pick in itertools.product(range(self.num_tiles), repeat=len(present)):
            combination = [0] * self.num_regions
            for region, tile in zip(present, pick, strict=True):
                combination[region] = tile
            yield combination

    def assemble(self, combination):
        """PIL RGB image of one combination (a tile index per region)."""
        image = Image.new("RGB", self.size)
        if self.shared is not None:
            image.paste(self.shared, self.boxes[self.num_regions][0][:2], self.shared)
        for region, tile in enumerate(combination):
            if self.tiles[region]:
                piece = self.tiles[region][tile]
                image.paste(piece, self.boxes[region][0][:2], piece)
        return image


def sample_combinations(name, present_regions, num_tiles, num_regions, count=8):
    """
    The combinations a library is judged on (paper Sec. 4.1): a uniform sample without replacement, seeded by the
    library's name so every method is judged on the same ones; all of them when there are no more than count.
    """
    total = num_tiles ** len(present_regions)
    rng = random.Random(f"{name}-combinations")
    if total <= count:
        picks = list(itertools.product(range(num_tiles), repeat=len(present_regions)))
    else:
        picks = []
        while len(picks) < count:
            pick = tuple(rng.randrange(num_tiles) for _ in present_regions)
            if pick not in picks:
                picks.append(pick)
    combinations = []
    for pick in picks:
        combination = [0] * num_regions
        for region, tile in zip(present_regions, pick, strict=True):
            combination[region] = tile
        combinations.append(combination)
    return combinations
