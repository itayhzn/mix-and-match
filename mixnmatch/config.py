"""
Library specifications: the JSON files generate.py reads.

    {
      "name": "portrait",                        optional; the file name without .json by default
      "seed": 123,
      "prompt": "A head-and-shoulders portrait of a person",          the global prompt y_0
      "style": "in a digital painting style",    optional; appended to every prompt the transformer reads
      "regions": [
        {"category": "hair or a hat",            the category word c_i
         "connector": "with",                    optional; joins the region to y_0 in the expanded prompt
         "tiles": ["a brown cowboy hat", "a red knitted beanie", "wavy brown hair", "a black beret"]},
        ...
      ]
    }

Every region has the same number k of tiles. The other fields (DEFAULTS) are the paper's settings; a file may set
them, which is how the ablations are run (see the README).
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

PATCH_SIZE = 16  # pixels per PixelDiT patch; image sizes are multiples of it

ATTENTION_RULES = ("average", "all_tiles", "sampled_tile", "none")
LAYOUTS = ("sam", "rectangles")


class ConfigError(Exception):
    """A specification file that is missing, malformed or inconsistent."""


@dataclass
class Region:
    """One region of a specification: its category word c_i and its k regional prompts (the tiles' descriptions)."""

    category: str
    tiles: list[str]
    connector: str = ""


@dataclass
class Config:
    """One specification with every setting filled in. The field comments give the paper's symbols."""

    name: str
    seed: int
    prompt: str  # y_0, the global prompt
    regions: list[Region]
    style: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 70  # T
    cfg_scale: float = 5.5  # w, the global prompt's guidance scale (and the shared region's)
    tile_cfg_scale: float = 7.5  # w_t, the guidance scale inside the tile regions after the split
    prefix_steps: int = 4  # tau: steps of the shared prefix before the k copies split
    branch_steps: int = 2  # tau_b: prefix steps under the category words before the branches
    branch_lookahead: int = 20  # h_b: steps every branch is denoised before SAM 3 segments it
    lookahead: int = 25  # h: steps the prefix is run ahead before its layout is segmented
    attention: str = "average"  # what a tile sees of the other regions (attention.py)
    layout: str = "sam"  # "sam": segmented from the prefix; "rectangles": horizontal bands (ablation)
    branches: bool = True  # composition variants chosen on branches; false: alternative 1 of every region
    neighbor_negatives: bool = True  # a tile's negative prompt holds its neighbors' regional prompts
    source: Path | None = field(default=None, repr=False)

    @property
    def num_regions(self):
        return len(self.regions)

    @property
    def num_tiles(self):
        return len(self.regions[0].tiles)

    def styled(self, prompt):
        """The prompt the transformer reads: "<prompt>, <style>" ("A garden gnome, in a pixel art style")."""
        return f"{prompt.strip().rstrip('.')}, {self.style.strip()}" if self.style else prompt

    def expanded_prompt(self, parts, regions=None):
        """
        y_0 expanded with one phrase per region, each after the region's connector: "A portrait of a person, with a
        red knitted beanie, wearing a gray hoodie".

        Args:
            parts: one phrase per region (a category word or a regional prompt).
            regions: the indices of the regions the parts belong to; all regions, in order, by default.
        """
        regions = self.regions if regions is None else [self.regions[index] for index in regions]
        parts = [part.strip().rstrip(".") for part in parts]
        parts = [f"{region.connector.strip()} {part[:1].lower() + part[1:]}".strip()
                 for region, part in zip(regions, parts, strict=True)]
        return f"{self.prompt.strip().rstrip('.')}, {', '.join(parts)}"

    def variant_prompt(self, choice):
        """y_0 expanded with alternative choice[i] of every region i."""
        return self.expanded_prompt([region.tiles[tile] for region, tile in zip(self.regions, choice, strict=True)])

    def category_prompt(self):
        """y_0 expanded with the category words, y_0[c_1, ..., c_n]."""
        return self.expanded_prompt([region.category for region in self.regions])

    def region_prompt(self, region, tile):
        """y_0 expanded with one region's alternative alone (used by the baselines)."""
        return self.expanded_prompt([self.regions[region].tiles[tile]], [region])


DEFAULTS = {f.name: f.default for f in fields(Config) if f.name not in ("name", "seed", "prompt", "regions",
                                                                         "source")}


def load_config(path):
    """
    Reads and checks one specification file.

    Returns:
        Config.

    Raises:
        ConfigError: with a readable message.
    """
    path = Path(path)
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ConfigError(f"cannot read {path}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("the file must hold one JSON object")
    unknown = sorted(set(raw) - {"name", "seed", "prompt", "regions"} - set(DEFAULTS))
    if unknown:
        raise ConfigError(f"unknown field(s) {unknown}")
    for key in ("seed", "prompt", "regions"):
        if key not in raw:
            raise ConfigError(f'"{key}" is missing')

    regions = []
    for index, entry in enumerate(raw["regions"] if isinstance(raw["regions"], list) else []):
        if not isinstance(entry, dict) or set(entry) - {"category", "connector", "tiles"} \
                or not isinstance(entry.get("category"), str) or not isinstance(entry.get("tiles"), list) \
                or not all(isinstance(tile, str) for tile in entry["tiles"]) \
                or not isinstance(entry.get("connector", ""), str):
            raise ConfigError(f'region {index} must be {{"category": str, "connector": str (optional), '
                              f'"tiles": [str, ...]}}')
        regions.append(Region(entry["category"], entry["tiles"], entry.get("connector", "")))
    if not regions:
        raise ConfigError('"regions" must be a non-empty list')
    if len({len(region.tiles) for region in regions}) != 1 or len(regions[0].tiles) < 1:
        raise ConfigError("every region must have the same number (at least 1) of tiles")

    config = Config(name=raw.get("name", path.stem), seed=raw["seed"], prompt=raw["prompt"], regions=regions,
                    source=path, **{key: raw[key] for key in DEFAULTS if key in raw})
    for f in fields(Config):
        expected = {int: int, float: (int, float), bool: bool, str: str}.get(f.type)
        value = getattr(config, f.name)
        # JSON booleans are ints in Python, so they are refused where a number is expected.
        if expected and (not isinstance(value, expected) or (f.type is not bool and isinstance(value, bool))):
            raise ConfigError(f'"{f.name}" has the wrong type ({type(value).__name__})')
    if config.attention not in ATTENTION_RULES:
        raise ConfigError(f'"attention" must be one of {list(ATTENTION_RULES)}')
    if config.layout not in LAYOUTS:
        raise ConfigError(f'"layout" must be one of {list(LAYOUTS)}')
    if config.width % PATCH_SIZE or config.height % PATCH_SIZE:
        raise ConfigError(f"width and height must be multiples of {PATCH_SIZE}")
    if not 1 <= config.prefix_steps < config.steps:
        raise ConfigError('"prefix_steps" must be in [1, steps)')
    if config.layout == "sam" and config.prefix_steps + config.lookahead > config.steps:
        raise ConfigError('"prefix_steps" + "lookahead" must not exceed "steps"')
    if config.branches and config.layout == "sam":
        if not 0 <= config.branch_steps < config.prefix_steps:
            raise ConfigError('"branch_steps" must be in [0, prefix_steps)')
        if config.branch_steps + config.branch_lookahead > config.steps:
            raise ConfigError('"branch_steps" + "branch_lookahead" must not exceed "steps"')
    return config
