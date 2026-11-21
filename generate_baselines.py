"""
Runs the baselines (mixnmatch/baselines.py) on the layouts of finished Mix-and-Match libraries.

    python generate_baselines.py configs/portrait.json --method multidiffusion
    python generate_baselines.py configs/ --method all --protocol equal combination

A specification's library is read from <libraries>/<name>/ (outputs/ by default); one without a library is
skipped. Results go to <out>/<method>_<protocol>/<name>/ (outputs/baselines/ by default):

    equal         k images, image j taking alternative j in every region, cut into tiles along the library's
                  layout and saved like a library (tiles/), so combine.py assembles them. "independent" cuts tile
                  (i, j) from regional prompt (i, j)'s image and "single" every tile from its one image; these two
                  have only this protocol.
    combination   one whole image per sampled combination (output.sample_combinations, the same 8 for every
                  method), combinations/<tile per region, joined by "-">.png, listed in combinations.json.

Each result also holds config.json (with "method" and "protocol") and run_stats.json (seconds, peak GPU memory,
images generated, and the images the transformer ran on, both guidance passes counted).
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchvision.transforms.functional import to_pil_image  # noqa: E402

from mixnmatch.baselines import EQUAL_COST_ONLY, METHODS, BaselinePipeline, method_prompts  # noqa: E402
from mixnmatch.config import ConfigError, load_config  # noqa: E402
from mixnmatch.output import LAYOUT_FILENAME, TILES_DIRNAME, sample_combinations, save_tiles  # noqa: E402
from mixnmatch.paths import OUTPUT_DIR  # noqa: E402
from mixnmatch.prompts import encode_pairs  # noqa: E402

PROTOCOLS = ("equal", "combination")


def protocol_choices(config, protocol, region_map):
    """The combinations a protocol generates: k diagonal ones (equal) or the 8 sampled ones (combination)."""
    if protocol == "equal":
        return [[tile] * config.num_regions for tile in range(config.num_tiles)]
    present = [region for region in range(config.num_regions) if (region_map == region).any()]
    return sample_combinations(config.name, present, config.num_tiles, config.num_regions)


def save_result(images, source, reference, config, method, protocol, choices, library, folder, region_map, stats):
    """Writes one result folder as described in the module docstring."""
    folder.mkdir(parents=True, exist_ok=True)
    settings = json.loads((library / "config.json").read_text())
    settings.update(method=method, protocol=protocol)
    (folder / "config.json").write_text(json.dumps(settings, indent=2))
    pil_images = [to_pil_image(image) for image in images]
    if reference is not None:
        to_pil_image(reference).save(folder / "reference.png")
    if protocol == "combination":
        (folder / "combinations").mkdir(exist_ok=True)
        names = []
        for choice, image in zip(choices, pil_images, strict=True):
            names.append(f"{'-'.join(map(str, choice))}.png")
            image.save(folder / "combinations" / names[-1])
        shutil.copy(library / TILES_DIRNAME / LAYOUT_FILENAME, folder / LAYOUT_FILENAME)
        (folder / "combinations.json").write_text(json.dumps({"combinations": choices, "files": names}))
    else:
        save_tiles(lambda region, tile: pil_images[source(region, tile)], config.num_tiles, region_map,
                   config.num_regions, folder / TILES_DIRNAME)
    (folder / "run_stats.json").write_text(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", type=Path, nargs="+", help="specification .json files or folders of them")
    parser.add_argument("--method", nargs="+", choices=(*METHODS, "all"), required=True)
    parser.add_argument("--protocol", nargs="+", choices=PROTOCOLS, default=["equal"])
    parser.add_argument("--libraries", type=Path, default=OUTPUT_DIR, help="where the Mix-and-Match libraries are")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR / "baselines")
    parser.add_argument("--force", action="store_true", help="redo results that exist already")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Error: the baselines need a CUDA GPU")
    methods = METHODS if "all" in args.method else args.method
    pairs = [(m, p) for m in methods for p in args.protocol if not (p == "combination" and m in EQUAL_COST_ONLY)]
    if not pairs:
        sys.exit(f"Error: {', '.join(EQUAL_COST_ONLY)} have only the equal protocol")

    jobs, skipped = [], []
    for spec in args.specs:
        for path in sorted(spec.glob("*.json")) if spec.is_dir() else [spec]:
            try:
                config = load_config(path)
            except ConfigError as error:
                skipped.append(f"{path}: {error}")
                continue
            library = args.libraries / config.name
            if not (library / TILES_DIRNAME / LAYOUT_FILENAME).is_file():
                skipped.append(f"{path}: no library at {library}")
                continue
            region_map = np.array(json.loads((library / TILES_DIRNAME / LAYOUT_FILENAME).read_text())["region_map"])
            for method, protocol in pairs:
                folder = args.out / f"{method}_{protocol}" / config.name
                if (folder / "run_stats.json").is_file() and not args.force:
                    continue
                jobs.append((config, method, protocol, library, folder, region_map,
                             protocol_choices(config, protocol, region_map)))
    for reason in skipped:
        print(f"Skipped {reason}")
    print(f"{len(jobs)} result(s) to make")
    if not jobs:
        sys.exit(1 if skipped else 0)

    # Every prompt is encoded before PixelDiT is loaded, so Gemma and PixelDiT are never on the GPU together.
    embeds = encode_pairs([pair for config, method, _, _, _, region_map, choices in jobs
                           for pair in method_prompts(config, method, choices, region_map)], "cuda")
    pipeline = BaselinePipeline.from_weights("cuda")
    failures = []
    for config, method, protocol, library, folder, region_map, choices in jobs:
        print(f"\n=== {method} ({protocol}): {config.name} ===")
        pipeline.image_passes = 0
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            images, source, reference, generated = pipeline.run(config, method, choices, region_map, embeds)
            torch.cuda.synchronize()
            stats = {"seconds": round(time.perf_counter() - start, 2),
                     "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "gpu": torch.cuda.get_device_name(), "images": generated,
                     "image_passes": pipeline.image_passes}
            save_result(images, source, reference, config, method, protocol, choices, library, folder, region_map,
                        stats)
            print(f"Saved {folder} ({stats['seconds']} s)")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            failures.append(f"{config.name} ({method}, {protocol}): out of GPU memory")
    print(f"\nFinished: {len(jobs) - len(failures)} of {len(jobs)} succeeded")
    for failure in failures:
        print(f"  FAILED {failure}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
