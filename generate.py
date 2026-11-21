"""
Generates tile libraries.

    python generate.py configs/portrait.json              # one specification -> outputs/portrait/
    python generate.py configs/                           # every *.json in the folder, in name order
    python generate.py configs/ --out outputs/mine --skip-existing

Every specification is checked first; invalid ones are reported and skipped. The models are loaded once and the
libraries run one after another; a failing one is reported and the rest go on.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

from mixnmatch.config import ConfigError, load_config  # noqa: E402
from mixnmatch.output import save_library  # noqa: E402
from mixnmatch.paths import OUTPUT_DIR  # noqa: E402
from mixnmatch.pipeline import MixNMatchPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", type=Path, nargs="+", help="specification .json files or folders of them")
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR, help="results go to <out>/<name>/")
    parser.add_argument("--skip-existing", action="store_true", help="skip libraries already in --out")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        sys.exit("Error: Mix-and-Match needs a CUDA GPU")

    configs, invalid, failures = [], [], []
    for spec in args.specs:
        for path in sorted(spec.glob("*.json")) if spec.is_dir() else [spec]:
            try:
                config = load_config(path)
            except ConfigError as error:
                invalid.append(f"{path}: {error}")
                continue
            if args.skip_existing and (args.out / config.name / "run_stats.json").is_file():
                continue
            configs.append(config)
    for reason in invalid:
        print(f"Skipped {reason}")
    if not configs:
        sys.exit(1 if invalid else 0)

    pipeline = MixNMatchPipeline.from_weights("cuda")
    for config in configs:
        print(f"\n=== {config.name} ({config.num_regions} regions x {config.num_tiles} tiles) ===")
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            library = pipeline(config)
            torch.cuda.synchronize()
            stats = {"seconds": round(time.perf_counter() - start, 2),
                     "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "gpu": torch.cuda.get_device_name()}
            folder = args.out / config.name
            save_library(library, config, folder)
            (folder / "run_stats.json").write_text(json.dumps(stats, indent=2))
            print(f"Saved {folder} ({stats['seconds']} s)")
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            failures.append(f"{config.source}: out of GPU memory")
    print(f"\nFinished: {len(configs) - len(failures)} of {len(configs)} succeeded")
    for failure in failures:
        print(f"  FAILED {failure}")
    sys.exit(1 if failures or invalid else 0)


if __name__ == "__main__":
    main()
