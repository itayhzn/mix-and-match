"""
Assembles combinations of a library (or of a baseline's equal-cost tiles) into whole images.

    python combine.py outputs/portrait 0123 3210        # the given combinations (one tile index per region)
    python combine.py outputs/portrait --random 16      # 16 random combinations
    python combine.py outputs/portrait --all            # all k^n of them
    python combine.py outputs/portrait --all --grid     # ... and a contact sheet of them

Images go to <library>/combinations/<tile indices>.png (and <library>/combinations.jpg with --grid).
"""

import argparse
import math
import random
import sys
from pathlib import Path

from PIL import Image

from mixnmatch.output import TileSet

GRID_CELL = 256  # pixels per image on the contact sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", type=Path)
    parser.add_argument("combinations", nargs="*", help="one digit per region, e.g. 0123")
    parser.add_argument("--random", type=int, help="this many random combinations")
    parser.add_argument("--all", action="store_true", help="every combination")
    parser.add_argument("--grid", action="store_true", help="also write a contact sheet")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    tiles = TileSet(args.library)
    if args.all:
        combinations = list(tiles.all_combinations())
    elif args.random:
        every = list(tiles.all_combinations())
        combinations = random.Random(args.seed).sample(every, min(args.random, len(every)))
    else:
        combinations = [[int(digit) for digit in text] for text in args.combinations]
    if not combinations:
        sys.exit("Error: give combinations, --random N or --all")
    for combination in combinations:
        if len(combination) != tiles.num_regions or not all(0 <= tile < tiles.num_tiles for tile in combination):
            sys.exit(f"Error: {''.join(map(str, combination))} needs {tiles.num_regions} digits below "
                     f"{tiles.num_tiles}")

    folder = args.library / "combinations"
    folder.mkdir(exist_ok=True)
    images = []
    for combination in combinations:
        image = tiles.assemble(combination)
        image.save(folder / f"{''.join(map(str, combination))}.png")
        images.append(image)
    print(f"Wrote {len(images)} image(s) to {folder}")
    if args.grid:
        columns = math.ceil(math.sqrt(len(images)))
        rows = math.ceil(len(images) / columns)
        sheet = Image.new("RGB", (columns * GRID_CELL, rows * GRID_CELL), "white")
        for index, image in enumerate(images):
            sheet.paste(image.resize((GRID_CELL, GRID_CELL), Image.LANCZOS),
                        ((index % columns) * GRID_CELL, (index // columns) * GRID_CELL))
        sheet.save(args.library / "combinations.jpg", quality=90)
        print(f"Wrote {args.library / 'combinations.jpg'}")


if __name__ == "__main__":
    main()
