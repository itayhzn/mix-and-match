"""Where weights, the prompt cache and results live (the first two can be moved with environment variables)."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = Path(os.environ.get("MIXNMATCH_WEIGHTS", ROOT / "weights"))
CACHE_DIR = Path(os.environ.get("MIXNMATCH_PROMPT_CACHE", ROOT / "prompt_cache"))
OUTPUT_DIR = ROOT / "outputs"
