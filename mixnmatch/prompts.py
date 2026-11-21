"""
Prompt encoding with Gemma-2-2b-it, exactly as PixelDiT's inference does it, with an on-disk cache.

Positive prompts get PixelDiT's "complex human instruction" in front and keep the first token plus the last
PROMPT_LENGTH - 1 tokens; negative prompts are encoded plainly to PROMPT_LENGTH tokens. Padding tokens are kept
(the model was trained without a text mask). Every encoded prompt is cached in CACHE_DIR, named by a hash of
everything its encoding depends on, so Gemma is loaded only for prompts never seen before.
"""

import gc
import hashlib
import os

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, Gemma2Model

from .paths import CACHE_DIR, WEIGHTS_DIR

TEXT_ENCODER_REPO_ID = "Efficient-Large-Model/gemma-2-2b-it"  # the Gemma copy PixelDiT was trained with
TEXT_ENCODER_DIR = WEIGHTS_DIR / "gemma-2-2b-it"
TEXT_ENCODER_FILES = ["*.json", "model-*.safetensors", "tokenizer.model"]
PROMPT_LENGTH = 300  # tokens per prompt (PixelDiT's model_max_length)
BATCH_SIZE = 16
DTYPE = torch.bfloat16

# Copied from PixelDiT's stage-3 config.
COMPLEX_HUMAN_INSTRUCTION = "\n".join([
    'Given a user prompt, generate an "Enhanced prompt" that provides detailed visual descriptions suitable for image generation. Evaluate the level of detail in the user prompt:',
    "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, and spatial relationships to create vivid and concrete scenes.",
    "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
    "Here are examples of how to transform or refine prompts:",
    "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat curled up in a round shape, sleeping peacefully on a warm sunny windowsill, surrounded by pots of blooming red flowers.",
    "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps, a diverse crowd of people in colorful clothing, and a double-decker bus passing by towering glass skyscrapers.",
    "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
    "User Prompt: ",
])


def cache_path(prompt, positive):
    """Where the features of one (prompt, positive) pair are cached."""
    kind = COMPLEX_HUMAN_INSTRUCTION if positive else "negative prompt"
    key = "\n".join([TEXT_ENCODER_REPO_ID, str(PROMPT_LENGTH), kind, prompt])
    return CACHE_DIR / f"{hashlib.sha256(key.encode()).hexdigest()}.pt"


def load_tokenizer():
    """Gemma's tokenizer, after downloading the text encoder into WEIGHTS_DIR once."""
    if not TEXT_ENCODER_DIR.exists():
        print(f"Downloading {TEXT_ENCODER_REPO_ID} to {TEXT_ENCODER_DIR} (once)...")
        snapshot_download(TEXT_ENCODER_REPO_ID, local_dir=TEXT_ENCODER_DIR, allow_patterns=TEXT_ENCODER_FILES)
    tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_DIR)
    tokenizer.padding_side = "right"
    return tokenizer


@torch.inference_mode()
def gemma_features(prompts, positive, tokenizer, text_encoder):
    """[len(prompts), PROMPT_LENGTH, text dim] features of the transformer's view of the prompts."""
    instruction_length = len(tokenizer.encode(COMPLEX_HUMAN_INSTRUCTION))
    if positive:
        prompts = [COMPLEX_HUMAN_INSTRUCTION + prompt for prompt in prompts]
    max_length = instruction_length + PROMPT_LENGTH - 2 if positive else PROMPT_LENGTH
    tokens = tokenizer(prompts, max_length=max_length, padding="max_length", truncation=True,
                       return_tensors="pt").to(text_encoder.device)
    features = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask).last_hidden_state
    kept = [0] + list(range(-PROMPT_LENGTH + 1, 0)) if positive else slice(None)
    return features[:, kept].cpu()


def encode(prompts, device):
    """
    Encodes (prompt, positive) pairs, from the cache where possible; Gemma is loaded only for the missing ones and
    freed afterwards.

    Returns:
        {(prompt, positive): [PROMPT_LENGTH, text dim] CPU tensor}.
    """
    keys = list(dict.fromkeys(prompts))
    encoded = {key: torch.load(cache_path(*key))["features"] for key in keys if cache_path(*key).exists()}
    missing = sorted(set(keys) - encoded.keys())
    if missing:
        print(f"Encoding {len(missing)} prompt(s) with Gemma")
        tokenizer = load_tokenizer()
        text_encoder = Gemma2Model.from_pretrained(TEXT_ENCODER_DIR, dtype=DTYPE).to(device).eval()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for positive in (False, True):
            batch_prompts = [prompt for prompt, is_positive in missing if is_positive == positive]
            for start in range(0, len(batch_prompts), BATCH_SIZE):
                batch = batch_prompts[start:start + BATCH_SIZE]
                for prompt, features in zip(batch, gemma_features(batch, positive, tokenizer, text_encoder),
                                            strict=True):
                    path = cache_path(prompt, positive)
                    # Written whole, then renamed, so a parallel run never reads half a file.
                    torch.save({"features": features.clone()}, f"{path}.{os.getpid()}.tmp")
                    os.replace(f"{path}.{os.getpid()}.tmp", path)
                    encoded[(prompt, positive)] = features
        del text_encoder
        gc.collect()
        torch.cuda.empty_cache()
    return encoded


def encode_pairs(pairs, device):
    """
    Args:
        pairs: (positive prompt, negative prompt) pairs.

    Returns:
        {pair: [2, 1, PROMPT_LENGTH, text dim] CPU tensor (negative, positive)}.
    """
    pairs = list(dict.fromkeys(pairs))
    encoded = encode([(positive, True) for positive, _ in pairs] + [(negative, False) for _, negative in pairs],
                     device)
    return {(positive, negative): torch.stack([encoded[(negative, False)], encoded[(positive, True)]])[:, None]
            for positive, negative in pairs}
