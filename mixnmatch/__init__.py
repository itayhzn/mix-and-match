"""
Mix-and-Match: generating globally coherent images from interchangeable tiles.

    config.py         specification files and their defaults
    pipeline.py       the method (MixNMatchPipeline)
    attention.py      the attention rule between tiles, as a flex-attention mask
    segmentation.py   SAM 3 maps and the layout rules
    model.py          PixelDiT, extended to run several images as one sequence
    prompts.py        Gemma prompt encoding, cached on disk
    baselines.py      the baselines of the paper
    output.py         libraries on disk and the assembly of combinations
    paths.py          where weights, the prompt cache and results live
"""
