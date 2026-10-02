"""Shared command-line options and model setup for inference and evaluation."""

import argparse
import logging
import random
from pathlib import Path


BENCHMARKS = ("screenspot_v1", "screenspot_v2", "screenspot_pro")
MODEL_TYPES = ("qwen2_5vl", "ugroundv1")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def add_runtime_arguments(parser):
    """Keep both entry points on the same model and search configuration."""
    models = parser.add_argument_group("models")
    models.add_argument("--model-type", choices=MODEL_TYPES, default="qwen2_5vl")
    models.add_argument("--model-path", required=True, help="Hugging Face model ID or local path")
    models.add_argument("--detector-path", required=True, help="OmniParser icon_detect/model.pt")
    models.add_argument("--caption-model", required=True, help="Florence-2 caption model ID or local path")
    models.add_argument(
        "--caption-processor", default=None,
        help="Optional complete local Florence-2 processor directory or model ID",
    )
    models.add_argument("--instructor-model", default="hkunlp/instructor-large")
    search = parser.add_argument_group("search and output")
    search.add_argument("--mcts-iterations", type=positive_int, default=8)
    search.add_argument("--max-depth", type=positive_int, default=3)
    search.add_argument("--seed", type=int, default=114514)
    search.add_argument(
        "--include-search-tree",
        action="store_true",
        help="Include structured and readable search trees in the JSON output",
    )
    search.add_argument("--output", default=None, help="JSON output path")


def configure_runtime(args):
    """Check hardware and seed the runtime without loading model weights."""
    import numpy as np
    import torch

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("DRS-GUI currently requires a CUDA-capable GPU")
    if not Path(args.detector_path).expanduser().is_file():
        raise FileNotFoundError(f"Icon detector not found: {args.detector_path}")
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)


def initialize_models(args):
    """Load the grounding model, detector, captioner, and semantic encoder."""
    from sentence_transformers import SentenceTransformer

    from model_factory import build_model
    from OmniParser.util.utils import get_caption_model_processor, get_yolo_model

    args.model = build_model(args.model_type, args.model_path)
    args.som_model = get_yolo_model(str(Path(args.detector_path).expanduser())).to("cuda")
    args.caption_model = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=args.caption_model,
        processor_name_or_path=getattr(args, "caption_processor", None),
        device="cuda",
    )
    args.semantic_model = SentenceTransformer(args.instructor_model)
    # SentenceTransformer applies this automatically for Hub IDs, but not local paths.
    if "instructor" in args.instructor_model.lower():
        args.semantic_model.set_pooling_include_prompt(include_prompt=False)
    return args
