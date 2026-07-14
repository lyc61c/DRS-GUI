"""Run DRS-GUI grounding on one screenshot and one instruction."""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import torch
from PIL import Image

from policies import policy_map
from run import initialize_models


def parse_args():
    parser = argparse.ArgumentParser(description="DRS-GUI single-image inference")
    parser.add_argument("--image", required=True, help="Path to one GUI screenshot")
    parser.add_argument("--instruction", required=True, help="Element to locate")
    parser.add_argument("--model-type", choices=["qwen2_5vl", "ugroundv1"], default="qwen2_5vl")
    parser.add_argument("--model-path", required=True, help="Hugging Face model ID or local path")
    parser.add_argument("--detector-path", required=True, help="OmniParser icon detector model.pt")
    parser.add_argument("--caption-model", required=True, help="OmniParser Florence-2 model path")
    parser.add_argument("--instructor-model", default="hkunlp/instructor-large")
    parser.add_argument("--platform", default="unknown", help="For example: windows, macos, web")
    parser.add_argument("--application", default="unknown", help="For example: vscode, excel")
    parser.add_argument("--mcts-iterations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


async def infer(args):
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    with Image.open(image_path) as image:
        image_size = [image.width, image.height]

    row = {
        "id": image_path.stem,
        "img_filename": str(image_path),
        "img_size": image_size,
        "instruction": args.instruction,
        "prompt_to_evaluate": args.instruction,
        "platform": args.platform,
        "application": args.application,
        "group": None,
        "language": "en",
        "instruction_style": "instruction",
        "gt_type": "positive",
        "ui_type": "unknown",
        "task_filename": "single_image",
    }
    sample = policy_map["drsgui.mcts"](row, args)
    result = await sample.process()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        logging.info("Saved prediction to %s", output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("DRS-GUI currently requires a CUDA-capable GPU")
    random.seed(114514)
    torch.manual_seed(114514)
    initialize_models(args)
    asyncio.run(infer(args))


if __name__ == "__main__":
    main()
