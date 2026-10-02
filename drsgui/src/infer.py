"""Run DRS-GUI grounding on one screenshot and one instruction."""

import argparse
import asyncio
import json
import logging
from pathlib import Path

from runtime import add_runtime_arguments, configure_runtime, initialize_models
from utils import save_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DRS-GUI single-image inference")
    parser.add_argument("--image", required=True, help="Path to one GUI screenshot")
    parser.add_argument("--instruction", required=True, help="Element to locate")
    parser.add_argument("--platform", default="unknown", help="For example: windows, macos, web")
    parser.add_argument("--application", default="unknown", help="For example: vscode, excel")
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if not args.instruction.strip():
        parser.error("--instruction must not be empty")
    return args


def prepare_sample(args):
    """Read the input before loading the GPU models."""
    from PIL import Image

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Screenshot not found: {image_path}")

    with Image.open(image_path) as image:
        image_size = [image.width, image.height]

    return {
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


async def infer(args, row=None):
    from policies import policy_map

    row = prepare_sample(args) if row is None else row
    sample = policy_map["drsgui.mcts"](row, args)
    result = await sample.process()

    if args.output:
        output_path = save_json(result, args.output)
        logging.info("Saved prediction to %s", output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return result


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    row = prepare_sample(args)
    configure_runtime(args)
    initialize_models(args)
    result = asyncio.run(infer(args, row))
    if result.get("error"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
