"""Run DRS-GUI on ScreenSpot-format annotations."""

import argparse
import asyncio
import logging
from pathlib import Path

from runtime import (
    BENCHMARKS,
    add_runtime_arguments,
    configure_runtime,
    initialize_models,
    positive_int,
)
from screenspot_data import evaluate, get_tasks
from utils import get_chunk, save_json, valid_point


def parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="DRS-GUI evaluation")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="screenspot_pro")
    parser.add_argument("--images", required=True, help="Root directory containing benchmark screenshots")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Directory containing ScreenSpot-format JSON annotations",
    )
    parser.add_argument("--task", default="all", help="Comma-separated annotation filenames without .json")
    parser.add_argument("--num-chunks", type=positive_int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if not 0 <= args.chunk_idx < args.num_chunks:
        parser.error("--chunk-idx must satisfy 0 <= chunk-idx < num-chunks")
    if args.output is None:
        args.output = str(project_root / "outputs" / f"{args.model_type}_{args.benchmark}.json")
    return args


def attach_grounding_metrics(result, row):
    result["bbox"] = row.get("bbox")
    pred = result.get("pred")
    bbox = row.get("bbox")
    width, height = row["img_size"]

    if valid_point(pred):
        result["pred_normalized"] = [float(pred[0]) / width, float(pred[1]) / height]
    else:
        result["pred_normalized"] = None

    if valid_point(pred) and bbox:
        x, y = map(float, pred)
        inside_box = bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]
        result["correctness"] = "correct" if inside_box else "wrong"
    else:
        result["correctness"] = "wrong_format"
    return result


def prepare_tasks(args):
    """Validate dataset paths before allocating GPU memory."""
    from PIL import Image

    args.screenspot_test = str(Path(args.annotations).expanduser())
    args.inst_style = "instruction"
    args.language = "en"
    args.gt_type = "positive"
    _, tasks = get_tasks(args)
    tasks = get_chunk(tasks, args.num_chunks, args.chunk_idx)
    if not tasks:
        raise ValueError("No samples selected; check --task and the requested chunk")
    image_root = Path(args.images).expanduser().resolve()
    for row in tasks:
        image_path = image_root / row["img_filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {image_path}")
        row["img_filename"] = str(image_path)
        if not row.get("img_size"):
            with Image.open(image_path) as image:
                row["img_size"] = [image.width, image.height]
        size = row["img_size"]
        if (
            not isinstance(size, (list, tuple))
            or not valid_point(size)
            or any(float(value) <= 0 for value in size)
        ):
            raise ValueError(f"Invalid image size for {image_path}")
        row["img_size"] = [float(value) for value in size]
    return tasks


async def evaluate_model(args, tasks):
    from tqdm import tqdm

    from policies import policy_map
    from policies.drsgui.policy import result_metadata

    results = []
    for row in tqdm(tasks, desc="DRS-GUI"):
        try:
            sample = policy_map["drsgui.mcts"](row, args)
            result = await sample.process()
        except Exception as error:
            logging.exception("Could not initialize sample %s", row["id"])
            result = {**result_metadata(row), "pred": None, "error": str(error)}
        results.append(attach_grounding_metrics(result, row))

    report = evaluate(results)
    output_path = save_json(report, args.output)
    logging.info("Saved %d predictions to %s", len(results), output_path)
    return report


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    tasks = prepare_tasks(args)
    configure_runtime(args)
    initialize_models(args)

    asyncio.run(evaluate_model(args, tasks))


if __name__ == "__main__":
    main()
