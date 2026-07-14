"""Run DRS-GUI on ScreenSpot-format annotations."""

import argparse
import asyncio
import json
import logging
import random
from pathlib import Path

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from model_factory import build_model
from OmniParser.util.utils import get_caption_model_processor, get_yolo_model
from policies import policy_map
from screenspot_data import evaluate, get_tasks
from utils import get_chunk


BENCHMARKS = ("screenspot_v1", "screenspot_v2", "screenspot_pro")


def parse_args():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="DRS-GUI evaluation")
    parser.add_argument("--benchmark", choices=BENCHMARKS, default="screenspot_pro")
    parser.add_argument("--model-type", choices=["qwen2_5vl", "ugroundv1"], default="qwen2_5vl")
    parser.add_argument("--model-path", required=True, help="Hugging Face model ID or local model path")
    parser.add_argument("--images", required=True, help="Root directory containing benchmark screenshots")
    parser.add_argument(
        "--annotations",
        required=True,
        help="Directory containing ScreenSpot-format JSON annotations",
    )
    parser.add_argument("--detector-path", required=True, help="OmniParser icon detector model.pt")
    parser.add_argument("--caption-model", required=True, help="Florence-2 caption model ID or local path")
    parser.add_argument("--instructor-model", default="hkunlp/instructor-large")
    parser.add_argument("--output", default=None)
    parser.add_argument("--task", default="all", help="Comma-separated annotation filenames without .json")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--mcts-iterations", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=3)
    args = parser.parse_args()
    if args.output is None:
        args.output = str(project_root / "outputs" / f"{args.model_type}_{args.benchmark}.json")
    return args


def attach_grounding_metrics(result, row):
    result["bbox"] = row.get("bbox")
    pred = result.get("pred")
    bbox = row.get("bbox")
    width, height = row.get("img_size", [0, 0])

    if pred and len(pred) == 2 and width and height:
        result["pred_normalized"] = [pred[0] / width, pred[1] / height]
    else:
        result["pred_normalized"] = None

    if pred and bbox and len(pred) == 2 and len(bbox) == 4:
        x, y = pred
        result["correctness"] = "correct" if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3] else "wrong"
    else:
        result["correctness"] = "wrong_format"
    return result


def initialize_models(args):
    """Load the grounding model and the three DRS-GUI perception components."""
    args.model = build_model(args.model_type, args.model_path)
    args.som_model = get_yolo_model(args.detector_path).to("cuda")
    args.caption_model = get_caption_model_processor(
        model_name="florence2",
        model_name_or_path=args.caption_model,
        device="cuda",
    )
    args.semantic_model = SentenceTransformer(args.instructor_model)
    return args


async def evaluate_model(args):
    _, tasks = get_tasks(args)
    tasks = get_chunk(tasks, args.num_chunks, args.chunk_idx)
    results = []

    for row in tqdm(tasks, desc="DRS-GUI"):
        image_path = Path(args.images) / row["img_filename"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {image_path}")
        row["img_filename"] = str(image_path)
        if not row.get("img_size"):
            with Image.open(image_path) as image:
                row["img_size"] = [image.width, image.height]
        sample = policy_map["drsgui.mcts"](row, args)
        result = await sample.process()
        results.append(attach_grounding_metrics(result, row))

    report = evaluate(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("Saved %d predictions to %s", len(results), output_path)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("DRS-GUI currently requires a CUDA-capable GPU")
    random.seed(114514)
    torch.manual_seed(114514)

    annotation_path = Path(args.annotations)
    if not annotation_path.is_dir() or not any(annotation_path.glob("*.json")):
        raise FileNotFoundError(
            f"No {args.benchmark} annotation JSON files found in {annotation_path}. "
            "Download the official benchmark annotations and check --annotations."
        )

    args.screenspot_imgs = args.images
    args.screenspot_test = args.annotations
    args.inst_style = "instruction"
    args.language = "en"
    args.gt_type = "positive"
    args.method_name = "drsgui.mcts"

    initialize_models(args)

    asyncio.run(evaluate_model(args))


if __name__ == "__main__":
    main()
