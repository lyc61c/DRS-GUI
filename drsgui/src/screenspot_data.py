"""ScreenSpot schema normalization and point-in-box evaluation."""

import copy
import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)
GT_TYPES = ["positive"]
INSTRUCTION_STYLES = ["instruction"]
LANGUAGES = ["en"]


def normalize_task(task: dict, benchmark: str, task_filename: str, index: int) -> dict:
    """Normalize ScreenSpot v1, v2, and Pro records to one schema."""
    task = copy.deepcopy(task)
    task["id"] = task.get("id", f"{task_filename}_{index}")
    task["ui_type"] = task.get("ui_type", task.get("data_type", "unknown"))
    task["platform"] = task.get("platform", task.get("data_source", "unknown"))
    task["application"] = task.get(
        "application", task.get("data_source", task["platform"])
    )
    task["group"] = task.get("group")

    bbox_format = task.pop(
        "bbox_format",
        "xywh" if benchmark in {"screenspot_v1", "screenspot_v2"} else "xyxy",
    )
    if bbox_format == "xywh":
        x, y, width, height = task["bbox"]
        task["bbox"] = [x, y, x + width, y + height]
    elif bbox_format != "xyxy":
        raise ValueError(f"Unsupported bbox_format={bbox_format!r} in {task_filename}.json")
    return task


def _metrics(results: list[dict]) -> dict:
    text_results = [item for item in results if item.get("ui_type") == "text"]
    icon_results = [item for item in results if item.get("ui_type") == "icon"]
    correct = sum(item.get("correctness") == "correct" for item in results)
    return {
        "num_correct_action": correct,
        "num_total": len(results),
        "wrong_format_num": sum(
            item.get("correctness") == "wrong_format" for item in results
        ),
        "action_acc": correct / len(results) if results else 0,
        "text_acc": (
            sum(item.get("correctness") == "correct" for item in text_results)
            / len(text_results)
            if text_results
            else 0
        ),
        "icon_acc": (
            sum(item.get("correctness") == "correct" for item in icon_results)
            / len(icon_results)
            if icon_results
            else 0
        ),
    }


def _grouped_metrics(
    results: list[dict], fields: list[tuple[str, str]]
) -> dict[str, dict]:
    groups: dict[tuple, list[dict]] = {}
    for result in results:
        values = tuple(result.get(field) for field, _ in fields)
        groups.setdefault(values, []).append(result)

    report = {}
    for values, group in groups.items():
        key = " ".join(
            f"{label}:{value}" for (_, label), value in zip(fields, values)
        )
        report[key] = _metrics(group)
    return report


def evaluate(results: list[dict]) -> dict:
    """Return detailed predictions and the benchmark's standard metric views."""
    overall = _metrics(results)
    LOGGER.info("Overall grounding accuracy: %.4f", overall["action_acc"])
    return {
        "details": results,
        "metrics": {
            "fine_grained": _grouped_metrics(
                results,
                [
                    ("platform", "plat"),
                    ("application", "app"),
                    ("instruction_style", "inst_style"),
                    ("gt_type", "gt_type"),
                ],
            ),
            "seeclick_style": _grouped_metrics(
                results,
                [
                    ("platform", "plat"),
                    ("instruction_style", "inst_style"),
                    ("gt_type", "gt_type"),
                ],
            ),
            "leaderboard_simple_style": _grouped_metrics(results, [("group", "group")]),
            "leaderboard_detailed_style": _grouped_metrics(
                results, [("application", "app")]
            ),
            "overall": overall,
        },
    }


def _selected(value: str, supported: list[str]) -> list[str]:
    return supported if value == "all" else value.split(",")


def _flatten_annotations(records: list[dict]) -> list[dict]:
    """Accept both official image-level records and flat target records."""
    flattened = []
    for record in records:
        annotations = record.get("annotations")
        if not isinstance(annotations, list):
            flattened.append(record)
            continue
        shared = {key: value for key, value in record.items() if key != "annotations"}
        for annotation in annotations:
            flattened.append({**shared, **annotation})
    return flattened


def get_tasks(args) -> tuple[dict | None, list[dict]]:
    annotation_dir = Path(args.screenspot_test)
    if args.task == "all":
        filenames = sorted(path.stem for path in annotation_dir.glob("*.json"))
    else:
        filenames = args.task.split(",")

    instruction_styles = _selected(args.inst_style, INSTRUCTION_STYLES)
    languages = _selected(args.language, LANGUAGES)
    ground_truth_types = _selected(args.gt_type, GT_TYPES)
    tasks = []

    for filename in filenames:
        path = annotation_dir / f"{filename}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        records = _flatten_annotations(data)
        for instruction_style in instruction_styles:
            for ground_truth_type in ground_truth_types:
                for language in languages:
                    if language != "en":
                        raise ValueError("The public DRS-GUI runner currently supports English only")
                    for index, raw_task in enumerate(records):
                        task = normalize_task(raw_task, args.benchmark, filename, index)
                        task.update(
                            {
                                "task_filename": filename,
                                "gt_type": ground_truth_type,
                                "instruction_style": instruction_style,
                                "language": language,
                                "prompt_to_evaluate": task["instruction"],
                            }
                        )
                        tasks.append(task)
        LOGGER.info("Loaded %d samples from %s", len(records), path)

    LOGGER.info("Loaded %d benchmark tasks", len(tasks))
    return (tasks[-1] if tasks else None), tasks
