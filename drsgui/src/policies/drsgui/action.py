"""Region transformations used by the DRS-GUI action planner."""

from collections import defaultdict
import math
from typing import Callable

import numpy as np
from PIL import Image

from ui_perceptor import get_topk_semantic_matches


Box = list[float]
Element = dict


def normalized_to_pixels(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def box_union(boxes: list[Box]) -> Box:
    x1, y1, x2, y2 = zip(*boxes)
    return [min(x1), min(y1), max(x2), max(y2)]


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(first: Box, second: Box) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(first) + box_area(second) - intersection
    return intersection / (union + 1e-9)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1, y1, x2, y2 = box
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    return [
        max(0.0, x1),
        max(0.0, y1),
        min(float(width), x2),
        min(float(height), y2),
    ]


def box_center(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _valid_elements(
    elements: list[Element],
    predicate: Callable[[Box], bool],
    width: int,
    height: int,
) -> list[Element]:
    selected = []
    for element in elements:
        box = element.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            pixel_box = normalized_to_pixels(box, width, height)
            if predicate(pixel_box):
                selected.append(element)
    return selected


def _center_is_outside(box: Box, parent: Box) -> bool:
    cx, cy = box_center(box)
    px1, py1, px2, py2 = parent
    return not (px1 <= cx <= px2 and py1 <= cy <= py2)


def _expands(child: Box, parent: Box, margin: float = 4.0) -> bool:
    x1, y1, x2, y2 = child
    px1, py1, px2, py2 = parent
    return (
        x1 <= px1 - margin
        or y1 <= py1 - margin
        or x2 >= px2 + margin
        or y2 >= py2 + margin
    )


def action_focus(
    image: Image.Image,
    parsed_content_list: list[Element],
    semantic_model,
    gui_prompt: str,
    instruction: str,
    parent_abs: Box | None = None,
    top_percent: float = 0.10,
    distance_threshold: float = 1.8,
    max_area_ratio: float = 0.7,
) -> tuple[Box, list[Element]]:
    """Contract around the top semantic elements and prune spatial outliers."""
    width, height = image.size
    parent = parent_abs or [0.0, 0.0, float(width), float(height)]
    if not parsed_content_list:
        return parent, []

    count = min(len(parsed_content_list), max(3, int(len(parsed_content_list) * top_percent)))
    selected = get_topk_semantic_matches(
        semantic_model, gui_prompt, instruction, parsed_content_list, topk=count
    )
    if not selected:
        return parent, []

    boxes = [normalized_to_pixels(item["bbox"], width, height) for item in selected]
    if len(boxes) == 1:
        padding = min(10.0, 0.02 * min(width, height))
        box = boxes[0]
        region = [
            box[0] - padding,
            box[1] - padding,
            box[2] + padding,
            box[3] + padding,
        ]
        return clamp_box(region, width, height), selected

    centers = np.asarray([box_center(box) for box in boxes])
    centroid = centers.mean(axis=0)
    distances = np.linalg.norm(centers - centroid, axis=1)
    threshold = np.median(distances) * distance_threshold
    keep = distances <= threshold
    if not np.any(keep):
        keep[np.argmin(distances)] = True

    candidates = sorted(
        (
            (boxes[index], selected[index], distances[index])
            for index in range(len(boxes))
            if keep[index]
        ),
        key=lambda item: item[2],
    )
    final_boxes = [item[0] for item in candidates]
    final_elements = [item[1] for item in candidates]
    parent_area = max(1.0, box_area(parent))

    while len(final_boxes) > 1:
        region = box_union(final_boxes)
        if box_area(region) / parent_area <= max_area_ratio:
            break
        final_boxes.pop()
        final_elements.pop()

    return clamp_box(box_union(final_boxes), width, height), final_elements


def _fit_scatter_region(
    parent: Box,
    candidates: list[tuple[Box, Element]],
    max_area_ratio: float,
) -> tuple[Box, list[Element]] | None:
    parent_area = max(1.0, box_area(parent))
    parent_center = box_center(parent)
    kept = list(candidates)

    while kept:
        region = box_union([parent] + [box for box, _ in kept])
        if box_area(region) / parent_area <= max_area_ratio:
            return region, [element for _, element in kept]
        farthest = max(
            range(len(kept)),
            key=lambda index: math.dist(box_center(kept[index][0]), parent_center),
        )
        kept.pop(farthest)
    return None


def action_scatter(
    image: Image.Image,
    parsed_content_list: list[Element],
    semantic_model,
    gui_prompt: str,
    instruction: str,
    parent_abs: Box | None = None,
    top_percent: float = 0.08,
    max_area_ratio: float = 1.5,
) -> tuple[Box | None, list[Element]]:
    """Expand the current region toward relevant elements outside its boundary."""
    width, height = image.size
    parent = parent_abs or [0.0, 0.0, float(width), float(height)]
    outside = _valid_elements(
        parsed_content_list,
        lambda box: _center_is_outside(box, parent) and box_iou(box, parent) < 0.2,
        width,
        height,
    )

    if outside:
        count = min(len(outside), max(1, int(len(outside) * top_percent)))
        selected = get_topk_semantic_matches(
            semantic_model, gui_prompt, instruction, outside, topk=count
        )
        candidates = [
            (normalized_to_pixels(item["bbox"], width, height), item)
            for item in selected
        ]
        fitted = _fit_scatter_region(parent, candidates, max_area_ratio)
        if fitted is not None:
            region, elements = fitted
            region = clamp_box(region, width, height)
            if _expands(region, parent):
                return region, elements

    px1, py1, px2, py2 = parent
    cx, cy = box_center(parent)
    linear_scale = math.sqrt(max_area_ratio)
    half_width = (px2 - px1) * linear_scale / 2
    half_height = (py2 - py1) * linear_scale / 2
    fallback = clamp_box(
        [cx - half_width, cy - half_height, cx + half_width, cy + half_height],
        width,
        height,
    )
    return (fallback, []) if _expands(fallback, parent) else (None, [])


def _direction(center: tuple[float, float], origin: tuple[float, float], tolerance: float) -> str:
    dx, dy = center[0] - origin[0], center[1] - origin[1]
    if abs(dx) < tolerance:
        return "top" if dy < 0 else "bottom"
    if abs(dy) < tolerance:
        return "left" if dx < 0 else "right"
    return f"{'top' if dy < 0 else 'bottom'}_{'left' if dx < 0 else 'right'}"


def action_shift(
    image: Image.Image,
    parsed_content_list: list[Element],
    semantic_model,
    gui_prompt: str,
    instruction: str,
    parent_abs: Box | None = None,
    top_percent: float = 0.15,
    padding_ratio: float = 0.01,
    min_distance_ratio: float = 0.5,
    max_parent_iou: float = 0.3,
) -> tuple[Box | None, list[Element]]:
    """Relocate to a distant, directionally coherent group of relevant elements."""
    width, height = image.size
    parent = parent_abs or [0.0, 0.0, float(width), float(height)]
    parent_center = box_center(parent)
    px1, py1, px2, py2 = parent
    min_distance = math.hypot(px2 - px1, py2 - py1) * min_distance_ratio

    outside = _valid_elements(
        parsed_content_list,
        lambda box: _center_is_outside(box, parent) and box_iou(box, parent) <= max_parent_iou,
        width,
        height,
    )
    if not outside:
        return None, []

    count = min(len(outside), max(3, int(len(outside) * top_percent)))
    selected = get_topk_semantic_matches(
        semantic_model, gui_prompt, instruction, outside, topk=count
    )
    candidates = []
    for element in selected:
        box = normalized_to_pixels(element["bbox"], width, height)
        center = box_center(box)
        if math.dist(center, parent_center) >= min_distance:
            candidates.append((box, element, center))
    if not candidates:
        return None, []

    groups: dict[str, list[tuple[Box, Element, tuple[float, float]]]] = defaultdict(list)
    tolerance = min_distance * 0.3
    for candidate in candidates:
        groups[_direction(candidate[2], parent_center, tolerance)].append(candidate)
    group = max(
        groups.values(),
        key=lambda items: (len(items), sum(item[1].get("similarity", 0.0) for item in items)),
    )

    region = box_union([item[0] for item in group])
    padding = max(6.0, padding_ratio * min(width, height))
    region = clamp_box(
        [region[0] - padding, region[1] - padding, region[2] + padding, region[3] + padding],
        width,
        height,
    )
    if box_iou(region, parent) > max_parent_iou:
        return None, []
    return region, [item[1] for item in group]
