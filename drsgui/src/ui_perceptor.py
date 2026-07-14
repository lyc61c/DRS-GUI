"""UI parsing and instruction-conditioned semantic scoring."""

import base64
import io

import numpy as np
from PIL import Image

from OmniParser.util.utils import check_ocr_box, get_som_labeled_img


def get_topk_semantic_matches(
    model,
    gui_prompt: str,
    domain_prefix: str,
    parsed_content_list: list[dict],
    topk: int = 10,
) -> list[dict]:
    """Return UI elements ranked by cosine similarity to the instruction."""
    elements = [
        element
        for element in parsed_content_list
        if isinstance(element.get("content"), str)
        and element["content"].strip()
        and isinstance(element.get("bbox"), (list, tuple))
        and len(element["bbox"]) == 4
    ]
    if not elements:
        return []

    instruction_embedding = model.encode(
        [gui_prompt], prompt=domain_prefix, normalize_embeddings=True
    )
    element_embeddings = model.encode(
        [element["content"] for element in elements],
        prompt=domain_prefix,
        normalize_embeddings=True,
    )
    similarities = np.asarray(instruction_embedding) @ np.asarray(element_embeddings).T
    similarities = similarities[0]

    count = min(max(1, topk), len(elements))
    indices = np.argsort(similarities)[-count:][::-1]
    ranked = []
    for index in indices:
        element = dict(elements[index])
        element["similarity"] = float(similarities[index])
        ranked.append(element)
    return ranked


def analyze_ui_image_simple(
    image_base64: str,
    som_model,
    caption_model_processor,
    box_threshold: float = 0.05,
    draw_bbox_scale: int = 3200,
    iou_threshold: float = 0.7,
    batch_size: int = 128,
    ocr_text_threshold: float = 0.9,
) -> tuple:
    """Parse a base64 screenshot into OmniParser UI elements."""
    image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
    ocr_result, _ = check_ocr_box(
        image,
        display_img=False,
        output_bb_format="xyxy",
        goal_filtering=None,
        text_threshold=ocr_text_threshold,
    )
    ocr_text, ocr_boxes = ocr_result

    overlay_ratio = max(image.size) / draw_bbox_scale
    draw_config = {
        "text_scale": 0.8 * overlay_ratio,
        "text_thickness": max(int(2 * overlay_ratio), 1),
        "text_padding": max(int(3 * overlay_ratio), 1),
        "thickness": max(int(3 * overlay_ratio), 1),
    }
    result = get_som_labeled_img(
        image,
        som_model,
        BOX_TRESHOLD=box_threshold,
        output_coord_in_ratio=True,
        ocr_bbox=ocr_boxes,
        draw_bbox_config=draw_config,
        caption_model_processor=caption_model_processor,
        ocr_text=ocr_text,
        use_local_semantics=True,
        iou_threshold=iou_threshold,
        scale_img=False,
        batch_size=batch_size,
    )
    return result if result is not None else (None, [], [])
