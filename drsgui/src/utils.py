"""Small shared helpers."""

import json
import math
from pathlib import Path


def get_chunk(items, number_of_chunks, chunk_index):
    if number_of_chunks < 1:
        raise ValueError("number_of_chunks must be positive")
    if not 0 <= chunk_index < number_of_chunks:
        raise ValueError("chunk_index must satisfy 0 <= chunk_index < number_of_chunks")
    chunk_size = max(1, math.ceil(len(items) / number_of_chunks))
    start = chunk_index * chunk_size
    return items[start : start + chunk_size]


def valid_point(point, normalized=False):
    """Reject malformed, non-finite, and out-of-range model coordinates."""
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return False
    try:
        values = [float(value) for value in point]
    except (TypeError, ValueError, OverflowError):
        return False
    return all(math.isfinite(value) for value in values) and (
        not normalized or all(0 <= value <= 1 for value in values)
    )


def load_rgb_image(image):
    """Accept a local path or a PIL image at the grounding-model boundary."""
    from PIL import Image

    if isinstance(image, (str, Path)):
        path = Path(image).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Screenshot not found: {path}")
        with Image.open(path) as source:
            return source.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError("image must be a local image path or PIL.Image.Image")


def save_json(data, path):
    """Write strict UTF-8 JSON; replace the destination only after a full write."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    temporary_path.replace(path)
    return path
