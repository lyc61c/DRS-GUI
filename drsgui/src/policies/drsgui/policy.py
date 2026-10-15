"""Base class for a single-image DRS-GUI grounding sample."""

from abc import ABC, abstractmethod
import base64
import io
import logging
import traceback

from PIL import Image


def result_metadata(row):
    """Use the same sample identity for successful and failed predictions."""
    return {
        "id": row["id"],
        "img_path": row["img_filename"],
        "img_size": row.get("img_size"),
        "platform": row.get("platform", "unknown"),
        "application": row.get("application", "unknown"),
        "instruction": row["instruction"],
    }


class QuestionSample(ABC):
    def __init__(self, row, args):
        self.row = row
        self.args = args
        with open(row["img_filename"], "rb") as image_file:
            self.image = base64.b64encode(image_file.read()).decode("utf-8")
        self.model = args.model
        self.som_model = args.som_model
        self.caption_model = args.caption_model
        self.semantic_model = args.semantic_model

    async def generate(self, instruction, image):
        pil_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        return self.model.ground_only_positive(instruction=instruction, image=pil_image)

    async def process(self):
        try:
            return await self._process()
        except Exception as error:
            logging.exception("Failed to process sample %s", self.row["id"])
            return {
                **result_metadata(self.row),
                "pred": None,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

    @abstractmethod
    async def _process(self):
        raise NotImplementedError
