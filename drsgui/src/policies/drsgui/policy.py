"""Base class for a DRS-GUI evaluation sample."""

from abc import ABC, abstractmethod
import base64
import io
import logging
import traceback

from PIL import Image


def result_metadata(row, round_idx=0):
    """Use the same sample identity for successful and failed predictions."""
    return {
        "id": row["id"],
        "round_id": round_idx,
        "img_path": row["img_filename"],
        "img_size": row.get("img_size"),
        "group": row.get("group"),
        "platform": row.get("platform", "unknown"),
        "application": row.get("application", "unknown"),
        "lang": row.get("language", "en"),
        "instruction_style": row.get("instruction_style", "instruction"),
        "prompt_to_evaluate": row.get("prompt_to_evaluate", row.get("instruction", "")),
        "gt_type": row.get("gt_type", "positive"),
        "ui_type": row.get("ui_type", "unknown"),
        "task_filename": row.get("task_filename", "single_image"),
    }


class QuestionSample(ABC):
    def __init__(self, row, args, round_idx=0):
        self.row = row
        self.args = args
        self.round_idx = round_idx
        with open(row["img_filename"], "rb") as image_file:
            self.image = base64.b64encode(image_file.read()).decode("utf-8")
        self.model = args.model
        self.som_model = args.som_model
        self.caption_model = args.caption_model
        self.semantic_model = args.semantic_model

    async def generate(self, _prompt, image, max_tokens=1024):
        del max_tokens
        instruction = self.row.get("prompt_to_evaluate", self.row.get("instruction", ""))
        pil_image = Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB")
        return self.model.ground_only_positive(instruction=instruction, image=pil_image)

    async def process(self):
        try:
            return await self._process()
        except Exception as error:
            logging.exception("Failed to process sample %s", self.row["id"])
            return {
                **result_metadata(self.row, self.round_idx),
                "pred": None,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }

    @abstractmethod
    async def _process(self):
        raise NotImplementedError
