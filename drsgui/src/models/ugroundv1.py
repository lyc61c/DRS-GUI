"""Minimal UGround-V1 adapter used by DRS-GUI."""

import os
import re
import tempfile

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.generation import GenerationConfig


class UGroundV1Model:
    def load_model(self, model_name_or_path="osunlp/UGround-V1-7B"):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.generation_config = {
            "do_sample": False,
            "temperature": 0.0,
            "use_cache": False,
            "max_new_tokens": 256,
        }
        self.set_generation_config()

    def set_generation_config(self, **kwargs):
        self.generation_config.update(kwargs)
        self.model.generation_config = GenerationConfig(**self.generation_config)

    @staticmethod
    def _parse_point(response):
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", response)]
        if len(values) >= 4:
            x = (values[-4] + values[-2]) / 2
            y = (values[-3] + values[-1]) / 2
        elif len(values) >= 2:
            x, y = values[-2:]
        else:
            return None
        return [x / 1000.0, y / 1000.0]

    def ground_only_positive(self, instruction, image):
        temporary_path = None
        if isinstance(image, Image.Image):
            handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temporary_path = handle.name
            handle.close()
            image.convert("RGB").save(temporary_path)
            image_path = temporary_path
        else:
            image_path = image

        try:
            prompt = (
                "Identify the precise point (x, y) of the GUI element described below. "
                "Return only the coordinate. Coordinates use a 0-1000 scale.\n\n"
                f"Description: {instruction}"
            )
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt},
                ],
            }]
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, videos=videos, padding=True, return_tensors="pt"
            ).to(self.model.device)
            generated = self.model.generate(**inputs)
            trimmed = [out[len(source):] for source, out in zip(inputs.input_ids, generated)]
            response = self.processor.batch_decode(
                trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )[0]
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)

        return {
            "result": "positive",
            "format": "point",
            "raw_response": response,
            "bbox": None,
            "point": self._parse_point(response),
        }

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError("DRS-GUI evaluates positive grounding samples only")
