"""Minimal UGround-V1 adapter used by DRS-GUI."""

import re

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from transformers.generation import GenerationConfig
from utils import load_rgb_image, valid_point


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
        number = r"-?\d+(?:\.\d+)?"
        matches = re.findall(rf"[\[(]\s*({number}(?:\s*,\s*{number}){{1,3}})\s*[\])]", response)
        if not matches:
            bare_point = re.fullmatch(rf"\s*({number}\s*,\s*{number})\s*", response)
            if bare_point is None:
                return None
            matches = [bare_point.group(1)]
        values = [float(value.strip()) for value in matches[-1].split(",")]
        if len(values) == 4:
            x = (values[-4] + values[-2]) / 2
            y = (values[-3] + values[-1]) / 2
        elif len(values) == 2:
            x, y = values
        else:
            return None
        point = [x / 1000.0, y / 1000.0]
        return point if valid_point(point, normalized=True) else None

    @torch.inference_mode()
    def ground_only_positive(self, instruction, image):
        image = load_rgb_image(image)
        prompt = (
            "Identify the precise point (x, y) of the GUI element described below. "
            "Return only the coordinate. Coordinates use a 0-1000 scale.\n\n"
            f"Description: {instruction}"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "result": "positive",
            "format": "point",
            "raw_response": response,
            "bbox": None,
            "point": self._parse_point(response),
        }

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError("DRS-GUI evaluates positive grounding samples only")
