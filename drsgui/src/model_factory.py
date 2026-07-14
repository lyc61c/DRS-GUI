"""Grounding-model construction for the models evaluated in DRS-GUI."""


def build_model(model_type, model_name_or_path):
    if model_type == "qwen2_5vl":
        from models.qwen2_5vl import Qwen2_5VLModel

        model = Qwen2_5VLModel()
    elif model_type == "ugroundv1":
        from models.ugroundv1 import UGroundV1Model

        model = UGroundV1Model()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_model(model_name_or_path)
    model.set_generation_config(temperature=0.0, max_new_tokens=256)
    return model
