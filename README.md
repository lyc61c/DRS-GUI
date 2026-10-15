<div align="center">

# DRS-GUI

### Dynamic Region Search for Training-Free GUI Grounding

Find a useful region first. Ground the target within that region.

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-required-76B900?logo=nvidia&logoColor=white)
![Grounding models](https://img.shields.io/badge/Grounding-Qwen2.5--VL%20%7C%20UGround-blue)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](LICENSE.txt)

[Overview](#overview) · [Setup](#setup) · [Models](#models) · [Inference](#single-image-inference) · [Outputs](#outputs-and-search-inspection) · [Code](#code-organization)

</div>

## Overview

DRS-GUI is a training-free search layer for screenshot-based GUI grounding. Given a screenshot and an instruction, it parses the interface, explores candidate fields of view, and sends the best region to a grounding model. The predicted click point is then mapped back to the original screenshot.

The implementation brings together three stages:

| Stage | Component | Responsibility |
| :--- | :--- | :--- |
| Perceive | OmniParser V2 + PaddleOCR + INSTRUCTOR | Detect UI elements, read text, caption icons, and score semantic relevance |
| Search | MCTS action planner | Explore candidate regions through Focus, Shift, and Scatter |
| Ground | Qwen2.5-VL or UGround-V1 | Predict the target point from the selected crop |

![DRS-GUI method overview](assets/drsgui_method.png)

*Method overview (Figure 2 of the [DRS-GUI paper](https://arxiv.org/abs/2605.15542), CC BY 4.0).*

### Region-search actions

| Action | Field-of-view change | Purpose |
| :--- | :--- | :--- |
| Focus | Contract around relevant elements | Reduce visual distraction |
| Shift | Move to another relevant region | Explore targets outside the current view |
| Scatter | Expand the current view | Recover surrounding context |

The search evaluates regions using relevance, coverage, and concentration. By default it uses **8 simulations** and a **maximum depth of 3**. MCTS does not call the base grounding model during search; coordinate prediction happens after region selection.

### Features

- Single-image grounding from a screenshot and a natural-language instruction.
- Interchangeable Qwen2.5-VL and UGround-V1 grounding adapters.
- Original-image click coordinates, selected regions, and optional search-tree inspection.

## Setup

Requirements: **Python 3.10** and an **NVIDIA GPU with CUDA-enabled PyTorch**.

Run the following from the repository root:

```bash
git clone https://github.com/lyc61c/DRS-GUI.git
cd DRS-GUI

conda create -n drsgui python=3.10 -y
conda activate drsgui

# CUDA 12.1 example; use a PyTorch build compatible with your driver.
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The CUDA check should print `True`. The dependency versions in `requirements.txt` are based on the project's `gui` environment.

You can inspect the single-image entry point without loading models:

```bash
python drsgui/src/infer.py --help
```

## Models

Download the perception components and **at least one** grounding model. The directories below are conventions; existing local model paths also work.

| Component | Source | Suggested local path |
| :--- | :--- | :--- |
| Icon detector and captioner | [OmniParser V2](https://huggingface.co/microsoft/OmniParser-v2.0) | `checkpoints/OmniParser-v2.0/` |
| Caption processor | [Florence-2-base](https://huggingface.co/microsoft/Florence-2-base) | `checkpoints/Florence-2-base/` |
| Semantic encoder | [INSTRUCTOR-large](https://huggingface.co/hkunlp/instructor-large) | `checkpoints/instructor-large/` |
| Grounding option A | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `checkpoints/Qwen2.5-VL-7B-Instruct/` |
| Grounding option B | [UGround-V1-7B](https://huggingface.co/osunlp/UGround-V1-7B) | `checkpoints/UGround-V1-7B/` |

<details>
<summary><b>Download commands</b></summary>

```bash
mkdir -p checkpoints

hf download microsoft/OmniParser-v2.0 \
  --include "icon_detect/*" "icon_caption/*" \
  --local-dir checkpoints/OmniParser-v2.0

# Processor, tokenizer, and custom-code files; no extra grounding weights.
hf download microsoft/Florence-2-base \
  --include "*.json" "*.py" "*.txt" \
  --local-dir checkpoints/Florence-2-base

hf download hkunlp/instructor-large \
  --local-dir checkpoints/instructor-large

# Choose one grounding model, or download both for comparison.
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir checkpoints/Qwen2.5-VL-7B-Instruct

hf download osunlp/UGround-V1-7B \
  --local-dir checkpoints/UGround-V1-7B
```

</details>

OmniParser's caption directory contains the caption weights; Florence-2-base supplies the processor. Use `--caption-processor` for an explicit local processor path. If omitted, the loader tries the caption directory and then the Florence-2-base cache/Hub. PaddleOCR initializes on the first parsing call and may download its OCR weights.

### Model arguments

The single-image command below uses this Bash array. Define it once in the same terminal session:

```bash
MODEL_ARGS=(
  --model-type qwen2_5vl
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct
  --detector-path checkpoints/OmniParser-v2.0/icon_detect/model.pt
  --caption-model checkpoints/OmniParser-v2.0/icon_caption
  --caption-processor checkpoints/Florence-2-base
  --instructor-model checkpoints/instructor-large
)
```

To switch to UGround-V1, replace the first two entries with:

```text
--model-type ugroundv1
--model-path checkpoints/UGround-V1-7B
```

## Single-image inference

Only one screenshot and a natural-language instruction are required; no dataset or annotation files are needed. After defining `MODEL_ARGS`, run:

```bash
CUDA_VISIBLE_DEVICES=0 python drsgui/src/infer.py \
  "${MODEL_ARGS[@]}" \
  --image /path/to/screenshot.png \
  --instruction "Click the Settings button" \
  --platform windows \
  --application unknown \
  --output outputs/prediction.json
```

`--platform` and `--application` are optional semantic hints. The result is printed as JSON and, when `--output` is provided, saved to disk.

The `pred` field is the click point in **original-screenshot pixels**, not crop-relative coordinates. This entry point predicts a location; it does not execute a mouse click.

### Useful options

| Option | Default | Use |
| :--- | :--- | :--- |
| `--mcts-iterations` | `8` | Search simulations per sample |
| `--max-depth` | `3` | Maximum action-search depth |
| `--seed` | `114514` | Random seed for the runtime |
| `--include-search-tree` | Off | Save structured and readable search trees |

## Outputs and search inspection

Inference prints one JSON object and optionally writes it to `--output`:

| Field | Content |
| :--- | :--- |
| `id`, `img_path`, `img_size`, `instruction` | Input screenshot identity, dimensions, and instruction |
| `pred` | Predicted `[x, y]` click point in original-image pixels, or `null` if unavailable |
| `best_region` | Selected `[x1, y1, x2, y2]` region in original-image pixels |
| `action_history` | Focus / Shift / Scatter actions leading to the selected region |
| `reward_components`, `search_summary` | Region-quality terms and search statistics |
| `raw_response`, `full_response` | Grounding-model response |
| `elapsed_seconds` | Time spent on region search and final grounding, excluding model initialization |

Each successful sample includes `pred`, `best_region`, `action_history`, `reward_components`, and `search_summary`. The following is an **illustrative subset** of the output:

```json
{
  "pred": [1280.0, 720.0],
  "best_region": [1000, 500, 1560, 940],
  "best_region_depth": 2,
  "action_history": ["focus", "shift"]
}
```

Append `--include-search-tree` to the inference command to also save `search_tree` and `search_tree_text`. Tree records contain actions, regions, rewards, and visits; they omit embedded screenshot bytes. If processing raises an exception, the output contains `pred: null`, `error`, and `traceback`, and the command exits with a nonzero status.

## Code organization

```text
DRS-GUI/
├── README.md
├── requirements.txt
├── LICENSE.txt
├── assets/
│   └── drsgui_method.png             # method overview from the paper
├── drsgui/
│   └── src/
│       ├── infer.py                  # single-image search and grounding CLI
│       ├── runtime.py                # options, seeds, model initialization
│       ├── model_factory.py          # grounding-model selection
│       ├── ui_perceptor.py           # UI parsing and semantic ranking
│       ├── utils.py                  # image, coordinate, JSON helpers
│       ├── models/                   # Qwen2.5-VL / UGround-V1 adapters
│       ├── policies/drsgui/
│       │   ├── mcts.py               # search, rewards, backpropagation, diagnostics
│       │   ├── action.py             # Focus / Shift / Scatter geometry
│       │   ├── instruction_config.py # domain-specific semantic prefixes
│       │   └── policy.py             # grounding sample and result interface
│       └── OmniParser/               # retained perception utilities and license
├── checkpoints/                      # local models; not tracked
└── outputs/                          # generated predictions; not tracked
```

For extending the project, start with `model_factory.py` and `models/` for a new grounding backend, or `policies/drsgui/` for search behavior. Grounding adapters return a normalized crop-relative `point`; the planner handles original-image coordinate mapping.

## Acknowledgements and license

DRS-GUI builds on [OmniParser](https://github.com/microsoft/OmniParser), [INSTRUCTOR](https://github.com/xlang-ai/instructor-embedding), [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), and [UGround](https://github.com/OSU-NLP-Group/UGround).

Project code is released under [Apache 2.0](LICENSE.txt). The retained OmniParser utilities carry their [separate license](drsgui/src/OmniParser/LICENSE). Downloaded model weights remain governed by their upstream licenses.
