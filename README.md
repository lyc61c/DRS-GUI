# DRS-GUI

DRS-GUI adds a training-free dynamic region search stage before coordinate prediction. Its UI Perceptor uses OmniParser V2 to parse GUI elements and INSTRUCTOR to match those elements to the user instruction. An MCTS Action Planner then schedules three reversible field-of-view actions:

- **Focus** contracts the view around instruction-relevant elements.
- **Shift** moves the view to a distant relevant region.
- **Scatter** expands the view to recover missing context.

The best region found by MCTS is passed to a base grounding model. This release supports the two model families used in the paper: [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and [UGround-V1-7B](https://huggingface.co/osunlp/UGround-V1-7B).

## Requirements

DRS-GUI currently assumes the following environment:

- Linux and Python 3.10.
- An NVIDIA GPU with a CUDA-enabled PyTorch installation. CPU-only inference is not supported by the current entry points.
- Enough GPU memory to load one 7B grounding model together with OmniParser's detector and Florence-2 caption model. A GPU with at least 24 GB VRAM is recommended; actual usage depends on screenshot resolution and the selected model.
- Approximately 45 GB of free disk space for both grounding models, OmniParser V2, INSTRUCTOR-large, and all three benchmarks. Downloading only one grounding model requires substantially less space.
- Internet access for the first download, or complete local copies of every model listed below.

Create the environment from the repository root:

```bash
conda create -n drsgui python=3.10 -y
conda activate drsgui

# Install a CUDA build of PyTorch that matches your driver first.
# The following is only an example for CUDA 12.1.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Verify that PyTorch can see the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The second value must be `True`. If your cluster already provides a suitable PyTorch build, keep that build and install the remaining requirements normally.

## Repository layout

```text
DRS-GUI/
├── checkpoints/                 # downloaded model weights (gitignored)
├── data/                        # downloaded screenshots (gitignored)
├── outputs/                     # predictions and metrics (gitignored)
└── drsgui/
    ├── scripts/
    │   ├── run_screenspot_v1.sh
    │   ├── run_screenspot_v2.sh
    │   └── run_screenspot_pro.sh
    └── src/
        ├── infer.py             # single-screenshot inference
        ├── run.py               # benchmark evaluation
        ├── screenspot_data.py   # v1/v2/Pro schema normalization
        ├── ui_perceptor.py
        ├── models/              # Qwen2.5-VL and UGround adapters
        └── policies/drsgui/     # MCTS planner and three actions
```

Model weights, screenshots, generated crops, logs, and outputs are intentionally excluded from Git.

## Download model weights

Install the Hugging Face CLI through `requirements.txt`, then create the local model directory:

```bash
mkdir -p checkpoints
# Optional for gated/private repositories: hf auth login
```

### 1. OmniParser V2

DRS-GUI needs both the icon detector and the Florence-2 icon caption model from [microsoft/OmniParser-v2.0](https://huggingface.co/microsoft/OmniParser-v2.0):

```bash
hf download microsoft/OmniParser-v2.0 \
  --local-dir checkpoints/OmniParser-v2.0
```

The two paths supplied at runtime will be:

```text
checkpoints/OmniParser-v2.0/icon_detect/model.pt
checkpoints/OmniParser-v2.0/icon_caption/
```

### 2. Semantic embedding model

```bash
hf download hkunlp/instructor-large \
  --local-dir checkpoints/instructor-large
```

You may also pass the Hub ID `hkunlp/instructor-large` directly, but a local path makes offline runs reproducible.

### 3. Base grounding model

At least one of the following is required:

```bash
# Qwen2.5-VL used with --model-type qwen2_5vl
hf download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir checkpoints/Qwen2.5-VL-7B-Instruct

# UGround-V1 used with --model-type ugroundv1
hf download osunlp/UGround-V1-7B \
  --local-dir checkpoints/UGround-V1-7B
```

After downloading all optional weights, the relevant structure is:

```text
checkpoints/
├── OmniParser-v2.0/
│   ├── icon_detect/model.pt
│   └── icon_caption/
├── instructor-large/
├── Qwen2.5-VL-7B-Instruct/
└── UGround-V1-7B/
```

## Download benchmarks

The `--images` argument must point to the directory relative to which every annotation's `img_filename` can be resolved. Pass the directory containing the benchmark JSON files through `--annotations`.

### ScreenSpot v1

The canonical ScreenSpot v1 release is maintained in the [SeeClick repository](https://github.com/njucckevin/SeeClick). The following Hugging Face mirror keeps the original `images/` and `annotations/` directory layout and is convenient for command-line download:

```bash
hf download benwiesel/ScreenSpot \
  --repo-type dataset \
  --local-dir data/screenspot_v1
```

Expected paths:

```text
data/screenspot_v1/images/<image files>
data/screenspot_v1/annotations/screenspot_desktop.json
data/screenspot_v1/annotations/screenspot_mobile.json
data/screenspot_v1/annotations/screenspot_web.json
```

If you use the original SeeClick download instead, pass its image and annotation directories directly.

### ScreenSpot v2

Download the official [OS-Copilot/ScreenSpot-v2](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) release and extract the image archive:

```bash
hf download OS-Copilot/ScreenSpot-v2 \
  --repo-type dataset \
  --local-dir data/screenspot_v2

unzip data/screenspot_v2/screenspotv2_image.zip -d data/screenspot_v2
```

Expected paths:

```text
data/screenspot_v2/screenspotv2_image/<image files>
data/screenspot_v2/screenspot_desktop_v2.json
data/screenspot_v2/screenspot_mobile_v2.json
data/screenspot_v2/screenspot_web_v2.json
```

### ScreenSpot-Pro

This repository includes the ScreenSpot-Pro annotation JSON files. Download the official [likaixin/ScreenSpot-Pro](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro) dataset to obtain the screenshots:

```bash
hf download likaixin/ScreenSpot-Pro \
  --repo-type dataset \
  --local-dir data/screenspot_pro
```

Use `data/screenspot_pro/images` as `--images` and `data/screenspot_pro/annotations` as `--annotations`.

### Annotation compatibility

The official v1 and v2 boxes are `[x, y, width, height]`; the loader converts them to the internal `[x1, y1, x2, y2]` representation. ScreenSpot-Pro already uses `[x1, y1, x2, y2]`.

For custom preprocessed v1/v2 JSON in `[x1, y1, x2, y2]` format, add `"bbox_format": "xyxy"` to each record to prevent conversion. Missing IDs, image sizes, applications, platforms, and UI types are filled when possible.

## Run benchmark evaluation

For each sample, DRS-GUI first parses the screenshot and searches for the highest-reward region. Only that selected crop is then sent to the base grounding model for coordinate prediction; MCTS itself does not invoke the base MLLM. The default search budget and depth match the paper: 8 simulations and a maximum depth of 3.

The released planner intentionally retains the implementation defaults used by this codebase: Focus selects 10% of elements, Scatter selects 8%, UCT includes the `sqrt(2)` exploration factor, and the available actions are filtered by region area. The paper reports 15% for Focus and 10% for Scatter and presents the standard UCT expression. The three-term reward follows the paper with weights `0.4/0.4/0.2`; because the paper does not state the non-interactive weight or semantic temperature, this release fixes them to `0.5` and `0.1`, respectively.

The examples below use Qwen2.5-VL and locally downloaded weights.

### ScreenSpot v1

```bash
CUDA_VISIBLE_DEVICES=0 bash drsgui/scripts/run_screenspot_v1.sh \
  --model-type qwen2_5vl \
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct \
  --images data/screenspot_v1/images \
  --annotations data/screenspot_v1/annotations \
  --detector-path checkpoints/OmniParser-v2.0/icon_detect/model.pt \
  --caption-model checkpoints/OmniParser-v2.0/icon_caption \
  --instructor-model checkpoints/instructor-large \
  --output outputs/qwen2_5vl_screenspot_v1.json
```

### ScreenSpot v2

```bash
CUDA_VISIBLE_DEVICES=0 bash drsgui/scripts/run_screenspot_v2.sh \
  --model-type qwen2_5vl \
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct \
  --images data/screenspot_v2/screenspotv2_image \
  --annotations data/screenspot_v2 \
  --detector-path checkpoints/OmniParser-v2.0/icon_detect/model.pt \
  --caption-model checkpoints/OmniParser-v2.0/icon_caption \
  --instructor-model checkpoints/instructor-large \
  --output outputs/qwen2_5vl_screenspot_v2.json
```

### ScreenSpot-Pro

```bash
CUDA_VISIBLE_DEVICES=0 bash drsgui/scripts/run_screenspot_pro.sh \
  --model-type qwen2_5vl \
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct \
  --images data/screenspot_pro/images \
  --annotations data/screenspot_pro/annotations \
  --detector-path checkpoints/OmniParser-v2.0/icon_detect/model.pt \
  --caption-model checkpoints/OmniParser-v2.0/icon_caption \
  --instructor-model checkpoints/instructor-large \
  --output outputs/qwen2_5vl_screenspot_pro.json
```

To evaluate UGround-V1, change only these two options:

```text
--model-type ugroundv1
--model-path checkpoints/UGround-V1-7B
```

Useful evaluation options:

- `--annotations /path/to/jsons`: directory containing the benchmark JSON files.
- `--task screenspot_web_v2`: run one JSON file; use comma-separated stems for multiple files.
- `--mcts-iterations 8`: number of MCTS simulations for each sample.
- `--max-depth 3`: maximum action-search depth.
- `--num-chunks N --chunk-idx K`: evaluate chunk `K` of `N`; give every chunk a different `--output` path.

Each output JSON contains per-sample predictions plus overall, text/icon, platform, application, and group metrics. Grounding accuracy is point-in-bounding-box accuracy.

## Run single-image inference

For a screenshot that has no benchmark annotation, provide the screenshot and the GUI element instruction directly:

```bash
CUDA_VISIBLE_DEVICES=0 python drsgui/src/infer.py \
  --image examples/example.png \
  --instruction "click the Settings button" \
  --platform windows \
  --application unknown \
  --model-type qwen2_5vl \
  --model-path checkpoints/Qwen2.5-VL-7B-Instruct \
  --detector-path checkpoints/OmniParser-v2.0/icon_detect/model.pt \
  --caption-model checkpoints/OmniParser-v2.0/icon_caption \
  --instructor-model checkpoints/instructor-large \
  --output outputs/example_prediction.json
```

`platform` and `application` are optional hints used to choose a domain-specific semantic prompt. Supported platform examples include `windows`, `macos`, `linux`, `web`, `ios`, and `android`. The returned `pred` field is the predicted `[x, y]` point in the original screenshot's pixel coordinate system.

## License

DRS-GUI is released under the [Apache License 2.0](LICENSE.txt). The retained OmniParser utility code is covered by its own license in `drsgui/src/OmniParser/LICENSE`. Dataset and model weights remain subject to their respective upstream licenses.
