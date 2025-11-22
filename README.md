# Lymphatic Vessel Segmentation

This project provides a semi-supervised pipeline for segmenting lymphatic vessels in videos. It utilizes a **UNet++** architecture with a **ResNet34** encoder and employs the **Mean Teacher** method for semi-supervised learning.

## Key Features

* **UNet++ Architecture:** High-performance deep learning model for medical image segmentation.
* **Stitch-ViT Enhanced Model**: An alternative model incorporating Vision Transformer (ViT) with a stitching mechanism for potentially improved feature extraction.
* **Semi-Supervised Learning (Mean Teacher):** Improves accuracy by leveraging unlabeled video data through a Teacher-Student consistency mechanism.
* **Boundary-Aware Loss:** Combines Dice Loss and Boundary Loss for precise edge detection.
* **Streamlined 2-Stage Pipeline:**
    1.  **Stage 1:** Train Baseline model on labeled data.
    2.  **Stage 2:** Train Final model using Mean Teacher.
* **Flexible Configuration:** Easy parameter tuning via stage-specific JSON files (`config_stage1.json`, `config_stage2.json`).
* **GUI Application:** Interactive tool for visualization and vessel diameter measurement.

## Project Structure

```text
.
├── app.py                   # GUI Application entry point
├── config.json              # Main config (select type: Human/Rat)
├── config_stage1.json       # Config for Stage 1 (Baseline)
├── config_stage2.json       # Config for Stage 2 (Final - Mean Teacher)
├── config_stage1_stitchvit.json # Config for Stage 1 (Stitch-ViT)
├── config_stage2_stitchvit.json # Config for Stage 2 (Stitch-ViT)
├── data/
│   ├── annotated/           # Labeled images and JSON annotations
│   ├── masks/               # Binary masks (converted from annotations)
│   └── video/               # Raw unlabeled videos
├── models/                  # Saved model weights
├── logs/                    # Training logs
└── src/                     # Source code
```

## Setup

### 1. Environment Setup

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Data & Type

Open `config.json`, `config_stage1.json`, and `config_stage2.json`. Set the `"type"` field to either `"Human"` or `"Rat"` to match your dataset.

**Example:**

```json
{
    "type": "Human",
    ...
}
```

### 3. Data Preparation

Based on your selected `type` (e.g., `Human`):

*   **Labeled Data:** Place images and `.json` annotations in `data/annotated/Human/`.
*   **Unlabeled Data:** Place videos in `data/video/Human/`.

## Usage (Pipeline)

The system automatically detects directories based on the `type` in your config files.

### Running Individual Stages

**Stage 1: Train Baseline**
This is a prerequisite to generate initialization weights for the Mean Teacher model.

```bash
python -m src.main baseline
```

**Stage 2: Train Final (Mean Teacher)**
Uses weights from Stage 1 and begins the semi-supervised training process.

```bash
python -m src.main final
```

### Running the Full Pipeline (Recommended)

Runs Stage 1 -> Stage 2 sequentially and automatically visualizes results upon completion.

```bash
python -m src.main all --visualize
```

### Running the Stitch-ViT Model

To use the Stitch-ViT model, specify its configuration files using the `--config` flag.

**Stage 1 (Stitch-ViT):**
```bash
python -m src.main baseline --config config_stage1_stitchvit.json
```

**Stage 2 (Stitch-ViT):**
```bash
python -m src.main final --config config_stage2_stitchvit.json
```

**Full Pipeline (Stitch-ViT):**
```bash
python -m src.main all --config config_stage1_stitchvit.json --visualize
python -m src.main all --config config_stage2_stitchvit.json --visualize
```

### Additional Flags

*   `--config <path>`: Use a custom configuration file.
*   `--small-test`: Run on a small subset for debugging purposes.
*   `--visualize`: Generate prediction plots after training.
*   `--early-stop-patience <int>`: Override early stopping patience.

## Tools & Scripts

### 1. Convert JSON Annotations to Masks

Prepares training masks from annotation files (e.g., from LabelMe).

```bash
python -m tools.scripts.convert_json_to_mask --input data/annotated --output data/masks
```

### 2. Extract Frames from Videos

Prepares unlabeled data for semi-supervised learning.

```bash
python -m tools.scripts.extract_frames --video_dir data/video --output_dir data/frames --fps 1
```

### 3. Visualization & Evaluation

**Visualize Predictions:**

```bash
python -m tools.scripts.visualize_predictions
```

**Plot Training Curves:**

```bash
python -m tools.scripts.plot_training_curves
```

**Generate Evaluation Summary:**

```bash
python -m src.main visualize_eval
```

### 4. Compare Models

Compares the prediction results of two models.

```bash
python -m tools.scripts.compare_models --log-dir1 <path_to_model1_logs> --log-dir2 <path_to_model2_logs>
```

## GUI Application

Launch the user interface for interactive analysis:

```bash
python app.py
```

## Technical Details

*   **Model:** UNet++ (ResNet34 Backbone).
*   **Semi-Supervised Strategy:** Mean Teacher. The Teacher model weights are an Exponential Moving Average (EMA) of the Student weights. The Student learns from labeled data (supervised loss) and from the Teacher's consistency on unlabeled data (consistency loss).
