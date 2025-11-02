# Lymphatic Vessel Segmentation with BCP

Semi-supervised segmentation of lymphatic vessels using a student-teacher model with consistency training and boundary-aware loss.

## Project Structure

```
BCP_self_imple/
├── config.json              # Main configuration file
├── main.py                  # Shim script, directs to src.main
├── README.md
├── requirements.txt
├── data/
│   ├── annotated/           # Labeled images + JSON annotations
│   ├── masks/               # Generated binary masks
│   ├── video/               # Unlabeled frames/videos
│   └── pseudo_labels/       # Generated pseudo-labels
├── models/                  # Saved model checkpoints (baseline.pth, final.pth)
├── src/                     # Main source code
│   ├── __init__.py
│   ├── main.py              # Main pipeline entry point
│   ├── config.py            # Configuration dataclasses
│   ├── pseudo_labeler.py    # Logic for generating pseudo-labels
│   ├── train.py             # Legacy or alternative training script
│   ├── visualization.py     # Script for generating evaluation tables
│   ├── data/                # Data loading utilities (datasets.py)
│   ├── models/              # Model definitions (model_factory.py)
│   ├── training/            # Core training logic (trainer.py)
│   └── utils/               # Utility scripts (augment.py, logging.py, etc.)
├── tools/
│   ├── smoke_test.py        # Script to test environment and imports
│   └── scripts/
│       └── convert_json_to_mask.py # Converts LabelMe JSON to masks
└── logs/                    # Directory for training logs
```

## Setup & Installation

1.  Create a Python environment and install dependencies:
    ```bash
    python -m venv venv
    source venv/bin/activate  # or venv\Scripts\activate on Windows
    pip install -r requirements.txt
    ```

2.  Prepare data:
    -   Place your labeled images and JSON annotations in `data/annotated/`.
    -   Place your unlabeled video file in `data/video/` (e.g., `data/video/your_video.mp4`).
    -   **Extract frames from video** (recommended: 500 frames):
        ```bash
        python extract_video_frames.py --num_frames 500 --video data/video/your_video.mp4 --output data/video_frames
        ```
        This will extract frames evenly distributed from the video and save them to `data/video_frames/`.
    -   Convert the JSON annotations to binary masks:
        ```bash
        python tools/scripts/convert_json_to_mask.py --input data/annotated --output data/masks
        ```

## Usage

The main training pipeline is handled by `src/main.py` and consists of three stages that can be run sequentially or all at once.

1.  **Train Baseline Model:** Train a model on the labeled data (supervised learning).
    ```bash
    python -m src.main baseline --config config.json
    ```

2.  **Generate Pseudo-Labels:** Use the baseline model to create labels for the unlabeled data (self-supervised learning - pseudo labeling).
    ```bash
    python -m src.main pseudo --config config.json
    ```
    **Note:** Stage 2 is **NOT REQUIRED** if using Mean Teacher (see Stage 3 below).

3.  **Train Final Model:** Train a new model on the combination of labeled and pseudo-labeled data (semi-supervised learning - consistency training).
    
    **Option A: Without Mean Teacher** (uses pseudo-labels from Stage 2)
    ```bash
    python -m src.main final --config config.json
    ```
    - Requires Stage 2 to generate pseudo-labels first
    - Stage 3 will automatically run Stage 2 if pseudo-labels are missing
    
    **Option B: With Mean Teacher** (recommended, uses unlabeled data directly)
    ```bash
    python -m src.main final --use_mean_teacher --config config.json
    ```
    - **Stage 2 is NOT NEEDED** - Mean Teacher uses unlabeled data directly
    - More efficient: pseudo-labels are updated dynamically during training
    - Stage 3 will automatically run Stage 1 if baseline.pth is missing

To run the entire pipeline from start to finish:
```bash
python -m src.main all --config config.json
```

Or with Mean Teacher (skips Stage 2):
```bash
python -m src.main all --use_mean_teacher --config config.json
```

For a quick test on a small sample dataset, you can add the `--small-test` flag to any of the commands above.

## Visualization and Comparison

### Basic Visualization

To visualize the model's predictions on a few sample images, you can use the `--visualize` flag when running the training command.

```bash
python -m src.main all --config config.json --visualize
```

This will save a PNG file (e.g., `baseline_predictions.png` or `final_predictions.png`) in the `models` directory.

### Compare Two Models (Mean Teacher vs Without Mean Teacher)

After training both models (with and without Mean Teacher), you can compare them using the visualization script:

```bash
# Compare on labeled data
python visualize_compare_models.py --dataset labeled --num_samples 6 --device mps --output models/comparison_labeled.png

# Compare on unlabeled video frames (500 frames)
python visualize_compare_models.py --dataset video_frames --num_samples 6 --device mps --output models/comparison_video_frames.png
```

### Large-Scale Comparison with Red Overlay (500 Frames)

For a comprehensive comparison with **red overlay visualization** on many frames (recommended for easy visual inspection):

```bash
python compare_and_visualize_500frames.py --num_samples 50 --output models/comparison_500frames_overlay_red.png
```

**Options:**
- `--num_samples`: Number of frames to visualize (default: 50, can increase up to 500)
- `--model1`: Path to first model (default: `models/final_no_mt.pth`)
- `--model2`: Path to second model (default: `models/final_mt.pth`)
- `--frames_dir`: Directory containing frames (default: `data/video_frames`)
- `--config`: Path to config file (default: `config.json`)
- `--output`: Output path for visualization PNG

**Output:** A single PNG file showing:
- **Column 1:** Original input frames
- **Column 2:** Model 1 (No Mean Teacher) predictions with **bright red overlay** and border
- **Column 3:** Model 2 (Mean Teacher) predictions with **bright red overlay** and border
- **Column 4:** Side-by-side comparison for direct visual comparison

**Features:**
- **Red overlay** (alpha=0.6) with **red border** for clear visibility
- Easy to distinguish predictions even on dark or similar-colored backgrounds
- Shows predictions from both models side-by-side for direct comparison

This visualization helps identify which model performs better on different frames and provides clear visual feedback on segmentation quality.

## GUI Application

This project includes a GUI application for interactive video segmentation and diameter measurement.

To run the application, use the following command:

```bash
python app.py
```

Make sure you have installed the necessary dependencies, including `PyQt5`, by running `pip install -r requirements.txt`.

## Testing and Visualization

### Testing

To run a quick smoke test to ensure the environment is set up correctly and all necessary components are importable, run the following command:

```bash
python tools/smoke_test.py
```

This script will attempt to import key modules and configurations. A successful run will print information about the default configuration, transforms, and datasets without any errors.

## Metrics Table

To generate a text-based table of the evaluation metrics from a training run, you can use the `src/visualization.py` script.

1.  Make sure you have a `metrics.csv` file in one of the log directories (e.g., `logs/your_experiment/metrics.csv`).
2.  Open the `src/visualization.py` file and modify the `metrics_file` variable to point to your `metrics.csv` file.
3.  Run the script:

```bash
python src/visualization.py
```

This will print the evaluation table to the console and save it as `evaluation_table.txt` in the project root directory.

## Configuration

The main configuration for the project is in `config.json`. You can modify this file to change the training parameters, model, and data paths.

### CPU Training

To run the training on a CPU, you need to change the `device` parameter in `config.json` from `"cuda"` to `"cpu"`:

```json
"training": {
    "device": "cpu",
}
```

### Model Improvements

The training process has been improved to achieve better results:

*   **Increased Training Time:** The number of training epochs has been increased to 50 to allow the model to learn more complex features.
*   **Improved Early Stopping:** The early stopping patience has been increased to 10 to prevent the training from stopping prematurely.
*   **Learning Rate Scheduling:** A `ReduceLROnPlateau` learning rate scheduler has been added to the training loop. This scheduler monitors the validation loss and reduces the learning rate if it stops improving, which helps the model to find a better minimum.

## Model Details

-   **Architecture:** UNet++ with a ResNet34 encoder.
-   **Loss Functions:** A combination of Dice loss for region segmentation and a boundary-aware loss for edge definition.
-   **Semi-supervised Learning:** 
    - **Without Mean Teacher:** Uses pseudo-labels generated offline from Stage 2
    - **With Mean Teacher:** Uses student-teacher model with Exponential Moving Average (EMA) for consistency regularization on unlabeled data (pseudo-labels updated dynamically during training)

## Stage 2: When is it needed?

**Stage 2 is OPTIONAL** and depends on which mode you use for Stage 3:

-   **❌ NOT NEEDED** if using Mean Teacher (`--use_mean_teacher` flag)
    - Mean Teacher uses unlabeled data directly, no need to generate pseudo-labels beforehand
    
-   **✅ NEEDED** if NOT using Mean Teacher
    - Stage 3 requires pseudo-labels to train
    - Stage 3 will automatically run Stage 2 if pseudo-labels are missing

**Recommended approach:** Use Mean Teacher for simplicity and better performance:
```bash
python -m src.main final --use_mean_teacher
```

### Complete Workflow with 500 Frames

**Step-by-step workflow:**

1. **Extract 500 frames from video:**
   ```bash
   python extract_video_frames.py --num_frames 500
   ```

2. **Train baseline model:**
   ```bash
   python -m src.main baseline --config config.json
   ```

3. **Generate pseudo-labels (for comparison with non-Mean Teacher method):**
   ```bash
   python -m src.main pseudo --config config.json
   ```
   This generates pseudo-labels from 500 frames in `data/video_frames/`.

4. **Train final model WITHOUT Mean Teacher:**
   ```bash
   python -m src.main final --config config.json
   ```
   Output: `models/final_no_mt.pth`

5. **Train final model WITH Mean Teacher:**
   ```bash
   python -m src.main final --use_mean_teacher --config config.json
   ```
   Output: `models/final_mt.pth`

6. **Compare and visualize both models with red overlay:**
   ```bash
   python compare_and_visualize_500frames.py --num_samples 50 --output models/comparison_500frames_overlay_red.png
   ```
   This generates a visualization with bright red overlay for easy visual inspection of predictions from both models.

For more details, see `HUONG_DAN_CHAY_STAGES.md` and `SO_SANH_MEAN_TEACHER_VS_PSEUDO_LABELS.md`.

## References

-   UNet++: [Paper](https://arxiv.org/abs/1807.10165)
-   Mean Teacher: [Paper](https://arxiv.org/abs/1703.01780)
-   Boundary Loss: [Paper](https://arxiv.org/abs/1905.07852)
