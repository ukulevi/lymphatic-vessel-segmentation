import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

from .models import get_baseline_model
from .data.datasets import LabeledVesselDataset, FinalVesselDataset
from .training.trainer import train_kfold
from .utils.augment import create_train_transform, create_val_transform
from .config import TrainingConfig

# --- Configuration (defaults match the repo structure) ---
# Device selection with MPS support for macOS
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
BATCH_SIZE = 8
NUM_EPOCHS_BASELINE = 50
NUM_EPOCHS_FINAL = 50
LEARNING_RATE = 1e-4

# paths (defaults match the repo structure)
IMAGE_DIR_LABELED = "data/annotated"          # expects subfolders per video (each contains .png and .json)
JSON_DIR = None                               # dataset finds .json next to image if None
BASELINE_MODEL_PATH = "models/baseline_unetpp.pth"

ALL_FRAMES_DIR = "data/video"                  # frames or videos (FinalVesselDataset expects frames)
LABELED_MASK_DIR = "data/labeled_masks"        # where you store mask PNGs converted from JSON
PSEUDO_MASK_DIR = "data/pseudo_masks"          # generated pseudo masks
FINAL_MODEL_PATH = "models/final_unetpp.pth"

def train_baseline(k_folds: int = 5, early_stopping_patience: int = 5):
    """Phase 1: Train the baseline model with k-fold + early stopping."""
    print("--- Phase 1: Training Baseline Model (k-fold=%d) ---" % k_folds)
    
    train_transform = create_train_transform(p=0.5)
    val_transform = create_val_transform()

    print(f"Loading labeled data from {IMAGE_DIR_LABELED} and {JSON_DIR}")
    train_dataset = LabeledVesselDataset(IMAGE_DIR_LABELED, JSON_DIR, transform=train_transform)
    val_dataset = LabeledVesselDataset(IMAGE_DIR_LABELED, JSON_DIR, transform=val_transform)

    if len(train_dataset) == 0:
        print("Error: No data found in the directory. Please check the paths.")
        return

    # Use a model factory to instantiate a fresh model per fold
    model_factory = lambda: get_baseline_model()

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS_BASELINE,
        learning_rate=LEARNING_RATE,
        early_stop_patience=early_stopping_patience,
        k_folds=k_folds,
        device=DEVICE,
        loss_type='combined'  # Use the combined loss
    )

    best_model = train_kfold(
        model_factory=model_factory,
        dataset=train_dataset,
        config=config
    )

    if best_model is not None:
        torch.save(best_model.state_dict(), BASELINE_MODEL_PATH)
        print(f"Saved baseline model to {BASELINE_MODEL_PATH}")
    return best_model

def train_final_model(k_folds: int = 5, early_stopping_patience: int = 5):
    """Phase 3: Train the final model with Copy-Paste + k-fold/early stopping."""
    print("\n--- Phase 3: Training Final Model (k-fold=%d) ---" % k_folds)
    if not os.path.exists(PSEUDO_MASK_DIR) or not os.listdir(PSEUDO_MASK_DIR):
        print("Warning: Pseudo-label directory is empty. Have you run Phase 2?")
    
    train_transform = create_train_transform(p=0.5)
    val_transform = create_val_transform()

    print("Loading combined data (real labels + pseudo labels)...")
    final_dataset = FinalVesselDataset(
        image_dir=ALL_FRAMES_DIR,
        labeled_mask_dir=LABELED_MASK_DIR,
        pseudo_mask_dir=PSEUDO_MASK_DIR,
        transform=train_transform,
        copy_paste_prob=0.5
    )
    if len(final_dataset) == 0:
        print("Error: No data found for training. Please check the paths.")
        return

    model_factory = lambda: get_baseline_model()

    config = TrainingConfig(
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS_FINAL,
        learning_rate=LEARNING_RATE,
        early_stop_patience=early_stopping_patience,
        k_folds=k_folds,
        device=DEVICE,
        loss_type='combined'  # Use the combined loss
    )

    best_model = train_kfold(
        model_factory=model_factory,
        dataset=final_dataset,
        config=config
    )

    if best_model is not None:
        torch.save(best_model.state_dict(), FINAL_MODEL_PATH)
        print(f"Saved final model to {FINAL_MODEL_PATH}")
    return best_model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Train segmentation models.")
    parser.add_argument('--baseline', action='store_true', help='Train the baseline model.')
    parser.add_argument('--final', action='store_true', help='Train the final model.')
    args = parser.parse_args()

    if args.baseline:
        train_baseline()
    if args.final:
        train_final_model()

    if not args.baseline and not args.final:
        print("Please specify which model to train: --baseline or --final")
