"""Main entry point for vessel segmentation pipeline.

Complete training pipeline with 2 stages:
1. Train baseline model on labeled data (with k-fold CV)
2. Train final model on combined labeled and unlabeled data using Mean Teacher.

Example usage:
    # Train baseline:
    python -m src.main baseline
    
    # Train final model:
    python -m src.main final
    
    # Or run complete pipeline:
    python -m src.main all
"""
import os
import json
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import ExperimentConfig
from src.models.model_factory import get_model, save_checkpoint, load_checkpoint
from src.data.datasets import LabeledVesselDataset, PseudoLabeledDataset, VideoDataset
from src.training.trainer import Trainer, train_kfold
from src.training.mean_teacher import MeanTeacherTrainer, create_mean_teacher_models
from src.utils.logging import TrainingLogger
from src.utils.augment import (
    create_train_transform,
    create_val_transform,
    create_strong_transform,
    create_weak_transform
)

def load_config(config_path: str) -> ExperimentConfig:
    """Load and validate configuration"""
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        return ExperimentConfig()
        
    with open(config_path) as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)

def train_baseline(config: ExperimentConfig, logger: TrainingLogger):
    """Stage 1: Train baseline model with k-fold CV"""
    print("\n=== Stage 1: Training Baseline Model ===")
    
    # Setup transforms
    train_transform = create_train_transform(p=config.augmentation.train_prob)
    val_transform = create_val_transform()
    
    # Create dataset
    dataset = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=train_transform,
        target_size=config.data.image_size
    )
    
    if len(dataset) == 0:
        raise RuntimeError(
            f"No data found in {config.paths.labeled_dir}"
        )
        
    # Train with k-fold CV
    model = train_kfold(
        model_factory=lambda: get_model(config.model),
        dataset=dataset,
        model_config=config.model,
        config=config.training,
        logger=logger
    )
    
    # Save final model
    save_path = os.path.join(config.paths.model_dir, "baseline.pth")
    save_checkpoint(model, save_path)
    logger.log_model(save_path)
    print(f"Saved baseline model to {save_path}")
    
    return model

def train_final(config: ExperimentConfig, logger: TrainingLogger):
    """
    Stage 2: Train final model with Mean Teacher.
    
    Args:
        config: Experiment configuration
        logger: Training logger
    """
    print("\n=== Stage 2: Training Final Model ===")
    print("Using Mean Teacher for semi-supervised learning")

    # Transforms
    train_transform = create_train_transform(p=config.augmentation.train_prob)
    val_transform = create_val_transform()
    weak_transform = create_weak_transform(p=0.3)  # Cho student
    strong_transform = create_strong_transform(p=0.5)  # Cho teacher/unlabeled

    # Datasets: labeled data
    labeled_ds = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=train_transform,
        target_size=config.data.image_size,
    )

    if len(labeled_ds) == 0:
        raise RuntimeError(f"No labeled data found in {config.paths.labeled_dir}")

    # Split labeled data thành train/val
    from torch.utils.data import DataLoader, random_split
    total_labeled = len(labeled_ds)
    val_len = max(1, int(0.1 * total_labeled))
    train_labeled_len = total_labeled - val_len
    train_labeled_set, val_set = random_split(
        labeled_ds, [train_labeled_len, val_len]
    )

    # Unlabeled dataset (cho Mean Teacher)
    # Note: Trong Mean Teacher, cả student và teacher đều nhận cùng unlabeled image
    # nhưng với augmentations khác nhau. Tuy nhiên, trong implementation này,
    # chúng ta sẽ dùng weak_transform cho dataset và apply strong augmentation
    # trong training loop nếu cần (hoặc dùng strong_transform cho teacher)
    unlabeled_ds = None
    # Thử tìm unlabeled data từ nhiều nguồn
    unlabeled_dirs = [
        config.paths.unlabeled_dir,  # data/frames/<type>
        config.paths.labeled_dir,  # Fallback: dùng labeled data (nếu không có unlabeled)
    ]
    
    for unlabeled_dir in unlabeled_dirs:
        if os.path.exists(unlabeled_dir) and os.path.isdir(unlabeled_dir):
            # Dùng weak_transform cho unlabeled dataset (student sẽ dùng)
            unlabeled_ds = VideoDataset(
                image_root=unlabeled_dir,
                transform=weak_transform,  # Weak augmentation cho student
                target_size=config.data.image_size,
            )
            if len(unlabeled_ds) > 0:
                print(f"✓ Found {len(unlabeled_ds)} unlabeled images in {unlabeled_dir} for Mean Teacher")
                break
    
    if unlabeled_ds is None or len(unlabeled_ds) == 0:
        print("Warning: No unlabeled data found, Mean Teacher will only use labeled data.")
        unlabeled_ds = None

    # DataLoaders
    labeled_loader = DataLoader(
        train_labeled_set,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    unlabeled_loader = None
    if unlabeled_ds is not None:
        unlabeled_loader = DataLoader(
            unlabeled_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
    
    val_loader = DataLoader(
        val_set,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Model và trainer
    base_model = get_model(config.model)
    
    # Load baseline model làm điểm khởi đầu (QUAN TRỌNG!)
    baseline_path = os.path.join(config.paths.model_dir, "baseline.pth")
    if not os.path.exists(baseline_path):
        raise RuntimeError(
            f"Baseline model not found at {baseline_path}.\n"
            "Mean Teacher requires a pre-trained baseline model. Run Stage 1 first."
        )
    print(f"Loading baseline model from {baseline_path}...")
    base_model = load_checkpoint(base_model, baseline_path, device=config.training.device)
    print("✓ Loaded baseline model weights")
    
    # Tạo student và teacher models từ baseline
    student_model, teacher_model = create_mean_teacher_models(base_model)
    
    # Optimizer chỉ cho student
    optimizer = torch.optim.Adam(
        student_model.parameters(),
        lr=config.training.learning_rate
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Mean Teacher Trainer
    trainer = MeanTeacherTrainer(
        student_model=student_model,
        teacher_model=teacher_model,
        optimizer=optimizer,
        model_config=config.model,
        config=config.training,
        logger=logger,
        device=config.training.device
    )
    
    # Training với Mean Teacher
    model = trainer.train(labeled_loader, unlabeled_loader, val_loader, scheduler)
    
    # Model là teacher (tốt hơn student)
    print("✓ Mean Teacher training completed. Using teacher model (best).")

    # Save final model
    os.makedirs(config.paths.model_dir, exist_ok=True)
    save_path = os.path.join(config.paths.model_dir, "final.pth")
    save_checkpoint(model, save_path)
    logger.log_model(save_path)
    print(f"Saved final model to {save_path}")

    return model

import platform
import psutil
def log_system_info(logger, device):
    """Log system information based on the selected device."""
    logger.log_message("--- System Information ---")
    logger.log_message(f"Operating System: {platform.system()} {platform.release()}")
    logger.log_message(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

    if device == "cuda" and torch.cuda.is_available():
        logger.log_message(f"Device: GPU ({torch.cuda.get_device_name(0)})")
    elif device == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.log_message("Device: Apple MPS")
    else:
        logger.log_message(f"Device: CPU ({platform.processor()})")
        logger.log_message(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    
    logger.log_message("--------------------------")

from src.visualization import visualize_predictions, visualize_evaluation_table

def run_visualize_eval(config: ExperimentConfig):
    """Find the latest log and generate an evaluation table."""
    print(f"\n=== Generating Evaluation Table for type '{config.type}' ===")
    log_dir = config.paths.log_dir
    if not os.path.isdir(log_dir):
        print(f"Log directory not found: {log_dir}")
        return

    # Find the most recent training log directory
    try:
        latest_log_dir = max(
            [os.path.join(log_dir, d) for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))],
            key=os.path.getmtime
        )
    except ValueError:
        print(f"No training logs found in {log_dir}")
        return

    metrics_file = os.path.join(latest_log_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"metrics.csv not found in the latest log directory: {latest_log_dir}")
        return

    print(f"Using metrics from: {metrics_file}")

    # Define output path inside the typed model directory
    output_path = os.path.join(config.paths.model_dir, f"evaluation_summary_{os.path.basename(latest_log_dir)}.txt")

    # Generate the table
    visualize_evaluation_table(metrics_file, output_path)
    print(f"Successfully generated evaluation table at: {output_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["baseline", "final", "all", "visualize_eval"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None, # Change default to None
        help="Path to a custom config file. If not provided, stage-specific defaults (config_stage1.json or config_stage3.json) will be used."
    )
    parser.add_argument(
        "--small-test",
        action="store_true",
        help="Run on small test dataset"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualizations after training"
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=None,
        help="Override early stopping patience from config"
    )
    args = parser.parse_args()
    
    # Load config based on stage
    if args.config: # If a custom config is provided, use it for all stages
        config = load_config(args.config)
    elif args.stage == "baseline":
        config = load_config("config_stage1.json")
    elif args.stage in ["final", "visualize_eval"]:
        config = load_config("config_stage2.json")
    elif args.stage == "all":
        # For "all" stage, load config_stage1 initially for the baseline.
        config = load_config("config_stage1.json")
    else:
        # This case should ideally not be reached due to choices in parser
        raise ValueError(f"Unknown stage: {args.stage}")

    # Auto-detect and set device
    device = config.training.device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    config.training.device = device

    if args.early_stop_patience is not None:
        config.training.early_stop_patience = args.early_stop_patience

    
    # Override paths for small test
    if args.small_test:
        # Prefer annotated/test_sample if it exists, otherwise use data/test_sample flat folder
        candidate_annot = os.path.join(config.paths.labeled_dir, "test_sample")
        candidate_flat = os.path.join("data", "test_sample")
        labeled_test_dir = candidate_annot if os.path.isdir(candidate_annot) else (
            candidate_flat if os.path.isdir(candidate_flat) else config.paths.labeled_dir
        )
        # Prefer a flat data/test_sample folder for unlabeled frames if present
        unlabeled_test_dir = (
            os.path.join(config.paths.unlabeled_dir, "test_sample")
            if os.path.isdir(os.path.join(config.paths.unlabeled_dir, "test_sample"))
            else (os.path.join("data", "test_sample") if os.path.isdir(os.path.join("data", "test_sample")) else config.paths.unlabeled_dir)
        )
        config.paths.labeled_dir = labeled_test_dir
        config.paths.unlabeled_dir = unlabeled_test_dir
        print(f"Running with test data from {config.paths.labeled_dir}")
        
    # Create experiment logger only for stages that need it
    logger = None
    if args.stage in ["baseline", "final", "all"]:
        logger = TrainingLogger(config.paths.log_dir)
        print("\n" + "="*60)
        print("SYSTEM INFORMATION")
        print("="*60)
        log_system_info(logger, device)
        print("="*60 + "\n")
        logger.log_config(config.to_dict())
    
    # Run pipeline stages
    try:
        if args.stage in ["baseline", "all"]:
            model = train_baseline(config, logger)
            if args.visualize and model is not None:
                ds = LabeledVesselDataset(
                    image_root=config.paths.labeled_dir,
                    mask_dir=config.paths.labeled_masks_dir,
                    transform=create_val_transform(),
                    target_size=config.data.image_size
                )
                fig = visualize_predictions(
                    model,
                    ds,
                    num_samples=4,
                    device=config.training.device
                )
                fig_path = os.path.join(
                    config.paths.model_dir,
                    "baseline_predictions.png"
                )
                fig.savefig(fig_path)
                print(f"Saved visualizations to {fig_path}")
            
        if args.stage in ["final", "all"]:
            if args.stage == "all":
                # If running "all" stage, reload config for the final stage
                config_path_for_stage2 = args.config if args.config else "config_stage2.json"
                config = load_config(config_path_for_stage2)
                # Re-apply early stop patience if overridden
                if args.early_stop_patience is not None:
                    config.training.early_stop_patience = args.early_stop_patience
            
            model = train_final(config, logger)
            if args.visualize and model is not None:
                ds = LabeledVesselDataset(
                    image_root=config.paths.labeled_dir,
                    mask_dir=config.paths.labeled_masks_dir,
                    transform=create_val_transform(),
                    target_size=config.data.image_size
                )
                fig = visualize_predictions(
                    model,
                    ds,
                    num_samples=4,
                    device=config.training.device
                )
                fig_path = os.path.join(
                    config.paths.model_dir,
                    "final_predictions.png"
                )
                fig.savefig(fig_path)
                print(f"Saved visualizations to {fig_path}")

        if args.stage == "visualize_eval":
            run_visualize_eval(config)
            
    except Exception as e:
        print(f"An error occurred during the pipeline execution: {str(e)}")
        if logger:
            logger.log_message(f"ERROR: {str(e)}")
        raise
if __name__ == "__main__":
    main()