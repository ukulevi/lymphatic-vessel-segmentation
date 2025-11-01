"""Main entry point for vessel segmentation pipeline.

Complete training pipeline with 3 stages:
1. Train baseline model on labeled data (with k-fold CV)
2. Generate pseudo-labels using baseline model
3. Train final model on combined data

Example usage:
    # Train baseline:
    python -m src.main baseline --config config.json
    
    # Generate pseudo-labels:
    python -m src.main pseudo
    
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
from src.visualization import visualize_predictions
from src.pseudo_labeler import generate_pseudo_labels as generate_pseudo_labels_impl

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
        config=config.training,
        logger=logger
    )
    
    # Save final model
    save_path = os.path.join(config.paths.model_dir, "baseline.pth")
    save_checkpoint(model, save_path)
    logger.log_model(save_path)
    print(f"Saved baseline model to {save_path}")
    
    return model

def generate_pseudo_labels(config: ExperimentConfig, logger: TrainingLogger):
    """Stage 2: Generate pseudo-labels using baseline model"""
    print("\n=== Stage 2: Generating Pseudo-Labels ===")

    # Resolve checkpoint
    preferred_path = os.path.join(config.paths.model_dir, "baseline.pth")
    fallback_path = os.path.join(
        config.paths.checkpoint_dir or config.paths.model_dir, 
        "baseline_unetpp.pth"
    )
    ckpt_path = preferred_path if os.path.exists(preferred_path) else fallback_path
    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Baseline checkpoint not found. Tried: {preferred_path} and {fallback_path}"
        )

    # Load baseline model
    model = get_model(config.model)
    model = load_checkpoint(model, ckpt_path, device=config.training.device)

    # Generate pseudo-labels
    os.makedirs(config.paths.pseudo_dir, exist_ok=True)
    saved = generate_pseudo_labels_impl(
        model=model,
        unlabeled_dir=config.paths.unlabeled_dir,
        output_dir=config.paths.pseudo_dir,
        device=config.training.device,
    )
    # Fallback: if no frames in unlabeled_dir, try labeled_dir to allow quick smoke-tests
    if saved == 0 and os.path.isdir(config.paths.labeled_dir):
        print("No unlabeled frames found; falling back to labeled_dir for pseudo-labels.")
        saved = generate_pseudo_labels_impl(
            model=model,
            unlabeled_dir=config.paths.labeled_dir,
            output_dir=config.paths.pseudo_dir,
            device=config.training.device,
        )
    logger.log_message(
        f"Generated {saved} pseudo-label(s) to {config.paths.pseudo_dir} using {ckpt_path}"
    )
    
def train_final(config: ExperimentConfig, logger: TrainingLogger, use_mean_teacher: bool = False):
    """
    Stage 3: Train final model với Mean Teacher
    
    Tự động chạy Stage 1 (baseline) nếu chưa có baseline.pth
    Tự động chạy Stage 2 (pseudo-labels) nếu không dùng Mean Teacher và chưa có pseudo-labels
    
    Args:
        config: Experiment configuration
        logger: Training logger
        use_mean_teacher: Nếu True, sử dụng Mean Teacher; nếu False, dùng Trainer thông thường
    """
    print("\n=== Stage 3: Training Final Model ===")
    if use_mean_teacher:
        print("Using Mean Teacher for semi-supervised learning")

    # ===== TỰ ĐỘNG CHẠY STAGE 1 NẾU CHƯA CÓ BASELINE MODEL =====
    baseline_path = os.path.join(config.paths.model_dir, "baseline.pth")
    if not os.path.exists(baseline_path):
        print("\n⚠️  Baseline model not found. Running Stage 1 first...")
        train_baseline(config, logger)
        print("✓ Stage 1 completed. Continuing with Stage 3...\n")
    
    # ===== TỰ ĐỘNG CHẠY STAGE 2 NẾU KHÔNG DÙNG MEAN TEACHER VÀ CHƯA CÓ PSEUDO-LABELS =====
    if not use_mean_teacher:
        if not os.path.exists(config.paths.pseudo_dir) or not os.listdir(config.paths.pseudo_dir):
            print("\n⚠️  Pseudo-labels not found. Running Stage 2 first...")
            generate_pseudo_labels(config, logger)
            print("✓ Stage 2 completed. Continuing with Stage 3...\n")

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
    if use_mean_teacher:
        # Thử tìm unlabeled data từ nhiều nguồn
        unlabeled_dirs = [
            config.paths.unlabeled_dir,  # data/video/
            "data/video_frames",  # Frames đã extract
            config.paths.labeled_dir,  # Fallback: dùng labeled data (nếu không có unlabeled)
        ]
        
        for unlabeled_dir in unlabeled_dirs:
            try:
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
            except Exception as e:
                continue
        
        if unlabeled_ds is None or len(unlabeled_ds) == 0:
            print("Warning: No unlabeled data found, Mean Teacher sẽ chỉ dùng labeled data")
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

    # Model và trainer - Load từ baseline.pth nếu có
    base_model = get_model(config.model)
    
    # Load weights từ baseline model để khởi tạo tốt hơn
    if os.path.exists(baseline_path):
        print(f"✓ Loading baseline model weights from {baseline_path}")
        base_model = load_checkpoint(base_model, baseline_path, device=config.training.device)
    else:
        print("⚠️  Training from scratch (no baseline model found)")
    
    if use_mean_teacher:
        # Tạo student và teacher models
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
            config=config.training,
            logger=logger,
            device=config.training.device
        )
        
        # Training với Mean Teacher
        model = trainer.train(labeled_loader, unlabeled_loader, val_loader, scheduler)
        
        # Model là teacher (tốt hơn student)
        print("✓ Mean Teacher training completed. Using teacher model (best).")
    
    # Dùng Trainer thông thường với pseudo-labels
    if not use_mean_teacher:
        if not os.path.exists(config.paths.pseudo_dir) or not os.listdir(config.paths.pseudo_dir):
            raise RuntimeError(
                f"Pseudo-labels not found in {config.paths.pseudo_dir}.\n"
                "Run the pseudo-label generation stage first or use Mean Teacher."
            )
        
        pseudo_ds_list = []
        # Pseudo-labeled data KHÔNG dùng augmentation (chỉ resize + normalize)
        # Vì pseudo-labels đã được generate từ model, augmentation có thể làm sai lệch
        try:
            ds_unl = PseudoLabeledDataset(
                image_root=config.paths.unlabeled_dir,
                mask_dir=config.paths.pseudo_dir,
                transform=val_transform,  # KHÔNG augmentation cho pseudo-labels
                target_size=config.data.image_size,
            )
            if len(ds_unl) > 0:
                pseudo_ds_list.append(ds_unl)
        except Exception:
            pass
        
        try:
            ds_lab = PseudoLabeledDataset(
                image_root=config.paths.labeled_dir,
                mask_dir=config.paths.pseudo_dir,
                transform=val_transform,  # KHÔNG augmentation cho pseudo-labels
                target_size=config.data.image_size,
            )
            if len(ds_lab) > 0:
                pseudo_ds_list.append(ds_lab)
        except Exception:
            pass
        
        if not pseudo_ds_list:
            raise RuntimeError(
                f"No pseudo-labeled pairs found using masks in {config.paths.pseudo_dir}"
            )
        
        from torch.utils.data import ConcatDataset
        combined = ConcatDataset([train_labeled_set] + pseudo_ds_list)
        
        train_loader = DataLoader(
            combined,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        
        trainer = Trainer(base_model, config.training, logger)
        model = trainer.train(train_loader, val_loader)

    # Save final model
    os.makedirs(config.paths.model_dir, exist_ok=True)
    save_path = os.path.join(config.paths.model_dir, "final.pth")
    save_checkpoint(model, save_path)
    logger.log_model(save_path)
    print(f"Saved final model to {save_path}")

    return model

import platform
import psutil

def log_system_info(logger):
    """Log system information."""
    logger.log_message("--- System Information ---")
    logger.log_message(f"Operating System: {platform.system()} {platform.release()}")
    logger.log_message(f"Processor: {platform.processor()}")
    logger.log_message(f"CPU Cores: {psutil.cpu_count(logical=True)}")
    logger.log_message(f"Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.log_message("--------------------------")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["baseline", "pseudo", "final", "all"],
        help="Pipeline stage to run"
    )
    parser.add_argument(
        "--use_mean_teacher",
        action="store_true",
        help="Use Mean Teacher for Stage 3 (semi-supervised learning)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to config file"
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
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
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
        
    # Create experiment logger
    logger = TrainingLogger(config.paths.log_dir)
    log_system_info(logger)
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
            
        if args.stage in ["pseudo", "all"]:
            generate_pseudo_labels(config, logger)
            
        if args.stage in ["final", "all"]:
            # Mean Teacher: mặc định False, có thể enable bằng --use_mean_teacher
            use_mean_teacher = args.use_mean_teacher if hasattr(args, 'use_mean_teacher') else False
            model = train_final(config, logger, use_mean_teacher=use_mean_teacher)
            if args.visualize and model is not None:
                ds = LabeledVesselDataset(
                    image_root=config.paths.labeled_dir,
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
            
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()