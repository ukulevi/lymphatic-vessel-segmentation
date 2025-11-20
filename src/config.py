# src/config.py
import os
from dataclasses import dataclass, asdict, field
from typing import Tuple, Optional, Dict, Any, List


@dataclass
class PathsConfig:
    labeled_dir: str = "data/annotated"
    labeled_masks_dir: str = "data/masks"
    unlabeled_dir: str = "data/video"
    pseudo_dir: str = "data/pseudo_labels"
    model_dir: str = "models"
    log_dir: str = "logs"
    checkpoint_dir: Optional[str] = None
    scripts_dir: Optional[str] = None


@dataclass
class DataConfig:
    image_size: Tuple[int, int] = (256, 256)
    batch_size: int = 8
    num_workers: int = 4


@dataclass
class AugmentationConfig:
    train_prob: float = 0.5


@dataclass
class ModelConfig:
    name: str = "unetpp"
    encoder: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    activation: Optional[str] = None
    classes: int = 1
    params: Optional[Dict[str, Any]] = None


@dataclass
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 50
    learning_rate: float = 1e-4
    early_stop_patience: int = 10
    k_folds: int = 5
    weight_decay: float = 1e-5
    device: str = "cuda"
    loss_type: str = "bce"
    alpha_schedule: str = "rebalance"
    initial_alpha: float = 0.005
    alpha_increment: float = 0.005
    aux_weights: Optional[List[float]] = None
    # Mean Teacher parameters
    ema_decay: float = 0.999
    consistency_weight: float = 10.0
    consistency_rampup: int = 10
    dice_weight: float = 0.5
    bce_weight: float = 0.5
    boundary_weight: float = 1.0


@dataclass
class ExperimentConfig:
    # Grouped sub-configs
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "default"
    type: str = "Human" # Added for visualization purposes

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        # Build sub-configs safely with defaults
        raw_paths = d.get("paths", {})
        # Filter unknown keys to avoid TypeError when config.json has extras
        allowed_path_keys = {
            "labeled_dir", "unlabeled_dir", "pseudo_dir",
            "model_dir", "log_dir", "checkpoint_dir", "scripts_dir", "labeled_masks_dir"
        }
        filtered_paths = {k: v for k, v in raw_paths.items() if k in allowed_path_keys}
        paths = PathsConfig(**filtered_paths)
        raw_data = d.get("data", {})
        allowed_data_keys = {"image_size", "batch_size", "num_workers"}
        filtered_data = {k: v for k, v in raw_data.items() if k in allowed_data_keys}
        data = DataConfig(**filtered_data)
        raw_aug = d.get("augmentation", {})
        allowed_aug_keys = {"train_prob"}
        filtered_aug = {k: v for k, v in raw_aug.items() if k in allowed_aug_keys}
        aug = AugmentationConfig(**filtered_aug)
        raw_model = d.get("model", {})
        allowed_model_keys = {"name", "encoder", "encoder_weights", "activation", "params", "classes"}
        filtered_model = {k: v for k, v in raw_model.items() if k in allowed_model_keys}
        model = ModelConfig(**filtered_model)

        raw_training = d.get("training", {})
        allowed_training_keys = {
            "batch_size", "epochs", "learning_rate", "early_stop_patience", 
            "k_folds", "weight_decay", "device", "ema_decay", 
            "consistency_weight", "consistency_rampup",
            "dice_weight", "bce_weight", "boundary_weight",
            "loss_type", "alpha_schedule", "initial_alpha", "alpha_increment", "aux_weights"
        }
        filtered_training = {k: v for k, v in raw_training.items() if k in allowed_training_keys}
        training = TrainingConfig(**filtered_training)
        name = d.get("experiment_name", "default")
        exp_type = d.get("type", "Human") # Get the new 'type' field

        # Update paths with type
        paths.labeled_dir = os.path.join(paths.labeled_dir, exp_type)
        paths.labeled_masks_dir = os.path.join(paths.labeled_masks_dir, exp_type)
        paths.unlabeled_dir = os.path.join(paths.unlabeled_dir, exp_type)
        paths.pseudo_dir = os.path.join(paths.pseudo_dir, exp_type)
        paths.model_dir = os.path.join(paths.model_dir, exp_type)
        paths.log_dir = os.path.join(paths.log_dir, exp_type)

        return ExperimentConfig(
            paths=paths,
            data=data,
            augmentation=aug,
            model=model,
            training=training,
            experiment_name=name,
            type=exp_type # Pass the new 'type' field
        )

    @staticmethod
    def from_json_file(path: str) -> "ExperimentConfig":
        import json
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return ExperimentConfig.from_dict(d)