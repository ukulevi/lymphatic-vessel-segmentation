"""
Model architectures and builders.
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from ..config import ModelConfig

def get_model(config: ModelConfig) -> nn.Module:
    """
    Build segmentation model from config
    """
    if config.name.lower() == "unetpp":
        # To keep training numerics consistent with BCEWithLogits, force logits output
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )
    else:
        raise ValueError(f"Unknown model: {config.name}")
    
    return model

def get_baseline_model() -> nn.Module:
    """Get the baseline model"""
    config = ModelConfig(
        name="unetpp",
        encoder="resnet34",
        encoder_weights="imagenet",
        activation="sigmoid",
        params=None
    )
    return get_model(config)

def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    device: Optional[str] = None
) -> nn.Module:
    """
    Load model weights from checkpoint
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    # Support both raw state_dict and dict with 'state_dict'
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    # Handle DataParallel/DistributedDataParallel prefixes
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model

def save_checkpoint(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save model checkpoint with optional metadata
    """
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    checkpoint = {'state_dict': state_dict, 'metadata': metadata or {}}
    torch.save(checkpoint, save_path)