"""
Model architectures and builders.
"""
import sys
import os
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from ..config import ModelConfig
from .cto.CTO_net import CTO
from .cto_stitchvit.CTO_net import CTO as CTO_StitchViT

def get_model(config: ModelConfig) -> nn.Module:
    """
    Build segmentation model from config
    """
    model_name = config.name.lower()
    
    if model_name == "unetpp":
        # Check for deep supervision flag in model params
        deep_supervision = False
        if config.params and 'deep_supervision' in config.params:
            deep_supervision = config.params['deep_supervision']

        # To keep training numerics consistent with BCEWithLogits, force logits output
        model = smp.UnetPlusPlus(
            encoder_name=config.encoder,
            encoder_weights=config.encoder_weights,
            in_channels=3,
            classes=config.classes,
            activation=None,
            deep_supervision=deep_supervision
        )
    elif model_name == "cto":
        model = CTO(seg_classes=config.classes)
    elif model_name == "cto_stitchvit":
        model = CTO_StitchViT(seg_classes=config.classes)
    else:
        raise ValueError(f"Unknown model: {config.name}")
    
    return model



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
    
    ckpt = torch.load(checkpoint_path, map_location=torch.device(device), weights_only=True)
    # Support both raw state_dict and dict with 'state_dict'
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
    # Handle DataParallel/DistributedDataParallel prefixes
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Filter out unnecessary keys from the loaded state_dict
    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}

    # Load the filtered state dict
    model.load_state_dict(filtered_state_dict, strict=False)

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