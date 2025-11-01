"""
Dataset classes and data loading utilities
"""
from typing import Optional, Tuple, Dict, Any, List, Union
import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, ConcatDataset
from albumentations.core.composition import Compose
from ..config import DataConfig, PathsConfig
from ..utils import logging

logger = logging.get_logger(__name__)


class BaseDataset(Dataset):
    """Base class for vessel datasets"""
    def __init__(
        self,
        image_root: str,
        transform: Optional[Compose] = None,
        target_size: Optional[Tuple[int, int]] = None
    ):
        self.image_root = image_root
        self.transform = transform
        self.target_size = target_size
        self.samples = self._collect_samples()
        
    def _collect_samples(self) -> list:
        """Collect all valid image samples"""
        samples = []
        for root, _, files in os.walk(self.image_root):
            for f in sorted(files):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    img_path = os.path.join(root, f)
                    samples.append(img_path)
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load and preprocess image"""
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.target_size is not None:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_LINEAR)
        return img
    
    def _prepare_sample(
        self, 
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Dict[str, torch.Tensor]:
        """Apply transforms and convert to tensor"""
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']
                
        # Ensure tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
            
        sample = {'image': image}
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            # Ensure channel dimension first
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            elif mask.ndim == 3 and mask.shape[0] != 1 and mask.shape[-1] == 1:
                # If mask is HxWx1 move to 1xHxW
                mask = mask.permute(2, 0, 1)
            sample['mask'] = mask
            
        return sample
        
def create_dataset(
    paths_config: PathsConfig,
    data_config: DataConfig,
    transform: Optional[Compose] = None,
    include_unlabeled: bool = False,
    include_pseudo: bool = False
) -> Dataset:
    """Factory function to create appropriate dataset(s)
    
    Args:
        data_config: Dataset configuration
        transform: Optional albumentations transform
        include_unlabeled: Whether to include unlabeled data
        include_pseudo: Whether to include pseudo-labeled data
        
    Returns:
        Dataset or ConcatDataset if multiple sources
    """
    datasets = []
    
    # Add labeled data
    if paths_config.labeled_dir:
        labeled_ds = LabeledVesselDataset(
            image_root=paths_config.labeled_dir,
            mask_dir=paths_config.labeled_masks_dir,
            transform=transform,
            target_size=data_config.image_size
        )
        datasets.append(labeled_ds)
        
    # Add unlabeled data
    if include_unlabeled and paths_config.unlabeled_dir:
        unlabeled_ds = VideoDataset(
            image_root=paths_config.unlabeled_dir,
            transform=transform,
            target_size=data_config.image_size
        )
        datasets.append(unlabeled_ds)
        
    # Add pseudo-labeled data
    if include_pseudo and paths_config.pseudo_dir:
        pseudo_ds = PseudoLabeledDataset(
            image_root=paths_config.unlabeled_dir,
            mask_dir=paths_config.pseudo_dir,
            transform=transform,
            target_size=data_config.image_size
        )
        datasets.append(pseudo_ds)
        
    if not datasets:
        raise RuntimeError("No valid data sources found in config")
        
    return (
        ConcatDataset(datasets) if len(datasets) > 1 
        else datasets[0]
    )

class LabeledVesselDataset(BaseDataset):
    """Dataset for labeled vessel images with masks"""
    def __init__(
        self,
        image_root: str,
        mask_dir: str,
        transform: Optional[Compose] = None,
        target_size: Optional[Tuple[int, int]] = None,
        mask_suffix: str = '_mask.png'
    ):
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        super().__init__(image_root, transform, target_size)
        
    def _collect_samples(self) -> list:
        """Collect image-mask pairs"""
        samples = []
        for root, _, files in os.walk(self.image_root):
            for f in sorted(files):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    img_path = os.path.join(root, f)
                    
                    # Construct mask path
                    rel_path = os.path.relpath(img_path, self.image_root)
                    mask_path = os.path.join(
                        self.mask_dir,
                        os.path.splitext(rel_path)[0] + self.mask_suffix
                    )
                    
                    if os.path.isfile(mask_path):
                        samples.append((img_path, mask_path))
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        mask = mask.astype(np.float32) / 255.0
        
        if self.target_size is not None:
            mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST
            )
            
        # Prepare sample
        sample = self._prepare_sample(image, mask)
        return sample

class VideoDataset(BaseDataset):
    """Dataset for loading video frames or images without annotations"""
    def __init__(
        self,
        image_root: str,
        transform: Optional[Compose] = None,
        target_size: Optional[Tuple[int, int]] = None,
        exts: Tuple[str, ...] = ('.png', '.jpg', '.jpeg')
    ):
        """Initialize unlabeled video dataset
        
        Args:
            image_root: Path to directory containing video frames/images
            transform: Optional albumentations transform
            target_size: Optional (height, width) to resize to
            exts: Tuple of valid image extensions to include
        """
        self.exts = exts
        super().__init__(image_root, transform, target_size)
        logger.info(f"Found {len(self)} frames in {image_root}")
        
    def _collect_samples(self) -> list:
        samples = []
        for root, _, files in os.walk(self.image_root):
            for f in sorted(files):
                if f.lower().endswith(self.exts):
                    img_path = os.path.join(root, f)
                    samples.append(img_path)
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.samples[idx]
        
        # Load image
        image = self._load_image(img_path)
            
        # Prepare sample
        sample = self._prepare_sample(image)
        sample['meta'] = {
            'img_path': img_path,
            'is_labeled': False
        }
        
        return sample
        
class PseudoLabeledDataset(BaseDataset):
    """Dataset for pseudo-labeled data (image + generated mask)"""
    def __init__(
        self,
        image_root: str, 
        mask_dir: str,
        transform: Optional[Compose] = None,
        target_size: Optional[Tuple[int, int]] = None,
        mask_suffix: str = '_mask.png'
    ):
        self.mask_dir = mask_dir
        self.mask_suffix = mask_suffix
        super().__init__(image_root, transform, target_size)
        logger.info(f"Found {len(self)} pseudo-labeled pairs")

    def _collect_samples(self) -> list:
        """Collect image-mask pairs"""
        samples = []
        for root, _, files in os.walk(self.image_root):
            for f in sorted(files):
                if f.lower().endswith(('.png','.jpg','.jpeg')):
                    img_path = os.path.join(root, f)
                    fname_no_ext = os.path.splitext(os.path.basename(f))[0]
                    mask_path = os.path.join(
                        self.mask_dir,
                        fname_no_ext + self.mask_suffix
                    )
                    if os.path.isfile(mask_path):
                        samples.append((img_path, mask_path))
        return samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        
        # Load image
        image = self._load_image(img_path)
        
        # Load generated mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot read mask: {mask_path}")
        mask = mask.astype(np.float32) / 255.0
            
        if self.target_size is not None:
            mask = cv2.resize(
                mask, 
                self.target_size,
                interpolation=cv2.INTER_NEAREST
            )
            
        # Prepare sample
        sample = self._prepare_sample(image, mask)
        return sample

class FinalVesselDataset(ConcatDataset):
    """A dataset that concatenates labeled and pseudo-labeled data."""
    def __init__(
        self,
        image_dir: str,
        labeled_mask_dir: str,
        pseudo_mask_dir: str,
        transform: Optional[Compose] = None,
        copy_paste_prob: float = 0.0
    ):
        labeled_dataset = LabeledVesselDataset(
            image_root=image_dir,
            mask_dir=labeled_mask_dir,
            transform=transform
        )
        pseudo_dataset = PseudoLabeledDataset(
            image_root=image_dir,
            mask_dir=pseudo_mask_dir,
            transform=transform
        )
        super().__init__([labeled_dataset, pseudo_dataset])