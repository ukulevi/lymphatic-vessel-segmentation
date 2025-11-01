"""
Image and data augmentation utilities
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

NORMALIZE_PARAMS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

def get_base_transforms():
    return [
        A.Resize(256, 256),
        A.Normalize(**NORMALIZE_PARAMS),
        ToTensorV2()
    ]

def get_train_transforms(p=0.5):
    return A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        *get_base_transforms()
    ])

def create_train_transform(p=0.5):
    """
    Create training augmentation pipeline
    Args:
        p: probability of applying each augmentation
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.OneOf([
            A.ElasticTransform(p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(p=1.0)
        ], p=p),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0)
        ], p=p),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
def create_val_transform():
    """Create validation/test transform pipeline"""
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def create_weak_transform(p=0.3):
    """
    Create weak augmentation for student model
    Args:
        p: probability of applying each augmentation
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.9, 1.0)), 
        A.HorizontalFlip(p=p),
        A.RandomBrightnessContrast(p=p),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def create_strong_transform(p=0.5):
    """
    Create strong augmentation for teacher model
    Args:
        p: probability of applying each augmentation  
    Returns:
        albumentations.Compose object
    """
    return A.Compose([
        A.RandomResizedCrop(size=(256, 256), scale=(0.7, 1.0)),
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.OneOf([
            A.ElasticTransform(p=1.0),
            A.GridDistortion(p=1.0),
            A.OpticalDistortion(p=1.0) 
        ], p=p),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0)
        ], p=p),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=p),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])