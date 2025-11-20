"""
Evaluation metrics for segmentation
"""
import torch
import numpy as np
import cv2

def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient
    Args:
        pred: prediction mask (B,1,H,W) or (H,1,H,W)
        target: ground truth mask (B,1,H,W) or (H,W)
        smooth: smoothing factor to avoid division by zero
    Returns:
        dice coefficient
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    if pred.ndim == 4:
        axes = (2,3) # for batched input
    else:
        axes = (0,1) # for single image
        
    intersection = np.sum(pred * target, axis=axes)
    pred_sum = np.sum(pred, axis=axes)
    target_sum = np.sum(target, axis=axes)
    
    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
    return np.mean(dice)

def iou_score(pred, target, smooth=1e-6):
    """
    Calculate IoU (Intersection over Union)
    Args:
        pred: prediction mask (B,1,H,W) or (H,W)
        target: ground truth mask (B,1,H,W) or (H,W) 
        smooth: smoothing factor
    Returns:
        IoU score
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    if pred.ndim == 4:
        axes = (2,3)
    else:
        axes = (0,1)
        
    intersection = np.sum(pred * target, axis=axes)
    union = np.sum(pred + target, axis=axes) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return np.mean(iou)

def pixel_accuracy(pred, target):
    """Pixel accuracy over batch or single image.
    Args:
        pred: binary mask (B,1,H,W) or (H,W)
        target: binary mask (B,1,H,W) or (H,W)
    Returns:
        float accuracy
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    if pred.ndim == 4:
        correct = (pred == target).sum(axis=(1,2,3))
        total = np.prod(pred.shape[1:])
        return float(correct.mean() / total)
    else:
        correct = (pred == target).sum()
        total = pred.size
        return float(correct / total)

def _binary_boundary(mask: np.ndarray) -> np.ndarray:
    """Compute 1-pixel wide boundary of binary mask (H,W) using morphological gradient.
    Returns uint8 0/1 array.
    """
    mask_u8 = (mask > 0.5).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    grad = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0).astype(np.uint8)

def boundary_f1_score(pred, target, tolerance: int = 2) -> float:
    """Boundary F1 score with tolerance via dilated boundary overlap.
    Args:
        pred: binary mask (B,1,H,W) or (H,W)
        target: binary mask (B,1,H,W) or (H,W)
        tolerance: pixel tolerance radius for boundary matching
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure shape (N,H,W)
    if pred.ndim == 4:
        pred_arr = pred[:,0]
        targ_arr = target[:,0]
    else:
        pred_arr = pred[None,...]
        targ_arr = target[None,...]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tolerance+1, 2*tolerance+1))
    f1_scores = []
    for p, t in zip(pred_arr, targ_arr):
        pb = _binary_boundary(p)
        tb = _binary_boundary(t)
        if pb.sum() == 0 and tb.sum() == 0:
            f1_scores.append(1.0)
            continue
        if pb.sum() == 0 or tb.sum() == 0:
            f1_scores.append(0.0)
            continue
        pb_dil = cv2.dilate(pb, kernel)
        tb_dil = cv2.dilate(tb, kernel)
        # Matches
        p_match = (pb & tb_dil).sum()
        t_match = (tb & pb_dil).sum()
        precision = p_match / (pb.sum() + 1e-6)
        recall = t_match / (tb.sum() + 1e-6)
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        f1_scores.append(float(f1))
    return float(np.mean(f1_scores))

def avg_score(iou: float, dice: float, pixel_acc: float, boundary_f1: float,
              weights: dict | None = None) -> float:
    """Weighted average score across metrics.
    Default weights follow references: IoU=0.2, Dice=0.25, PixelAcc=0.15, BoundaryF1=0.4
    """
    if weights is None:
        weights = {"iou": 0.2, "dice": 0.25, "pixel_acc": 0.15, "boundary_f1": 0.4}
    wsum = sum(weights.values()) + 1e-8
    total = (
        iou * weights.get("iou", 0.0)
        + dice * weights.get("dice", 0.0)
        + pixel_acc * weights.get("pixel_acc", 0.0)
        + boundary_f1 * weights.get("boundary_f1", 0.0)
    )
    return float(total / wsum)

def precision_recall(pred, target):
    """
    Calculate precision and recall
    Args:
        pred: prediction mask (B,1,H,W) or (H,W)
        target: ground truth mask (B,1,H,W) or (H,W)
    Returns:
        precision, recall
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
        
    if pred.ndim == 4:
        axes = (2,3)
    else:
        axes = (0,1)
        
    tp = np.sum(pred * target, axis=axes)
    fp = np.sum(pred * (1-target), axis=axes)
    fn = np.sum((1-pred) * target, axis=axes)
    
    precision = np.mean(tp / (tp + fp + 1e-6))
    recall = np.mean(tp / (tp + fn + 1e-6))
    
    return precision, recall
