"""
Visualization utilities for model predictions.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pandas as pd
import segmentation_models_pytorch as smp
from tabulate import tabulate
import cv2
from scipy import ndimage

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Overlay a mask on an image.

    Args:
        image: The input image.
        mask: The mask to overlay.
        color: The color of the mask.
        alpha: The transparency of the mask.

    Returns:
        The image with the mask overlayed.
    """
    # Create a colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color

    # Blend the image and the mask
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def visualize_evaluation_table(metrics_path, save_path, fold_number=None, epochs_per_fold=None):
    """
    Create a text-based table visualization of evaluation metrics using tabulate.
    Args:
        metrics_path (str): Path to the metrics.csv file.
        save_path (str): Path to save the table text file.
        fold_number: Số fold cụ thể (1-5) để hiển thị. Nếu None thì hiển thị tất cả
        epochs_per_fold: Số epochs mỗi fold (tự động tính nếu None)
    """
    metrics_df = pd.read_csv(metrics_path)
    
    # Nếu chọn fold cụ thể
    if fold_number is not None and epochs_per_fold is not None:
        start_idx = (fold_number - 1) * epochs_per_fold
        end_idx = fold_number * epochs_per_fold
        metrics_df = metrics_df.iloc[start_idx:end_idx].copy()
        metrics_df['epoch'] = range(1, len(metrics_df) + 1)
        title_suffix = f" - Fold {fold_number}"
    else:
        title_suffix = ""
    
    # Round the metrics to 4 decimal places for better readability
    metrics_df = metrics_df.round(4)

    # Generate the table using tabulate
    table = tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False)
    
    # Add title if fold specified
    if fold_number:
        title = f"\n{'='*80}\nTRAINING METRICS{title_suffix}\n{'='*80}\n\n"
        full_table = title + table
    else:
        full_table = table
    
    # Save the table to a text file
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(full_table)
    
    # Print summary instead of full table
    if fold_number:
        print(f"✓ Fold {fold_number}: {len(metrics_df)} epochs")
    else:
        print(f"✓ Total: {len(metrics_df)} epochs")
    print(f"  Saved to: {save_path}")

def post_process_mask(mask, min_size=100, morph_kernel_size=3):
    """
    Post-process segmentation mask to remove noise and small artifacts.
    
    Args:
        mask: Binary mask (numpy array, 0-1 or 0-255)
        min_size: Minimum size of connected component to keep
        morph_kernel_size: Size of morphological kernel
        
    Returns:
        Cleaned binary mask
    """
    # Ensure binary mask (0-1)
    if mask.max() > 1:
        mask = (mask > 127).astype(np.uint8)
    else:
        mask = (mask > 0.5).astype(np.uint8)
    
    # Morphological closing (fill holes)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Remove small connected components
    labeled, num_features = ndimage.label(mask_closed)
    sizes = np.bincount(labeled.ravel())[1:]  # Skip background (0)
    
    if len(sizes) > 0:
        # Keep only components larger than min_size
        mask_cleaned = np.zeros_like(mask_closed)
        for label_id, size in enumerate(sizes, start=1):
            if size >= min_size:
                mask_cleaned[labeled == label_id] = 1
    else:
        mask_cleaned = mask_closed
    
    # Morphological opening (remove small protrusions)
    mask_opened = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)
    
    # Gaussian blur for smooth edges
    mask_smooth = cv2.GaussianBlur(mask_opened.astype(np.float32), (5, 5), 0)
    mask_final = (mask_smooth > 0.5).astype(np.uint8)
    
    return mask_final

def visualize_predictions(model, dataset, num_samples=4, device='cuda', use_post_process=True):
    """
    Visualize model predictions on random samples from dataset.
    Args:
        model: Trained model
        dataset: Dataset to sample from
        num_samples: Number of samples to visualize
        device: Device to run model on
        use_post_process: If True, apply post-processing to predictions
    Returns:
        Matplotlib figure
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(loader))
    
    images = batch['image'].to(device)
    masks = batch['mask']
    
    with torch.no_grad():
        outputs = model(images)
        if isinstance(outputs, dict):
            outputs = outputs['out']
        preds = torch.sigmoid(outputs) > 0.5
    
    # Convert to numpy
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    preds_np = preds.cpu().numpy().astype(np.uint8)
    
    # Post-process predictions if requested
    if use_post_process:
        preds_cleaned = []
        for i in range(num_samples):
            cleaned = post_process_mask(preds_np[i,0], min_size=100, morph_kernel_size=3)
            preds_cleaned.append(cleaned)
        preds_cleaned = np.array(preds_cleaned)
    else:
        preds_cleaned = preds_np[:,0]
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4*num_samples))
    
    for i in range(num_samples):
        # Original image
        img = images_np[i].transpose(1,2,0)
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0,1)
        axes[i,0].imshow(img)
        axes[i,0].set_title('Input')
        axes[i,0].axis('off')
        
        # Ground truth
        gt_mask = masks_np[i,0]
        axes[i,1].imshow(gt_mask, cmap='gray')
        axes[i,1].set_title('Ground Truth')
        axes[i,1].axis('off')
        
        # Raw Prediction (before post-processing)
        pred_mask_raw = preds_np[i,0]
        axes[i,2].imshow(pred_mask_raw, cmap='gray')
        axes[i,2].set_title('Prediction (Raw)')
        axes[i,2].axis('off')
        
        # Cleaned Prediction (after post-processing)
        if use_post_process:
            pred_mask_cleaned = preds_cleaned[i]
            axes[i,3].imshow(pred_mask_cleaned, cmap='gray')
            axes[i,3].set_title('Prediction (Cleaned)')
            axes[i,3].axis('off')
        else:
            axes[i,3].axis('off')

        # Ground Truth Overlay
        gt_overlay = overlay_mask(img, gt_mask, color=(0, 1, 0)) # Green for GT
        axes[i,4].imshow(gt_overlay)
        axes[i,4].set_title('GT Overlay')
        axes[i,4].axis('off')

        # Prediction Overlay (use cleaned if available)
        pred_mask_display = preds_cleaned[i] if use_post_process else pred_mask_raw
        pred_overlay = overlay_mask(img, pred_mask_display, color=(1, 0, 0)) # Red for prediction
        axes[i,5].imshow(pred_overlay)
        axes[i,5].set_title('Pred Overlay (Cleaned)' if use_post_process else 'Pred Overlay')
        axes[i,5].axis('off')

    plt.tight_layout()
    return fig

def evaluate_metrics(model, dataset, device='cuda'):
    """
    Evaluate model on dataset using standard metrics.
    Args:
        model: Trained model
        dataset: Dataset to evaluate on
        device: Device to run model on
    Returns:
        Dict of metrics
    """
    model.eval()
    metrics = smp.utils.metrics.IoU(threshold=0.5)
    
    loader = DataLoader(dataset, batch_size=8)
    scores = []
    
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            masks = batch['mask']
            
            outputs = model(images)
            if isinstance(outputs, dict):
                outputs = outputs['out']
            preds = torch.sigmoid(outputs)
            
            score = metrics(preds, masks.to(device))
            scores.append(score.cpu().numpy())
    
    mean_iou = np.mean(scores)
    return {'IoU': mean_iou}

def plot_loss_curve(metrics_csv_path, save_path=None, show_val=True, fold_number=None, epochs_per_fold=None):
    """
    Plot đơn giản: Epoch vs Loss
    Giống như trong hình - chỉ hiển thị loss theo epoch
    
    Args:
        metrics_csv_path: Path to metrics.csv file
        save_path: Optional path to save figure
        show_val: Nếu True thì hiển thị cả train và val loss, nếu False chỉ train loss
        fold_number: Số fold cụ thể (1-5) để hiển thị. Nếu None thì hiển thị tất cả
        epochs_per_fold: Số epochs mỗi fold (tự động tính nếu None)
    
    Returns:
        Matplotlib figure
    """
    # Đọc CSV
    df = pd.read_csv(metrics_csv_path)
    
    # Nếu chọn fold cụ thể
    if fold_number is not None and epochs_per_fold is not None:
        # Tính toán range của fold
        start_idx = (fold_number - 1) * epochs_per_fold
        end_idx = fold_number * epochs_per_fold
        
        # Filter data của fold đó
        df_filtered = df.iloc[start_idx:end_idx].copy()
        
        # Reset epoch về 1-N cho fold này
        df_filtered['epoch'] = range(1, len(df_filtered) + 1)
        
        df = df_filtered
        title_suffix = f" - Fold {fold_number}"
    else:
        title_suffix = ""
    
    # Tạo figure đơn giản
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    epochs = df['epoch'].values
    
    # Vẽ train loss
    ax.plot(epochs, df['train_loss'], 'b-', label='Training Loss', 
            linewidth=2, marker='s', markersize=8, markerfacecolor='blue', markeredgecolor='blue')
    
    # Vẽ val loss nếu muốn
    if show_val and 'val_loss' in df.columns:
        ax.plot(epochs, df['val_loss'], 'r-', label='Validation Loss',
                linewidth=2, marker='o', markersize=8, markerfacecolor='red', markeredgecolor='red')
    
    ax.set_title(f'Training Loss Over Epochs{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Đặt ticks rõ ràng hơn
    ax.set_xticks(epochs)
    
    plt.tight_layout()
    
    # Save nếu có path
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Loss curve saved to {save_path}")
    
    return fig

def plot_training_curves_from_csv(metrics_csv_path, save_path=None):
    """
    Plot training curves từ metrics.csv file.
    Không cần train lại, chỉ cần đọc CSV.
    
    Args:
        metrics_csv_path: Path to metrics.csv file
        save_path: Optional path to save figure (nếu None thì chỉ hiển thị)
    
    Returns:
        Matplotlib figure
    """
    # Đọc CSV
    df = pd.read_csv(metrics_csv_path)
    
    # Tạo figure với nhiều subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves', fontsize=16, fontweight='bold')
    
    epochs = df['epoch'].values
    
    # 1. Loss curves
    axes[0, 0].plot(epochs, df['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
    axes[0, 0].plot(epochs, df['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
    axes[0, 0].set_title('Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. IoU curves
    axes[0, 1].plot(epochs, df['val_iou'], 'g-', label='Val IoU', linewidth=2, marker='o', markersize=4)
    axes[0, 1].set_title('IoU Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Dice curves
    axes[1, 0].plot(epochs, df['val_dice'], 'm-', label='Val Dice', linewidth=2, marker='o', markersize=4)
    axes[1, 0].set_title('Dice Coefficient', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Combined metrics
    axes[1, 1].plot(epochs, df['val_iou'], label='IoU', linewidth=2)
    axes[1, 1].plot(epochs, df['val_dice'], label='Dice', linewidth=2)
    axes[1, 1].plot(epochs, df['val_pixel_acc'], label='Pixel Acc', linewidth=2)
    axes[1, 1].plot(epochs, df['val_boundary_f1'], label='Boundary F1', linewidth=2)
    axes[1, 1].set_title('All Metrics', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save nếu có path
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    return fig

def plot_training_curves(history):
    """
    Plot training curves for loss and IoU.
    Args:
        history: Training history containing loss and IoU
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    
    axes[1].plot(history['train_iou'], label='Train')
    axes[1].plot(history['val_iou'], label='Val')
    axes[1].set_title('IoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].legend()
    
    plt.tight_layout()
    return fig

if __name__ == '__main__':
    # You can change this path to point to your desired metrics.csv file
    metrics_file = 'c:\\Users\\PC\\Downloads\\ai_doantonghop\\BCP_self_imple\\logs\\20251029_183722\\metrics.csv'
    print(f"Evaluation table from {metrics_file}")
    output_path = 'evaluation_table.txt'
    visualize_evaluation_table(metrics_file, output_path)