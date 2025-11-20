"""
Visualization utilities for model predictions.
"""
import re
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
        fold_number: The specific fold (1-N) to display. If None, all are shown.
        epochs_per_fold: Number of epochs per fold (auto-calculated if None).
    """
    metrics_df = pd.read_csv(metrics_path)

    # --- Data Cleaning and Preparation ---

    # 1. Drop completely empty columns to clean up the table
    original_cols = metrics_df.columns
    metrics_df = metrics_df.dropna(axis=1, how='all')
    dropped_cols = set(original_cols) - set(metrics_df.columns)
    if dropped_cols:
        print(f"  (i) Dropping empty columns: {sorted(list(dropped_cols))}")

    # 2. Define a function to clean tensor string representations
    def clean_tensor_str(val):
        if isinstance(val, str) and 'tensor' in val:
            match = re.search(r"tensor\((.*?)[,)]", val)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, TypeError):
                    return np.nan # Return NaN if conversion fails
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan # Return NaN for non-numeric strings

    # 3. Apply cleaning to all columns that are not purely numeric
    for col in metrics_df.columns:
        if metrics_df[col].dtype == 'object':
            metrics_df[col] = metrics_df[col].apply(clean_tensor_str)

    # 4. Add a 'fold' column for clarity if k-fold is detected
    if 'epoch' in metrics_df.columns and (metrics_df['epoch'].diff() < 0).any():
        print("  (i) K-fold data detected, adding 'fold' column for clarity.")
        metrics_df.insert(0, 'fold', (metrics_df['epoch'].diff() < 0).cumsum() + 1)

    # 5. Handle fold-specific display if requested
    if fold_number is not None and epochs_per_fold is not None:
        start_idx = (fold_number - 1) * epochs_per_fold
        end_idx = fold_number * epochs_per_fold
        metrics_df = metrics_df.iloc[start_idx:end_idx].copy()
        metrics_df['epoch'] = range(1, len(metrics_df) + 1)
        title_suffix = f" - Fold {fold_number}"
    else:
        title_suffix = ""

    # 6. Round all numeric columns for better readability
    numeric_cols = metrics_df.select_dtypes(include=np.number).columns
    metrics_df[numeric_cols] = metrics_df[numeric_cols].round(4)

    # --- Table Generation ---

    # Generate the table using tabulate
    table = tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False)
    
    title = f"\n{'='*80}\nTRAINING METRICS SUMMARY{title_suffix}\n{'='*80}\n\n"
    full_table = title + table
    
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
    Handles single-output, dual-output, and special four-output (CTO) models.
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=num_samples, shuffle=True)
    batch = next(iter(loader))

    images = batch['image'].to(device)
    masks = batch['mask']

    with torch.no_grad():
        outputs = model(images)

        # --- Handle different output structures ---
        is_tuple = isinstance(outputs, tuple)
        num_outputs = len(outputs) if is_tuple else 1
        
        # Default single output
        core_preds = None
        boundary_preds = None
        
        # Four-output model (CTO-Net)
        if is_tuple and num_outputs == 4:
            print("  (i) Detected 4-output model (CTO-Net). Visualizing all outputs.")
            # o3, o2, o1, oe
            o3_preds = torch.sigmoid(outputs[0]) > 0.5
            o2_preds = torch.sigmoid(outputs[1]) > 0.5
            o1_preds = torch.sigmoid(outputs[2]) > 0.5
            oe_preds = torch.sigmoid(outputs[3]) > 0.5
            
            # Per analysis, o3 (outputs[0]) is the best semantic prediction.
            # We will use it as the base for the final segmentation.
            core_preds = o3_preds
            boundary_preds = None # Not using a separate boundary for final combination

        # Dual-output model
        elif is_tuple and num_outputs > 1:
            print("  (i) Detected 2-output model. Visualizing core and boundary.")
            core_output, boundary_output = outputs[0], outputs[1]
            core_preds = torch.sigmoid(core_output) > 0.5
            boundary_preds = torch.sigmoid(boundary_output) > 0.5
            
        # Single-output model
        else:
            print("  (i) Detected 1-output model.")
            single_output = outputs[0] if is_tuple else outputs
            if isinstance(single_output, dict): # Handle models like DeepLabV3
                single_output = single_output['out']
            core_preds = torch.sigmoid(single_output) > 0.5

    # --- Convert to numpy ---
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    # --- Determine Figure Layout ---
    if num_outputs == 4:
        num_cols = 10
    elif num_outputs > 1:
        num_cols = 8
    else:
        num_cols = 6
        
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))

    # --- Process and Display Each Sample ---
    for i in range(num_samples):
        # Prepare image for display
        img = images_np[i].transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

        # --- Column 0: Input Image ---
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        # --- Column 1: Ground Truth Mask ---
        gt_mask = masks_np[i, 0]
        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # --- Visualization for 4-output model (CTO-Net) ---
        if num_outputs == 4:
            # Get individual predictions
            o3_pred = o3_preds[i, 0].cpu().numpy().astype(np.uint8)
            o2_pred = o2_preds[i, 0].cpu().numpy().astype(np.uint8)
            o1_pred = o1_preds[i, 0].cpu().numpy().astype(np.uint8)
            oe_pred = oe_preds[i, 0].cpu().numpy().astype(np.uint8)
            
            # Display all 4 raw outputs
            axes[i, 2].imshow(o3_pred, cmap='gray'); axes[i, 2].set_title('Output 1 (o3)'); axes[i, 2].axis('off')
            axes[i, 3].imshow(o2_pred, cmap='gray'); axes[i, 3].set_title('Output 2 (o2)'); axes[i, 3].axis('off')
            axes[i, 4].imshow(o1_pred, cmap='gray'); axes[i, 4].set_title('Output 3 (o1)'); axes[i, 4].axis('off')
            axes[i, 5].imshow(oe_pred, cmap='gray'); axes[i, 5].set_title('Output 4 (oe)'); axes[i, 5].axis('off')
            
            # Based on analysis, the deepest output (oe_pred, from outputs[3]) is the cleanest.
            # We will use it for the final segmentation.
            # First, we must upsample it to the original image size.
            deepest_pred_np = oe_pred.astype(np.float32)
            if deepest_pred_np.shape != img.shape[:2]:
                # Use cv2.resize for numpy arrays
                deepest_pred_np = cv2.resize(deepest_pred_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            best_pred_cleaned = post_process_mask(deepest_pred_np) if use_post_process else (deepest_pred_np > 0.5).astype(np.uint8)
            axes[i, 6].imshow(best_pred_cleaned, cmap='gray'); axes[i, 6].set_title('Cleaned Deep Out'); axes[i, 6].axis('off')
 
            # Use the cleaned deepest prediction as the final segmentation
            final_seg = best_pred_cleaned
            axes[i, 7].imshow(final_seg, cmap='gray'); axes[i, 7].set_title('Final Seg (from Deep)'); axes[i, 7].axis('off')

            # GT Overlay
            gt_overlay = overlay_mask(img, gt_mask, color=(0, 1, 0))
            axes[i, 8].imshow(gt_overlay); axes[i, 8].set_title('GT Overlay'); axes[i, 8].axis('off')

            # Final Prediction Overlay
            final_overlay = overlay_mask(img, final_seg, color=(0, 0, 1), alpha=0.5) # Blue for final
            axes[i, 9].imshow(final_overlay); axes[i, 9].set_title('Final Overlay'); axes[i, 9].axis('off')

        # --- Visualization for other models ---
        else:
            core_preds_np = core_preds[i, 0].cpu().numpy().astype(np.uint8)
            core_cleaned = post_process_mask(core_preds_np) if use_post_process else core_preds_np
            
            axes[i, 2].imshow(core_preds_np, cmap='gray'); axes[i, 2].set_title('Core (Raw)'); axes[i, 2].axis('off')
            axes[i, 3].imshow(core_cleaned, cmap='gray'); axes[i, 3].set_title('Core (Cleaned)'); axes[i, 3].axis('off')

            # Dual-output models
            if boundary_preds is not None:
                boundary_pred = boundary_preds[i, 0].cpu().numpy().astype(np.uint8)
                axes[i, 4].imshow(boundary_pred, cmap='gray'); axes[i, 4].set_title('Boundary Pred'); axes[i, 4].axis('off')
                
                gt_overlay = overlay_mask(img, gt_mask, color=(0, 1, 0))
                axes[i, 5].imshow(gt_overlay); axes[i, 5].set_title('GT Overlay'); axes[i, 5].axis('off')

                pred_overlay = overlay_mask(img, core_cleaned, color=(1, 0, 0), alpha=0.4)
                pred_overlay = overlay_mask(pred_overlay, boundary_pred, color=(1, 1, 0), alpha=0.6)
                axes[i, 6].imshow(pred_overlay); axes[i, 6].set_title('Pred Overlay'); axes[i, 6].axis('off')

                final_seg = np.logical_or(core_cleaned, boundary_pred).astype(np.uint8)
                final_overlay = overlay_mask(img, final_seg, color=(0, 1, 0), alpha=0.5)
                axes[i, 7].imshow(final_overlay); axes[i, 7].set_title('Final Seg Overlay'); axes[i, 7].axis('off')
            
            # Single-output models
            else:
                # We have 6 columns (0-5)
                # 0: Input, 1: GT, 2: Raw, 3: Cleaned
                
                # Column 4: Prediction Overlay
                pred_overlay = overlay_mask(img, core_cleaned, color=(1, 0, 0))
                axes[i, 4].imshow(pred_overlay); axes[i, 4].set_title('Pred Overlay'); axes[i, 4].axis('off')

                # Column 5: Ground Truth Overlay
                gt_overlay = overlay_mask(img, gt_mask, color=(0, 1, 0))
                axes[i, 5].imshow(gt_overlay); axes[i, 5].set_title('GT Overlay'); axes[i, 5].axis('off')
                
                # Hide other unused axes
                for k in range(6, num_cols):
                    axes[i, k].axis('off')

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
            if isinstance(outputs, tuple):
                outputs = outputs[0]
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
    import argparse
    import os
    parser = argparse.ArgumentParser(description="Generate evaluation tables from metrics.")
    parser.add_argument("--metrics_file", type=str, required=True, help="Path to the metrics.csv file from a training run.")
    parser.add_argument("--type", type=str, choices=["Human", "Rat"], default="Human", help="The type of model/data.")
    args = parser.parse_args()

    # Create the output directory inside the correct typed models folder
    output_dir = os.path.join("models", args.type)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating evaluation table from {args.metrics_file} for type '{args.type}'")
    output_path = os.path.join(output_dir, 'evaluation_summary.txt')
    visualize_evaluation_table(args.metrics_file, output_path)
    print(f"Saved summary table to {output_path}")
