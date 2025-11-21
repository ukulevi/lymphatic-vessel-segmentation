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

def visualize_evaluation_table(metrics_data, save_path, fold_number=None, epochs_per_fold=None):
    """
    Create a text-based table visualization of evaluation metrics using tabulate.
    Args:
        metrics_data (str or pd.DataFrame): Path to the metrics.csv file or a pandas DataFrame.
        save_path (str): Path to save the table text file.
        fold_number: The specific fold (1-N) to display. If None, all are shown.
        epochs_per_fold: Number of epochs per fold (auto-calculated if None).
    """
    if isinstance(metrics_data, pd.DataFrame):
        metrics_df = metrics_data
    else:
        metrics_df = pd.read_csv(metrics_data)

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

        # Check for the 5-output CTO model structure
        if is_tuple and num_outputs == 5:
            print("  (i) Detected 5-output CTO model. Visualizing all outputs.")
            # Unpack all outputs: o, o3, o2, o1, oe
            o_final_raw = torch.sigmoid(outputs[0])
            o3_raw = torch.sigmoid(outputs[1])
            o2_raw = torch.sigmoid(outputs[2])
            o1_raw = torch.sigmoid(outputs[3])
            oe_raw = torch.sigmoid(outputs[4])
        else: # Fallback for other models
            print(f"  (i) Detected {num_outputs}-output model. Visualizing main output.")
            main_output = outputs[0] if is_tuple else outputs
            o_final_raw = torch.sigmoid(main_output)
            # Set others to None so they don't get plotted
            o3_raw, o2_raw, o1_raw, oe_raw = None, None, None, None

    # --- Convert to numpy ---
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    # --- Determine Figure Layout ---
    # Layout: Input, GT, o3, o2, o1, oe, Final(o), Cleaned, Overlay
    num_cols = 9
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols * 3, num_samples * 3))
    if num_samples == 1:
        axes = axes.reshape(1, -1) # Ensure axes is always 2D

    # --- Process and Display Each Sample ---
    for i in range(num_samples):
        # --- Prepare data for the current sample ---
        img = images_np[i].transpose(1, 2, 0)
        img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
        gt_mask = masks_np[i, 0]
        
        # --- Column 0: Input & Column 1: Ground Truth ---
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt_mask, cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # --- Columns 2-6: Raw Auxiliary Outputs ---
        axes[i, 2].imshow(o3_raw[i, 0].cpu().numpy(), cmap='gray'); axes[i, 2].set_title('Raw o3'); axes[i, 2].axis('off')
        axes[i, 3].imshow(o2_raw[i, 0].cpu().numpy(), cmap='gray'); axes[i, 3].set_title('Raw o2'); axes[i, 3].axis('off')
        axes[i, 4].imshow(o1_raw[i, 0].cpu().numpy(), cmap='gray'); axes[i, 4].set_title('Raw o1'); axes[i, 4].axis('off')
        axes[i, 5].imshow(oe_raw[i, 0].cpu().numpy(), cmap='gray'); axes[i, 5].set_title('Raw oe (edge)'); axes[i, 5].axis('off')

        # --- Columns 7-9: Final Prediction and Overlays ---
        final_pred_raw = o_final_raw[i, 0].cpu().numpy()
        final_pred_binary = (final_pred_raw > 0.5).astype(np.uint8)
        final_pred_cleaned = post_process_mask(final_pred_binary) if use_post_process else final_pred_binary

        axes[i, 6].imshow(final_pred_raw, cmap='gray'); axes[i, 6].set_title('Final Pred (o)'); axes[i, 6].axis('off')
        axes[i, 7].imshow(final_pred_cleaned, cmap='gray'); axes[i, 7].set_title('Cleaned Final'); axes[i, 7].axis('off')
        
        # Overlay the cleaned final prediction on the original image
        pred_overlay = overlay_mask(img, final_pred_cleaned, color=(0, 1, 0)) # Green overlay
        axes[i, 8].imshow(pred_overlay); axes[i, 8].set_title('Final Overlay'); axes[i, 8].axis('off')

    plt.tight_layout()
    return fig









def plot_training_curves(metrics_csv_path, save_path=None, show_val=True, fold_number=None, epochs_per_fold=None, metrics_to_plot=None):
    """
    Plot training curves from a metrics.csv file.
    This function can plot a simple loss curve or a comprehensive set of metrics.

    Args:
        metrics_csv_path (str): Path to the metrics.csv file.
        save_path (str, optional): Path to save the figure. Defaults to None.
        show_val (bool, optional): Whether to show validation metrics. Defaults to True.
        fold_number (int, optional): The specific fold (1-N) to display. If None, all are shown.
        epochs_per_fold (int, optional): Number of epochs per fold.
        metrics_to_plot (list, optional): A list of metrics to plot. If None, plots a default set.
                                           Example: ['loss', 'iou', 'dice']
    """
    df = pd.read_csv(metrics_csv_path)

    if fold_number is not None and epochs_per_fold is not None:
        start_idx = (fold_number - 1) * epochs_per_fold
        end_idx = fold_number * epochs_per_fold
        df_filtered = df.iloc[start_idx:end_idx].copy()
        df_filtered['epoch'] = range(1, len(df_filtered) + 1)
        df = df_filtered
        title_suffix = f" - Fold {fold_number}"
    else:
        title_suffix = ""

    if metrics_to_plot is None:
        metrics_to_plot = ['loss', 'iou', 'dice', 'pixel_acc', 'boundary_f1']

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1:
        axes = [axes] # Make it iterable

    fig.suptitle(f'Training Curves{title_suffix}', fontsize=16, fontweight='bold')

    epochs = df['epoch'].values

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        # Find the correct training metric column. It could be 'train_loss', 'loss', or specific to the metric.
        train_metric_name = None
        if f'train_{metric}' in df.columns:
            train_metric_name = f'train_{metric}'
        elif metric == 'loss' and 'loss' in df.columns: # For Mean Teacher logs
            train_metric_name = 'loss'
        elif metric == 'loss' and 'train_loss' in df.columns: # For baseline logs
            train_metric_name = 'train_loss'

        val_metric = f'val_{metric}'

        # Plot training and validation curves
        if train_metric_name and train_metric_name in df.columns:
            ax.plot(epochs, df[train_metric_name], 'b-', label='Train', linewidth=2, marker='o', markersize=4)
        if show_val and val_metric in df.columns and not df[val_metric].isnull().all():
            ax.plot(epochs, df[val_metric], 'r-', label='Val', linewidth=2, marker='s', markersize=4)

        ax.set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    return fig
