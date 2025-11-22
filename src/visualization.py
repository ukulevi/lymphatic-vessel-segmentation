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

def visualize_evaluation_table(metrics_data=None, save_path=None, fold_number=None, epochs_per_fold=None, df_main=None, df_compare=None, model_labels=('Model A', 'Model B')):
    """
    Create a text-based table of evaluation metrics.
    Handles single model from CSV/DataFrame or comparison of two models.
    
    Args:
        metrics_data (str or pd.DataFrame, optional): Path to metrics.csv or a DataFrame for a single model.
        save_path (str, optional): Path to save the table text file.
        fold_number (int, optional): The specific fold to display.
        epochs_per_fold (int, optional): Number of epochs per fold.
        df_main (pd.DataFrame, optional): DataFrame for the first model (for comparison).
        df_compare (pd.DataFrame, optional): DataFrame for the second model (for comparison).
        model_labels (tuple, optional): Names for the models being compared.
    """
    is_comparison = df_main is not None and df_compare is not None

    # --- Data Cleaning Function ---
    def clean_df(df):
        df = df.dropna(axis=1, how='all')
        def clean_tensor_str(val):
            if isinstance(val, str) and 'tensor' in val:
                match = re.search(r"tensor\((.*?)[,)]", val)
                return float(match.group(1)) if match else np.nan
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(clean_tensor_str)
        return df

    if is_comparison:
        # --- Comparison Logic ---
        df1 = clean_df(df_main.copy())
        df2 = clean_df(df_compare.copy())
        
        # Get the last row (best epoch) for each model
        summary1 = df1.iloc[-1:].copy()
        summary2 = df2.iloc[-1:].copy()
        
        summary1.insert(0, 'Model', model_labels[0])
        summary2.insert(0, 'Model', model_labels[1])
        
        # Combine summaries and set Model as index
        combined_summary = pd.concat([summary1, summary2], ignore_index=True)
        
        # Select relevant validation metrics, if available
        val_cols = [col for col in combined_summary.columns if col.startswith('val_')]
        if not val_cols: # Fallback to train if no val
            val_cols = [col for col in combined_summary.columns if not col in ['Model', 'epoch']]
            
        display_cols = ['Model'] + val_cols
        combined_summary = combined_summary[display_cols].round(4)
        
        # Transpose for a vertical comparison table
        table_df = combined_summary.set_index('Model').T

        title = f"\n{'='*80}\nMETRICS COMPARISON: {model_labels[0]} vs {model_labels[1]}\n{'='*80}\n"
        table = tabulate(table_df, headers='keys', tablefmt='grid')
        full_table = title + table
        
    elif metrics_data is not None:
        # --- Single Model Logic ---
        if isinstance(metrics_data, pd.DataFrame):
            metrics_df = metrics_data
        else:
            metrics_df = pd.read_csv(metrics_data)

        metrics_df = clean_df(metrics_df.copy())
        
        # Add fold column if k-fold is detected
        if 'epoch' in metrics_df.columns and (metrics_df['epoch'].diff() < 0).any():
            metrics_df.insert(0, 'fold', (metrics_df['epoch'].diff() < 0).cumsum() + 1)

        if fold_number is not None and epochs_per_fold is not None:
            start_idx = (fold_number - 1) * epochs_per_fold
            end_idx = fold_number * epochs_per_fold
            metrics_df = metrics_df.iloc[start_idx:end_idx].copy()
            metrics_df['epoch'] = range(1, len(metrics_df) + 1)
            title_suffix = f" - Fold {fold_number}"
        else:
            title_suffix = ""

        numeric_cols = metrics_df.select_dtypes(include=np.number).columns
        metrics_df[numeric_cols] = metrics_df[numeric_cols].round(4)

        table = tabulate(metrics_df, headers='keys', tablefmt='grid', showindex=False)
        title = f"\n{'='*80}\nTRAINING METRICS SUMMARY{title_suffix}\n{'='*80}\n\n"
        full_table = title + table
    else:
        raise ValueError("Either 'metrics_data' or both 'df_main' and 'df_compare' must be provided.")

    # Save the table
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_table)
        print(f"Comparison table saved to {save_path}")
    else:
        print(full_table)

    return full_table

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









def plot_training_curves(metrics_csv_path=None, save_path=None, show_val=True, fold_number=None, epochs_per_fold=None, metrics_to_plot=None, df_main=None, df_compare=None, model_labels=('Model A', 'Model B')):
    """
    Plot training curves. Handles single model from CSV or comparison of two models from DataFrames.

    Args:
        metrics_csv_path (str, optional): Path to the metrics.csv file for single model plotting.
        save_path (str, optional): Path to save the figure.
        show_val (bool, optional): Whether to show validation metrics.
        fold_number (int, optional): The specific fold (1-N) to display.
        epochs_per_fold (int, optional): Number of epochs per fold.
        metrics_to_plot (list, optional): List of metrics to plot (e.g., ['loss', 'iou']).
        df_main (pd.DataFrame, optional): DataFrame for the first model (for comparison).
        df_compare (pd.DataFrame, optional): DataFrame for the second model (for comparison).
        model_labels (tuple, optional): A tuple of names for the models being compared.
    """
    is_comparison = df_main is not None and df_compare is not None

    if is_comparison:
        dfs = {model_labels[0]: df_main, model_labels[1]: df_compare}
        title_suffix = f" - {model_labels[0]} vs {model_labels[1]}"
    elif metrics_csv_path:
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
        dfs = {"single_model": df}
    else:
        raise ValueError("Either 'metrics_csv_path' or both 'df_main' and 'df_compare' must be provided.")

    if metrics_to_plot is None:
        metrics_to_plot = ['loss', 'iou', 'dice', 'pixel_acc', 'boundary_f1']

    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 4))
    if num_metrics == 1:
        axes = [axes] # Make it iterable

    fig.suptitle(f'Training Curves{title_suffix}', fontsize=16, fontweight='bold')

    colors = {'Train': 'b', 'Val': 'r'}
    linestyles = {model_labels[0]: '-', model_labels[1]: '--'}

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        
        for model_name, df in dfs.items():
            linestyle = linestyles.get(model_name, '-') if is_comparison else '-'
            
            epochs = df['epoch'].values
            
            # --- Plot Training Curve ---
            train_metric_name = None
            if f'train_{metric}' in df.columns:
                train_metric_name = f'train_{metric}'
            elif metric == 'loss' and 'loss' in df.columns: # For Mean Teacher logs
                train_metric_name = 'loss'
            elif metric == 'loss' and 'train_loss' in df.columns: # For baseline logs
                train_metric_name = 'train_loss'
            
            if train_metric_name and train_metric_name in df.columns:
                label = f'{model_name} Train' if is_comparison else 'Train'
                ax.plot(epochs, df[train_metric_name], color=colors['Train'], linestyle=linestyle, label=label, marker='o', markersize=3, alpha=0.8)

            # --- Plot Validation Curve ---
            val_metric = f'val_{metric}'
            if show_val and val_metric in df.columns and not df[val_metric].isnull().all():
                label = f'{model_name} Val' if is_comparison else 'Val'
                ax.plot(epochs, df[val_metric], color=colors['Val'], linestyle=linestyle, label=label, marker='s', markersize=3, alpha=0.8)

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
