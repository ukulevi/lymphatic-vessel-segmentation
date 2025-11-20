import os
import sys
import pandas as pd
from src.visualization import visualize_evaluation_table

def main():
    # Find the latest metrics.csv
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"Directory {logs_dir} not found")
        return
    
    # Get the latest folder
    log_folders = []
    for root, dirs, files in os.walk(logs_dir):
        if "metrics.csv" in files:
            log_folders.append(root)

    if not log_folders:
        print("No log folders with metrics.csv found")
        return
    
    # Sort by name (timestamp)
    log_folders.sort(reverse=True)
    latest_folder = log_folders[0]
    metrics_file = os.path.join(latest_folder, "metrics.csv")
    model_type = os.path.basename(os.path.dirname(latest_folder))
    model_dir = os.path.join("models", model_type)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(metrics_file):
        print(f"metrics.csv not found in {latest_folder}")
        # Allow user to enter path
        metrics_file = input("Enter the path to metrics.csv: ").strip()
        if not os.path.exists(metrics_file):
            print(f"File not found: {metrics_file}")
            return
    
    print(f"Reading metrics from: {metrics_file}")
    
    # Read CSV to calculate the number of epochs per fold
    df = pd.read_csv(metrics_file)
    total_rows = len(df)
    
    # Automatically detect the number of epochs per fold (when epoch resets to 1)
    fold_starts = [0]
    for i in range(1, len(df)):
        if df.iloc[i]['epoch'] < df.iloc[i-1]['epoch']:
            fold_starts.append(i)
    fold_starts.append(len(df))
    
    epochs_per_fold_list = [fold_starts[i+1] - fold_starts[i] for i in range(len(fold_starts)-1)]
    epochs_per_fold = epochs_per_fold_list[0] if epochs_per_fold_list else None  # Keep for compatibility
    
    print(f"\nTotal number of metric rows: {total_rows}")
    print(f"Number of epochs per fold: {epochs_per_fold_list}")
    
    # Print all 5 folds
    print(f"\n{'='*80}")
    print("PRINTING METRICS TABLE FOR ALL 5 FOLDS")
    print(f"{'='*80}\n")
    
    all_tables = []
    
    for fold_num in range(1, len(fold_starts)):
        save_path = os.path.join(model_dir, f"metrics_table_fold{fold_num}.txt")
        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}/{len(fold_starts)-1}")
        print(f"{'='*80}")
        
        # Use the actual number of epochs for this fold
        actual_epochs = epochs_per_fold_list[fold_num - 1] if fold_num <= len(epochs_per_fold_list) else epochs_per_fold
        
        # Print to console
        visualize_evaluation_table(metrics_file, save_path, 
                                   fold_number=fold_num, epochs_per_fold=actual_epochs)
        
        # Read the file again to add to all_tables
        with open(save_path, 'r', encoding='utf-8') as f:
            all_tables.append(f.read())
    
    # Save all folds to 1 file
    save_path_all = os.path.join(model_dir, "metrics_table_all_folds.txt")
    with open(save_path_all, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TRAINING METRICS - ALL 5 FOLDS\n")
        f.write("="*80 + "\n\n")
        for i, table in enumerate(all_tables, 1):
            f.write(table)
            if i < len(all_tables):
                f.write("\n\n" + "="*80 + "\n\n")
    
    # Calculate the average of the final metrics of the 5 folds
    print(f"\n{'='*80}")
    print("CALCULATING THE AVERAGE OF THE FINAL METRICS OF THE 5 FOLDS")
    print(f"{'='*80}\n")
    
    # Read the CSV again and calculate the average of the best epoch of each fold
    df_all = pd.read_csv(metrics_file)
    summary_rows = []
    
    for fold_num in range(1, len(fold_starts)):
        start_idx = fold_starts[fold_num - 1]
        end_idx = fold_starts[fold_num]
        fold_df = df_all.iloc[start_idx:end_idx]
        
        if len(fold_df) > 0:
            # Get the best epoch (lowest val_loss)
            best_epoch_idx = fold_df['val_loss'].idxmin()
            best_row = fold_df.loc[best_epoch_idx]
            summary_rows.append({
                'fold': fold_num,
                'epoch': int(best_row['epoch']),
                'train_loss': best_row['train_loss'],
                'val_loss': best_row['val_loss'],
                'val_dice': best_row['val_dice'],
                'val_iou': best_row['val_iou'],
                'val_pixel_acc': best_row['val_pixel_acc'],
                'val_boundary_f1': best_row['val_boundary_f1'],
                'val_avg': best_row['val_avg']
            })
    
    # Calculate the average
    summary_df = pd.DataFrame(summary_rows)
    avg_metrics = {
        'fold': 'AVERAGE',
        'epoch': '-',
        'train_loss': summary_df['train_loss'].mean(),
        'val_loss': summary_df['val_loss'].mean(),
        'val_dice': summary_df['val_dice'].mean(),
        'val_iou': summary_df['val_iou'].mean(),
        'val_pixel_acc': summary_df['val_pixel_acc'].mean(),
        'val_boundary_f1': summary_df['val_boundary_f1'].mean(),
        'val_avg': summary_df['val_avg'].mean()
    }
    
    # Create summary table
    from tabulate import tabulate
    summary_table_data = []
    for row in summary_rows:
        summary_table_data.append([
            f"Fold {row['fold']}",
            row['epoch'],
            f"{row['train_loss']:.4f}",
            f"{row['val_loss']:.4f}",
            f"{row['val_dice']:.4f}",
            f"{row['val_iou']:.4f}",
            f"{row['val_pixel_acc']:.4f}",
            f"{row['val_boundary_f1']:.4f}",
            f"{row['val_avg']:.4f}"
        ])
    
    # Add average row
    summary_table_data.append([
        "AVERAGE",
        " - ",
        f"{avg_metrics['train_loss']:.4f}",
        f"{avg_metrics['val_loss']:.4f}",
        f"{avg_metrics['val_dice']:.4f}",
        f"{avg_metrics['val_iou']:.4f}",
        f"{avg_metrics['val_pixel_acc']:.4f}",
        f"{avg_metrics['val_boundary_f1']:.4f}",
        f"{avg_metrics['val_avg']:.4f}"
    ])
    
    headers = ["Fold", "Best Epoch", "Train Loss", "Val Loss", "Val Dice", "Val IoU", 
               "Val Pixel Acc", "Val Boundary F1", "Val Avg"]
    summary_table = tabulate(summary_table_data, headers=headers, tablefmt="grid", floatfmt=".4f")
    
    # Print to console
    print("\n" + "="*120)
    print("SUMMARY: BEST EPOCH FROM EACH FOLD (Val Loss Lowest)")
    print("="*120)
    print(summary_table)
    
    # Save to file
    summary_path = os.path.join(model_dir, "metrics_summary_5folds.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*120 + "\n")
        f.write("K-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*120 + "\n\n")
        f.write("How it works:\n")
        f.write("- The dataset is divided into 5 folds\n")
        f.write("- Train 50 epochs on EACH fold (not train 5 epochs and then average)\n")
        f.write("- Each epoch calculates: train_loss, val_loss, and metrics (dice, iou, pixel_acc, boundary_f1)\n")
        f.write("- After training, select the BEST epoch (lowest val_loss) from each fold\n")
        f.write("- Calculate the AVERAGE of the metrics of these 5 best epochs\n\n")
        f.write("="*120 + "\n")
        f.write("BEST EPOCH FROM EACH FOLD\n")
        f.write("="*120 + "\n\n")
        f.write(summary_table)
        f.write("\n\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Created metrics table for all {len(fold_starts)-1} folds")
    print(f"✓ Individual files: {model_dir}/metrics_table_fold1.txt ... fold{len(fold_starts)-1}.txt")
    print(f"✓ Consolidated file: {save_path_all}")
    print(f"✓ Summary file (average): {summary_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
