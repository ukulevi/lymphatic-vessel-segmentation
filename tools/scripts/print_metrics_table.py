import os
import sys
import json
import pandas as pd
from tabulate import tabulate
from src.visualization import visualize_evaluation_table

def detect_model_type(log_folder):
    """Detect if the log is for a baseline or final model based on models.jsonl"""
    models_file = os.path.join(log_folder, "models.jsonl")
    if not os.path.exists(models_file):
        return None
    
    try:
        with open(models_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                model_path = data.get('path', '')
                if 'baseline' in model_path.lower():
                    return 'baseline'
                elif 'final' in model_path.lower():
                    return 'final'
    except:
        pass
    return None

def find_all_logs():
    """Find all logs that contain metrics.csv, searching recursively."""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return []
    
    log_folders = []
    for root, dirs, files in os.walk(logs_dir):
        if "metrics.csv" in files:
            metrics_file = os.path.join(root, "metrics.csv")
            model_type = detect_model_type(root)
            descriptive_name = os.path.relpath(root, logs_dir)
            log_folders.append({
                'folder': descriptive_name.replace("\\\\", "/"),
                'path': root,
                'metrics': metrics_file,
                'type': model_type or 'unknown'
            })
    
    log_folders.sort(key=lambda x: x['path'], reverse=True)
    return log_folders

def main():
    print("="*70)
    print("📄 PRINT METRICS TABLE FROM LOGS")
    print("="*70)

    all_logs = find_all_logs()
    
    if not all_logs:
        print("❌ No log folders with metrics.csv found")
        return

    available_types = sorted(list(set([log['folder'].split('/')[0] for log in all_logs])))

    if len(available_types) > 1:
        print(f"\nFound logs for model types: {', '.join(available_types)}")
        type_choice = input(f"Filter by type (e.g., {available_types[0]}), or press Enter to show all: ").strip()

        if type_choice and type_choice in available_types:
            all_logs = [log for log in all_logs if log['folder'].startswith(type_choice)]
            print(f"\nShowing latest logs for type: {type_choice}")
        else:
            print("\nShowing latest logs for all types.")

    print(f"\n📁 Found {len(all_logs)} log folder(s):")
    print("-" * 70)
    for i, log_info in enumerate(all_logs[:15], 1):
        model_type_icon = "🔵" if log_info['type'] == 'baseline' else "🟢" if log_info['type'].startswith('final') else "⚪"
        print(f"  {i}. {model_type_icon} {log_info['folder']} ({log_info['type']})")
    
    if len(all_logs) > 15:
        print(f"  ... and {len(all_logs) - 15} other log(s)")
    
    print("\n" + "-" * 70)
    choice = input("Select a log to process (1-{}), or press Enter for latest: ".format(min(len(all_logs), 15))).strip()
    
    if choice.isdigit() and 1 <= int(choice) <= len(all_logs):
        selected_log = all_logs[int(choice) - 1]
    else:
        selected_log = all_logs[0]
    
    metrics_file = selected_log['metrics']
    model_type_folder = selected_log['folder'].split('/')[0]
    timestamp = selected_log['folder'].split('/')[-1]
    print(f"\n✓ Selected: {selected_log['folder']} ({selected_log['type']})")

    model_dir = os.path.join("models", model_type_folder) if model_type_folder else "models"
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"Reading metrics from: {metrics_file}")
    
    df = pd.read_csv(metrics_file)
    if df.empty:
        print(f"❌ Error: '{metrics_file}' is empty. Cannot generate summary.")
        return

    # Check if this is a k-fold log
    is_kfold = 'fold' in df.columns or (df['epoch'].diff() < 0).any()

    if not is_kfold:
        print("\n(i) Single run log detected. Generating a simple evaluation summary.")
        output_filename = f"evaluation_summary_{timestamp}.txt"
        output_path = os.path.join(model_dir, output_filename)
        visualize_evaluation_table(metrics_file, output_path)
        return

    print("\n(i) K-fold log detected. Running full k-fold analysis.")
    total_rows = len(df)
    
    fold_starts = [0] + df[df['epoch'].diff() < 0].index.tolist()
    fold_starts.append(len(df))
    
    epochs_per_fold_list = [fold_starts[i+1] - fold_starts[i] for i in range(len(fold_starts)-1)]
    
    print(f"\nTotal metric rows: {total_rows}")
    print(f"Epochs per fold: {epochs_per_fold_list}")
    
    all_tables = []
    summary_rows = []

    for fold_num in range(1, len(fold_starts)):
        start_idx = fold_starts[fold_num - 1]
        end_idx = fold_starts[fold_num]
        fold_df = df.iloc[start_idx:end_idx].copy()
        
        if len(fold_df) > 0:
            # Generate table for this fold
            save_path = os.path.join(model_dir, f"metrics_table_{timestamp}_fold{fold_num}.txt")
            print(f"\n--- FOLD {fold_num}/{len(fold_starts)-1} ---")
            visualize_evaluation_table(fold_df, save_path)
            
            with open(save_path, 'r', encoding='utf-8') as f:
                all_tables.append(f.read())

            # Find best epoch for summary
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
                'val_avg': best_row.get('val_avg', 0)
            })

    # Save all folds to one file
    save_path_all = os.path.join(model_dir, f"metrics_table_all_folds_{timestamp}.txt")
    with open(save_path_all, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*80 + f"\nTRAINING METRICS - ALL {len(fold_starts)-1} FOLDS\n" + "="*80 + "\n\n")
        f.write("\n\n" + "="*80 + "\n\n".join(all_tables))

    # Create and save summary table
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        avg_metrics = summary_df.mean(numeric_only=True).to_dict()
        avg_metrics['fold'] = 'AVERAGE'
        
        summary_df_display = summary_df.round(4)
        avg_row = pd.DataFrame([avg_metrics])
        summary_df_display = pd.concat([summary_df_display, avg_row], ignore_index=True)
        
        summary_table = tabulate(summary_df_display, headers='keys', tablefmt="grid", showindex=False, floatfmt=".4f")

        print("\n" + "="*120)
        print("SUMMARY: BEST EPOCH FROM EACH FOLD (by lowest val_loss)")
        print("="*120)
        print(summary_table)

        summary_path = os.path.join(model_dir, f"metrics_summary_5folds_{timestamp}.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("K-FOLD CROSS-VALIDATION SUMMARY\n" + "="*120 + "\n\n")
            f.write(summary_table)
        
        print(f"\n✓ Summary saved to: {summary_path}")

    print(f"\n{'='*80}")
    print(f"✓ Created metrics tables for all {len(fold_starts)-1} folds.")
    print(f"✓ Consolidated file: {save_path_all}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()