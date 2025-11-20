import os
import sys
import json
from src.visualization import plot_loss_curve, plot_training_curves_from_csv

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
                elif 'final_mt' in model_path.lower():
                    return 'final_mt'
                elif 'final_no_mt' in model_path.lower():
                    return 'final_no_mt'
                elif 'final' in model_path.lower(): # fallback
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
            # Get a descriptive name like "Human/20251102_175217"
            descriptive_name = os.path.relpath(root, logs_dir)
            log_folders.append({
                'folder': descriptive_name.replace("\\", "/"), # Use forward slashes
                'path': root,
                'metrics': metrics_file,
                'type': model_type or 'unknown'
            })
    
    # Sort by the full path, which includes the timestamp, to get latest first
    log_folders.sort(key=lambda x: x['path'], reverse=True)
    return log_folders

def main():
    print("="*70)
    print("📊 PLOT TRAINING CURVES - BASELINE & FINAL")
    print("="*70)
    
    # Finding all logs
    all_logs = find_all_logs()
    
    if not all_logs:
        print("❌ No log folders with metrics.csv found")
        metrics_file = input("Enter the path to metrics.csv: ").strip()
        if not os.path.exists(metrics_file):
            print(f"❌ File not found: {metrics_file}")
            return
        log_type = 'unknown'
        model_type_folder = ""
    else:
        # Get available model types from the found logs
        available_types = sorted(list(set([log['folder'].split('/')[0] for log in all_logs])))

        # Ask user to filter by type
        if len(available_types) > 1:
            print(f"\nFound logs for model types: {', '.join(available_types)}")
            type_choice = input(f"Filter by type (e.g., {available_types[0]}), or press Enter to show all: ").strip()

            if type_choice and type_choice in available_types:
                # Filter logs for the chosen type
                all_logs = [log for log in all_logs if log['folder'].startswith(type_choice)]
                print(f"\nShowing latest logs for type: {type_choice}")
            else:
                print("\nShowing latest logs for all types.")

        # Display list of logs
        print(f"\n📁 Found {len(all_logs)} log folder(s):")
        print("-" * 70)
        for i, log_info in enumerate(all_logs[:10], 1):  # Showing only the latest 10
            model_type_icon = "🔵" if log_info['type'] == 'baseline' else "🟢" if log_info['type'].startswith('final') else "⚪"
            print(f"  {i}. {model_type_icon} {log_info['folder']} ({log_info['type']})")
        
        if len(all_logs) > 10:
            print(f"  ... and {len(all_logs) - 10} other log(s)")
        
        # Allow log selection
        print("\n" + "-" * 70)
        choice = input("Select a log (1-{}), or press Enter to use the latest log: ".format(min(len(all_logs), 10))).strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(all_logs):
            selected_log = all_logs[int(choice) - 1]
        else:
            selected_log = all_logs[0]  # Latest log
        
        metrics_file = selected_log['metrics']
        log_type = selected_log['type']
        model_type_folder = selected_log['folder'].split('/')[0]
        print(f"\n✓ Selected: {selected_log['folder']} ({log_type})")
    
    print(f"📄 Reading: {metrics_file}")
    
    # Read CSV
    import pandas as pd
    df = pd.read_csv(metrics_file)
    total_rows = len(df)

    # Check if the dataframe is empty
    if total_rows == 0:
        print(f"\n❌ Error: '{metrics_file}' is empty or contains no data.")
        print("Cannot generate plots. Please check the training log.")
        return
    
    print(f"📊 Total epochs: {total_rows}")
    
    # Define model directory
    model_dir = os.path.join("models", model_type_folder) if model_type_folder else "models"
    os.makedirs(model_dir, exist_ok=True)

    # Processing by model type
    if log_type == 'baseline':
        # Baseline has k-fold CV
        epoch_resets = sum(1 for i in range(1, len(df)) if df.iloc[i]['epoch'] < df.iloc[i-1]['epoch'])
        num_folds = epoch_resets + 1 if epoch_resets > 0 else 1
        
        print(f"📈 Detected {num_folds} fold(s) (K-Fold CV)")
        
        choice = input("\nWhat do you want to see:\n  1. All folds\n  2. A specific fold (1-{0})\nSelect (1 or 2): ".format(num_folds)).strip()
        
        if choice == "2":
            fold_num = input(f"Select fold (1-{num_folds}): ").strip()
            try:
                fold_num = int(fold_num)
                if fold_num < 1 or fold_num > num_folds:
                    print(f"Fold must be between 1-{num_folds}, showing fold 1")
                    fold_num = 1
            except:
                print("Invalid number, showing fold 1")
                fold_num = 1
            
            epochs_per_fold = total_rows // num_folds if num_folds > 0 else None
            save_path = os.path.join(model_dir, f"baseline_training_loss_fold{fold_num}.png")
            fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True, 
                                 fold_number=fold_num, epochs_per_fold=epochs_per_fold)
            print(f"✓ Loss curve for Fold {fold_num} has been saved to: {save_path}")
        else:
            save_path = os.path.join(model_dir, "baseline_training_loss_all_folds.png")
            fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
            print(f"✓ Loss curve (all folds) has been saved to: {save_path}")
    
    elif log_type.startswith('final'):
        # Final model does not have k-fold, plotting directly
        save_path = os.path.join(model_dir, f"{log_type}_training_loss.png")
        fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
        print(f"✓ Loss curve for {log_type.replace('_', ' ').upper()} Model has been saved to: {save_path}")
    
    else:
        # Unknown type - plotting directly
        save_path = os.path.join(model_dir, "training_loss_curve.png")
        fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
        print(f"✓ Loss curve has been saved to: {save_path}")
    
    # Display plot
    print("\nDisplaying plot...")
    import matplotlib.pyplot as plt
    plt.show()
    
    # Option: Plot detailed curves
    choice = input("\nDo you want to plot detailed curves with all metrics? (y/n): ").strip().lower()
    if choice == 'y':
        save_path_detailed = os.path.join(model_dir, f"{log_type}_training_curves_detailed.png")
        fig_detailed = plot_training_curves_from_csv(metrics_file, save_path=save_path_detailed)
        print(f"✓ Detailed plot has been saved to: {save_path_detailed}")
        plt.show()

if __name__ == "__main__":
    main()