"""
Script để vẽ plot training curves từ metrics.csv
Hỗ trợ cả BASELINE và FINAL model
KHÔNG CẦN TRAIN LẠI, chỉ cần đọc file CSV đã có
"""
import os
import sys
import json
from src.visualization import plot_loss_curve, plot_training_curves_from_csv

def detect_model_type(log_folder):
    """Phát hiện log là baseline hay final dựa vào models.jsonl"""
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
    """Tìm tất cả logs có metrics.csv"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        return []
    
    log_folders = []
    for f in os.listdir(logs_dir):
        folder_path = os.path.join(logs_dir, f)
        if os.path.isdir(folder_path):
            metrics_file = os.path.join(folder_path, "metrics.csv")
            if os.path.exists(metrics_file):
                model_type = detect_model_type(folder_path)
                log_folders.append({
                    'folder': f,
                    'path': folder_path,
                    'metrics': metrics_file,
                    'type': model_type or 'unknown'
                })
    
    # Sắp xếp theo timestamp (tên folder)
    log_folders.sort(key=lambda x: x['folder'], reverse=True)
    return log_folders

def main():
    print("="*70)
    print("📊 PLOT TRAINING CURVES - BASELINE & FINAL")
    print("="*70)
    
    # Tìm tất cả logs
    all_logs = find_all_logs()
    
    if not all_logs:
        print("❌ Không tìm thấy log folder nào có metrics.csv")
        metrics_file = input("Nhập đường dẫn đến metrics.csv: ").strip()
        if not os.path.exists(metrics_file):
            print(f"❌ File không tồn tại: {metrics_file}")
            return
        log_type = 'unknown'
    else:
        # Hiển thị danh sách logs
        print(f"\n📁 Tìm thấy {len(all_logs)} log folder(s):")
        print("-" * 70)
        for i, log_info in enumerate(all_logs[:10], 1):  # Chỉ hiển thị 10 mới nhất
            model_type_icon = "🔵" if log_info['type'] == 'baseline' else "🟢" if log_info['type'] == 'final' else "⚪"
            print(f"  {i}. {model_type_icon} {log_info['folder']} ({log_info['type']})")
        
        if len(all_logs) > 10:
            print(f"  ... và {len(all_logs) - 10} log(s) khác")
        
        # Cho phép chọn log
        print("\n" + "-" * 70)
        choice = input("Chọn log (1-{}), hoặc Enter để dùng log mới nhất: ".format(min(len(all_logs), 10))).strip()
        
        if choice.isdigit() and 1 <= int(choice) <= len(all_logs):
            selected_log = all_logs[int(choice) - 1]
        else:
            selected_log = all_logs[0]  # Log mới nhất
        
        metrics_file = selected_log['metrics']
        log_type = selected_log['type']
        print(f"\n✓ Đã chọn: {selected_log['folder']} ({log_type})")
    
    print(f"📄 Đang đọc: {metrics_file}")
    
    # Đọc CSV
    import pandas as pd
    df = pd.read_csv(metrics_file)
    total_rows = len(df)
    
    print(f"📊 Tổng số epochs: {total_rows}")
    
    # Xử lý theo loại model
    if log_type == 'baseline':
        # Baseline có k-fold CV
        epoch_resets = sum(1 for i in range(1, len(df)) if df.iloc[i]['epoch'] < df.iloc[i-1]['epoch'])
        num_folds = epoch_resets + 1 if epoch_resets > 0 else 1
        
        print(f"📈 Phát hiện {num_folds} fold(s) (K-Fold CV)")
        
        choice = input("\nBạn muốn xem:\n  1. Tất cả folds\n  2. Một fold cụ thể (1-{})\nChọn (1 hoặc 2): ".format(num_folds)).strip()
        
        if choice == "2":
            fold_num = input(f"Chọn fold (1-{num_folds}): ").strip()
            try:
                fold_num = int(fold_num)
                if fold_num < 1 or fold_num > num_folds:
                    print(f"Fold phải từ 1-{num_folds}, sẽ hiển thị fold 1")
                    fold_num = 1
            except:
                print("Số không hợp lệ, sẽ hiển thị fold 1")
                fold_num = 1
            
            epochs_per_fold = total_rows // num_folds if num_folds > 0 else None
            save_path = os.path.join("models", f"baseline_training_loss_fold{fold_num}.png")
            fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True, 
                                 fold_number=fold_num, epochs_per_fold=epochs_per_fold)
            print(f"✓ Loss curve của Fold {fold_num} đã được lưu tại: {save_path}")
        else:
            save_path = os.path.join("models", "baseline_training_loss_all_folds.png")
            fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
            print(f"✓ Loss curve (tất cả folds) đã được lưu tại: {save_path}")
    
    elif log_type == 'final':
        # Final model không có k-fold, chỉ plot trực tiếp
        save_path = os.path.join("models", "final_training_loss.png")
        fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
        print(f"✓ Loss curve của Final Model đã được lưu tại: {save_path}")
    
    else:
        # Unknown type - plot trực tiếp
        save_path = os.path.join("models", "training_loss_curve.png")
        fig = plot_loss_curve(metrics_file, save_path=save_path, show_val=True)
        print(f"✓ Loss curve đã được lưu tại: {save_path}")
    
    os.makedirs("models", exist_ok=True)
    
    # Hiển thị plot
    print("\nĐang hiển thị plot...")
    import matplotlib.pyplot as plt
    plt.show()
    
    # Tùy chọn: Vẽ plot chi tiết
    choice = input("\nBạn có muốn vẽ plot chi tiết với tất cả metrics? (y/n): ").strip().lower()
    if choice == 'y':
        save_path_detailed = os.path.join("models", "training_curves_detailed.png")
        fig_detailed = plot_training_curves_from_csv(metrics_file, save_path=save_path_detailed)
        print(f"✓ Plot chi tiết đã được lưu tại: {save_path_detailed}")
        plt.show()

if __name__ == "__main__":
    main()

