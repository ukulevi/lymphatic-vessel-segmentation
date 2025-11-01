"""
Script để in ra bảng metrics dạng text từ metrics.csv
KHÔNG CẦN TRAIN LẠI, chỉ cần đọc file CSV đã có
"""
import os
import sys
import pandas as pd
from src.visualization import visualize_evaluation_table

def main():
    # Tìm metrics.csv mới nhất
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        print(f"Không tìm thấy thư mục {logs_dir}")
        return
    
    # Lấy folder mới nhất
    log_folders = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f))]
    if not log_folders:
        print("Không tìm thấy log folder nào")
        return
    
    # Sắp xếp theo tên (timestamp)
    log_folders.sort(reverse=True)
    latest_folder = log_folders[0]
    metrics_file = os.path.join(logs_dir, latest_folder, "metrics.csv")
    
    if not os.path.exists(metrics_file):
        print(f"Không tìm thấy metrics.csv trong {latest_folder}")
        # Cho phép user nhập path
        metrics_file = input("Nhập đường dẫn đến metrics.csv: ").strip()
        if not os.path.exists(metrics_file):
            print(f"File không tồn tại: {metrics_file}")
            return
    
    print(f"Đang đọc metrics từ: {metrics_file}")
    
    # Đọc CSV để tính số epochs mỗi fold
    df = pd.read_csv(metrics_file)
    total_rows = len(df)
    
    # Tự động phát hiện số epochs mỗi fold (khi epoch reset về 1)
    fold_starts = [0]
    for i in range(1, len(df)):
        if df.iloc[i]['epoch'] < df.iloc[i-1]['epoch']:
            fold_starts.append(i)
    fold_starts.append(len(df))
    
    epochs_per_fold_list = [fold_starts[i+1] - fold_starts[i] for i in range(len(fold_starts)-1)]
    epochs_per_fold = epochs_per_fold_list[0] if epochs_per_fold_list else None  # Giữ để tương thích
    
    print(f"\nTổng số dòng metrics: {total_rows}")
    print(f"Số epochs mỗi fold: {epochs_per_fold_list}")
    
    # In ra tất cả 5 folds
    print(f"\n{'='*80}")
    print("IN RA BẢNG METRICS CHO TẤT CẢ 5 FOLDS")
    print(f"{'='*80}\n")
    
    all_tables = []
    
    for fold_num in range(1, len(fold_starts)):
        save_path = f"models/metrics_table_fold{fold_num}.txt"
        print(f"\n{'='*80}")
        print(f"FOLD {fold_num}/{len(fold_starts)-1}")
        print(f"{'='*80}")
        
        # Sử dụng số epochs thực tế của fold này
        actual_epochs = epochs_per_fold_list[fold_num - 1] if fold_num <= len(epochs_per_fold_list) else epochs_per_fold
        
        # In ra console
        visualize_evaluation_table(metrics_file, save_path, 
                                   fold_number=fold_num, epochs_per_fold=actual_epochs)
        
        # Đọc lại file để thêm vào all_tables
        with open(save_path, 'r', encoding='utf-8') as f:
            all_tables.append(f.read())
    
    # Lưu tất cả folds vào 1 file
    save_path_all = "models/metrics_table_all_folds.txt"
    with open(save_path_all, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TRAINING METRICS - ALL 5 FOLDS\n")
        f.write("="*80 + "\n\n")
        for i, table in enumerate(all_tables, 1):
            f.write(table)
            if i < len(all_tables):
                f.write("\n\n" + "="*80 + "\n\n")
    
    # Tính trung bình metrics cuối cùng của 5 folds
    print(f"\n{'='*80}")
    print("TÍNH TRUNG BÌNH METRICS CUỐI CÙNG CỦA 5 FOLDS")
    print(f"{'='*80}\n")
    
    # Đọc lại CSV và tính trung bình best epoch của mỗi fold
    df_all = pd.read_csv(metrics_file)
    summary_rows = []
    
    for fold_num in range(1, len(fold_starts)):
        start_idx = fold_starts[fold_num - 1]
        end_idx = fold_starts[fold_num]
        fold_df = df_all.iloc[start_idx:end_idx]
        
        if len(fold_df) > 0:
            # Lấy epoch tốt nhất (val_loss thấp nhất)
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
    
    # Tính trung bình
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
    
    # Tạo bảng summary
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
    
    # Thêm dòng trung bình
    summary_table_data.append([
        "AVERAGE",
        "-",
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
    
    # In ra console
    print("\n" + "="*120)
    print("SUMMARY: BEST EPOCH FROM EACH FOLD (Val Loss Lowest)")
    print("="*120)
    print(summary_table)
    
    # Lưu vào file
    summary_path = "models/metrics_summary_5folds.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n" + "="*120 + "\n")
        f.write("K-FOLD CROSS-VALIDATION SUMMARY\n")
        f.write("="*120 + "\n\n")
        f.write("Cách hoạt động:\n")
        f.write("- Dataset được chia thành 5 folds\n")
        f.write("- Train 50 epochs trên MỖI fold (không phải train 5 epochs rồi tính trung bình)\n")
        f.write("- Mỗi epoch tính: train_loss, val_loss, và các metrics (dice, iou, pixel_acc, boundary_f1)\n")
        f.write("- Sau khi train xong, chọn epoch TỐT NHẤT (val_loss thấp nhất) từ mỗi fold\n")
        f.write("- Tính TRUNG BÌNH metrics của 5 epochs tốt nhất này\n\n")
        f.write("="*120 + "\n")
        f.write("BEST EPOCH FROM EACH FOLD\n")
        f.write("="*120 + "\n\n")
        f.write(summary_table)
        f.write("\n\n")
    
    print(f"\n{'='*80}")
    print(f"✓ Đã tạo bảng metrics cho tất cả {len(fold_starts)-1} folds")
    print(f"✓ Các file riêng lẻ: models/metrics_table_fold1.txt ... fold{len(fold_starts)-1}.txt")
    print(f"✓ File tổng hợp: {save_path_all}")
    print(f"✓ File tóm tắt (trung bình): {summary_path}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

