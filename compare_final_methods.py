#!/usr/bin/env python3
"""
Script để so sánh Stage 3 với và không có Mean Teacher
Chạy cả hai phương pháp và so sánh kết quả trên 100 frames
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Thêm src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ExperimentConfig
from src.models.model_factory import get_model, load_checkpoint, save_checkpoint
from src.data.datasets import VideoDataset
from src.utils.augment import create_val_transform
from src.utils.metrics import dice_coefficient, iou_score, pixel_accuracy, boundary_f1_score, avg_score

def load_config(config_path="config.json"):
    """Load configuration"""
    with open(config_path) as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)

def evaluate_model_on_frames(model, frames_dir, device="mps", max_frames=100):
    """Đánh giá model trên unlabeled frames"""
    print(f"\n{'='*60}")
    print(f"Đánh giá model trên {max_frames} frames từ {frames_dir}")
    print(f"{'='*60}")
    
    # Tạo dataset
    val_transform = create_val_transform()
    dataset = VideoDataset(
        image_root=frames_dir,
        transform=val_transform,
        target_size=[256, 256]
    )
    
    # Giới hạn số frames
    total_frames = len(dataset)
    num_eval = min(max_frames, total_frames)
    print(f"Tổng số frames: {total_frames}, Đánh giá trên: {num_eval} frames")
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for i in range(num_eval):
            if i % 10 == 0:
                print(f"  Processing frame {i+1}/{num_eval}...", end='\r')
            
            sample = dataset[i]
            image = sample['image'].unsqueeze(0).to(device)
            
            # Predict
            pred = model(image)
            if isinstance(pred, dict):
                pred = pred.get('out', pred.get('logits', pred))
            
            # Convert to binary mask
            prob = torch.sigmoid(pred).squeeze().cpu().numpy()
            binary_pred = (prob > 0.5).astype(np.float32)
            
            all_preds.append(binary_pred)
    
    print(f"\n  ✓ Hoàn thành đánh giá {num_eval} frames")
    
    # Tính statistics trên predictions
    all_preds = np.array(all_preds)
    
    # Thống kê cơ bản
    total_pixels = all_preds.size
    vessel_pixels = np.sum(all_preds)
    vessel_ratio = vessel_pixels / total_pixels if total_pixels > 0 else 0
    
    stats = {
        'num_frames': num_eval,
        'total_pixels': total_pixels,
        'vessel_pixels': int(vessel_pixels),
        'vessel_ratio': vessel_ratio,
        'mean_prediction': float(np.mean(all_preds)),
        'std_prediction': float(np.std(all_preds)),
        'frames_with_vessels': int(np.sum(np.any(all_preds.reshape(num_eval, -1), axis=1)))
    }
    
    return stats, all_preds

def compare_models(model1_path, model2_path, frames_dir, device="mps", max_frames=100):
    """So sánh hai models"""
    config = load_config()
    
    # Load models
    print(f"\n{'='*60}")
    print("Loading Models...")
    print(f"{'='*60}")
    
    model1 = get_model(config.model)
    model1 = load_checkpoint(model1, model1_path, device=device)
    print(f"✓ Loaded model 1: {model1_path}")
    
    model2 = get_model(config.model)
    model2 = load_checkpoint(model2, model2_path, device=device)
    print(f"✓ Loaded model 2: {model2_path}")
    
    # Evaluate both models
    print(f"\n{'='*60}")
    print("ĐÁNH GIÁ MODEL 1 (Không Mean Teacher)")
    print(f"{'='*60}")
    stats1, preds1 = evaluate_model_on_frames(model1, frames_dir, device, max_frames)
    
    print(f"\n{'='*60}")
    print("ĐÁNH GIÁ MODEL 2 (Có Mean Teacher)")
    print(f"{'='*60}")
    stats2, preds2 = evaluate_model_on_frames(model2, frames_dir, device, max_frames)
    
    # Compare predictions
    print(f"\n{'='*60}")
    print("SO SÁNH PREDICTIONS")
    print(f"{'='*60}")
    
    # Tính agreement giữa hai models
    preds1_flat = preds1.reshape(-1)
    preds2_flat = preds2.reshape(-1)
    
    agreement = np.mean(preds1_flat == preds2_flat)
    disagreement = 1 - agreement
    
    # Chỉ tính trên pixels có prediction khác nhau
    diff_mask = preds1_flat != preds2_flat
    diff_pixels = np.sum(diff_mask)
    diff_ratio = diff_pixels / len(preds1_flat) if len(preds1_flat) > 0 else 0
    
    # Thống kê
    model1_only = np.sum((preds1_flat == 1) & (preds2_flat == 0))
    model2_only = np.sum((preds1_flat == 0) & (preds2_flat == 1))
    both_vessel = np.sum((preds1_flat == 1) & (preds2_flat == 1))
    both_background = np.sum((preds1_flat == 0) & (preds2_flat == 0))
    
    print(f"\n📊 KẾT QUẢ SO SÁNH:")
    print(f"{'─'*60}")
    print(f"\n1. THỐNG KÊ MODEL 1 (Không Mean Teacher):")
    print(f"   - Số frames: {stats1['num_frames']}")
    print(f"   - Frames có vessel: {stats1['frames_with_vessels']} ({stats1['frames_with_vessels']/stats1['num_frames']*100:.1f}%)")
    print(f"   - Vessel pixels: {stats1['vessel_pixels']:,} ({stats1['vessel_ratio']*100:.2f}%)")
    print(f"   - Mean prediction: {stats1['mean_prediction']:.4f}")
    
    print(f"\n2. THỐNG KÊ MODEL 2 (Có Mean Teacher):")
    print(f"   - Số frames: {stats2['num_frames']}")
    print(f"   - Frames có vessel: {stats2['frames_with_vessels']} ({stats2['frames_with_vessels']/stats2['num_frames']*100:.1f}%)")
    print(f"   - Vessel pixels: {stats2['vessel_pixels']:,} ({stats2['vessel_ratio']*100:.2f}%)")
    print(f"   - Mean prediction: {stats2['mean_prediction']:.4f}")
    
    print(f"\n3. SO SÁNH GIỮA HAI MODELS:")
    print(f"   - Agreement: {agreement*100:.2f}%")
    print(f"   - Disagreement: {disagreement*100:.2f}%")
    print(f"   - Pixels khác nhau: {diff_pixels:,} ({diff_ratio*100:.2f}%)")
    print(f"\n4. PHÂN BỐ PREDICTIONS:")
    print(f"   - Cả hai detect vessel: {both_vessel:,} pixels")
    print(f"   - Cả hai detect background: {both_background:,} pixels")
    print(f"   - Chỉ Model 1 detect vessel: {model1_only:,} pixels")
    print(f"   - Chỉ Model 2 detect vessel: {model2_only:,} pixels")
    
    # Tính difference
    vessel_diff = abs(stats1['vessel_ratio'] - stats2['vessel_ratio'])
    print(f"\n5. KHÁC BIỆT:")
    print(f"   - Khác biệt vessel ratio: {vessel_diff*100:.2f}%")
    if stats1['vessel_ratio'] > stats2['vessel_ratio']:
        print(f"   → Model 1 detect nhiều vessel hơn Model 2")
    elif stats2['vessel_ratio'] > stats1['vessel_ratio']:
        print(f"   → Model 2 detect nhiều vessel hơn Model 1")
    else:
        print(f"   → Hai models detect tương đương")
    
    return {
        'model1_stats': stats1,
        'model2_stats': stats2,
        'agreement': agreement,
        'disagreement': disagreement,
        'diff_pixels': int(diff_pixels),
        'diff_ratio': diff_ratio,
        'comparison': {
            'both_vessel': int(both_vessel),
            'both_background': int(both_background),
            'model1_only': int(model1_only),
            'model2_only': int(model2_only)
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="So sánh Stage 3 với và không có Mean Teacher")
    parser.add_argument("--model1", type=str, default="models/final_no_mt.pth",
                       help="Model không dùng Mean Teacher")
    parser.add_argument("--model2", type=str, default="models/final_mt.pth",
                       help="Model có dùng Mean Teacher")
    parser.add_argument("--frames_dir", type=str, default="data/video_frames",
                       help="Thư mục chứa frames để đánh giá")
    parser.add_argument("--max_frames", type=int, default=100,
                       help="Số frames tối đa để đánh giá")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device (mps, cuda, cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model1):
        print(f"❌ Không tìm thấy model 1: {args.model1}")
        print("   Chạy Stage 3 không Mean Teacher trước:")
        print("   python -m src.main final --config config.json")
        sys.exit(1)
    
    if not os.path.exists(args.model2):
        print(f"❌ Không tìm thấy model 2: {args.model2}")
        print("   Chạy Stage 3 có Mean Teacher trước:")
        print("   python -m src.main final --use_mean_teacher --config config.json")
        sys.exit(1)
    
    if not os.path.exists(args.frames_dir):
        print(f"❌ Không tìm thấy thư mục frames: {args.frames_dir}")
        sys.exit(1)
    
    # So sánh
    results = compare_models(
        args.model1, 
        args.model2, 
        args.frames_dir,
        device=args.device,
        max_frames=args.max_frames
    )
    
    # Lưu kết quả
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"models/comparison_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Đã lưu kết quả vào: {results_file}")

