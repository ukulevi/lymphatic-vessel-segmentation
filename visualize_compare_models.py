#!/usr/bin/env python3
"""
Script để visualize và so sánh 2 models:
- final_no_mt.pth (Không Mean Teacher)
- final_mt.pth (Có Mean Teacher)
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Thêm src vào path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import ExperimentConfig
from src.models.model_factory import get_model, load_checkpoint
from src.data.datasets import LabeledVesselDataset, VideoDataset
from src.utils.augment import create_val_transform
from src.utils.metrics import dice_coefficient, iou_score, pixel_accuracy, boundary_f1_score, avg_score

def load_config(config_path="config.json"):
    """Load configuration"""
    with open(config_path) as f:
        config_dict = json.load(f)
    return ExperimentConfig.from_dict(config_dict)

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay mask trên image"""
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    colored_mask = np.zeros_like(image)
    if len(colored_mask.shape) == 2:
        colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_GRAY2RGB)
    
    if len(mask.shape) == 2:
        colored_mask[mask == 1] = color
    else:
        colored_mask[mask > 0.5] = color
    
    overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def visualize_comparison(model1, model1_name, model2, model2_name, dataset, device="mps", num_samples=6):
    """Visualize và so sánh 2 models"""
    model1.eval()
    model2.eval()
    
    # Lấy samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    fig, axes = plt.subplots(num_samples, 7, figsize=(28, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('So Sánh Models: Không Mean Teacher vs Có Mean Teacher', fontsize=16, fontweight='bold', y=0.995)
    
    all_metrics_model1 = {'dice': [], 'iou': [], 'pixel_acc': []}
    all_metrics_model2 = {'dice': [], 'iou': [], 'pixel_acc': []}
    
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            image = sample['image'].unsqueeze(0).to(device)
            mask = sample.get('mask', None)  # Có thể không có mask (unlabeled data)
            
            # Model 1 predictions
            pred1 = model1(image)
            if isinstance(pred1, dict):
                pred1 = pred1.get('out', pred1.get('logits', pred1))
            prob1 = torch.sigmoid(pred1).squeeze().cpu().numpy()
            binary1 = (prob1 > 0.5).astype(np.float32)
            
            # Model 2 predictions
            pred2 = model2(image)
            if isinstance(pred2, dict):
                pred2 = pred2.get('out', pred2.get('logits', pred2))
            prob2 = torch.sigmoid(pred2).squeeze().cpu().numpy()
            binary2 = (prob2 > 0.5).astype(np.float32)
            
            # Convert to numpy
            img_np = image.squeeze().cpu().numpy().transpose(1, 2, 0)
            img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)
            
            # Tính metrics nếu có ground truth
            if mask is not None:
                mask_np = mask.squeeze().cpu().numpy()
            else:
                mask_np = None
            
            if mask_np is not None and mask_np.max() > 0:  # Nếu có ground truth
                mask_binary = (mask_np > 0.5).astype(np.float32)
                
                dice1 = dice_coefficient(binary1[np.newaxis], mask_binary[np.newaxis])
                iou1 = iou_score(binary1[np.newaxis], mask_binary[np.newaxis])
                acc1 = pixel_accuracy(binary1[np.newaxis], mask_binary[np.newaxis])
                
                dice2 = dice_coefficient(binary2[np.newaxis], mask_binary[np.newaxis])
                iou2 = iou_score(binary2[np.newaxis], mask_binary[np.newaxis])
                acc2 = pixel_accuracy(binary2[np.newaxis], mask_binary[np.newaxis])
                
                all_metrics_model1['dice'].append(dice1)
                all_metrics_model1['iou'].append(iou1)
                all_metrics_model1['pixel_acc'].append(acc1)
                
                all_metrics_model2['dice'].append(dice2)
                all_metrics_model2['iou'].append(iou2)
                all_metrics_model2['pixel_acc'].append(acc2)
            else:
                dice1 = iou1 = acc1 = dice2 = iou2 = acc2 = 0.0
            
            # 1. Original Image
            axes[row, 0].imshow(img_np)
            axes[row, 0].set_title('Input Image', fontsize=10, fontweight='bold')
            axes[row, 0].axis('off')
            
            # 2. Ground Truth (nếu có)
            if mask_np is not None and mask_np.max() > 0:
                axes[row, 1].imshow(mask_np, cmap='gray')
                axes[row, 1].set_title('Ground Truth', fontsize=10, fontweight='bold')
                axes[row, 1].axis('off')
            else:
                axes[row, 1].text(0.5, 0.5, 'No GT\n(Unlabeled)', ha='center', va='center', fontsize=12)
                axes[row, 1].axis('off')
            
            # 3. Model 1 Prediction
            axes[row, 2].imshow(binary1, cmap='gray')
            title1 = f'Model 1: {model1_name}\nDice: {dice1:.3f}, IoU: {iou1:.3f}'
            axes[row, 2].set_title(title1, fontsize=9)
            axes[row, 2].axis('off')
            
            # 4. Model 2 Prediction
            axes[row, 3].imshow(binary2, cmap='gray')
            title2 = f'Model 2: {model2_name}\nDice: {dice2:.3f}, IoU: {iou2:.3f}'
            axes[row, 3].set_title(title2, fontsize=9)
            axes[row, 3].axis('off')
            
            # 5. GT Overlay
            if mask_np is not None and mask_np.max() > 0:
                gt_overlay = overlay_mask((img_np * 255).astype(np.uint8), 
                                         (mask_np * 255).astype(np.uint8), 
                                         color=(0, 255, 0), alpha=0.5) / 255.0
                axes[row, 4].imshow(gt_overlay)
                axes[row, 4].set_title('GT Overlay', fontsize=10)
                axes[row, 4].axis('off')
            else:
                axes[row, 4].imshow(img_np)
                axes[row, 4].set_title('No GT', fontsize=10)
                axes[row, 4].axis('off')
            
            # 6. Model 1 Overlay
            pred1_overlay = overlay_mask((img_np * 255).astype(np.uint8), 
                                        (binary1 * 255).astype(np.uint8), 
                                        color=(255, 0, 0), alpha=0.5) / 255.0
            axes[row, 5].imshow(pred1_overlay)
            axes[row, 5].set_title(f'{model1_name} Overlay', fontsize=10)
            axes[row, 5].axis('off')
            
            # 7. Model 2 Overlay
            pred2_overlay = overlay_mask((img_np * 255).astype(np.uint8), 
                                        (binary2 * 255).astype(np.uint8), 
                                        color=(0, 0, 255), alpha=0.5) / 255.0
            axes[row, 6].imshow(pred2_overlay)
            axes[row, 6].set_title(f'{model2_name} Overlay', fontsize=10)
            axes[row, 6].axis('off')
    
    # Tính average metrics
    if len(all_metrics_model1['dice']) > 0:
        avg1 = {
            'dice': np.mean(all_metrics_model1['dice']),
            'iou': np.mean(all_metrics_model1['iou']),
            'pixel_acc': np.mean(all_metrics_model1['pixel_acc'])
        }
        avg2 = {
            'dice': np.mean(all_metrics_model2['dice']),
            'iou': np.mean(all_metrics_model2['iou']),
            'pixel_acc': np.mean(all_metrics_model2['pixel_acc'])
        }
        
        # Thêm text box với average metrics
        fig.text(0.5, 0.02, 
                f"Average Metrics - {model1_name}: Dice={avg1['dice']:.3f}, IoU={avg1['iou']:.3f}, Acc={avg1['pixel_acc']:.3f} | "
                f"{model2_name}: Dice={avg2['dice']:.3f}, IoU={avg2['iou']:.3f}, Acc={avg2['pixel_acc']:.3f}",
                ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    return fig, avg1 if len(all_metrics_model1['dice']) > 0 else None, avg2 if len(all_metrics_model2['dice']) > 0 else None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize và so sánh 2 models")
    parser.add_argument("--model1", type=str, default="models/final_no_mt.pth",
                       help="Model 1 (không Mean Teacher)")
    parser.add_argument("--model2", type=str, default="models/final_mt.pth",
                       help="Model 2 (có Mean Teacher)")
    parser.add_argument("--dataset", type=str, default="labeled",
                       choices=["labeled", "video_frames"],
                       help="Dataset để visualize (labeled hoặc video_frames)")
    parser.add_argument("--num_samples", type=int, default=6,
                       help="Số samples để visualize")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device (mps, cuda, cpu)")
    parser.add_argument("--output", type=str, default="models/comparison_visualization.png",
                       help="Output file path")
    
    args = parser.parse_args()
    
    # Kiểm tra models
    if not os.path.exists(args.model1):
        print(f"❌ Không tìm thấy model 1: {args.model1}")
        sys.exit(1)
    
    if not os.path.exists(args.model2):
        print(f"❌ Không tìm thấy model 2: {args.model2}")
        sys.exit(1)
    
    # Load config
    config = load_config()
    
    # Load models
    print(f"\n{'='*60}")
    print("Loading Models...")
    print(f"{'='*60}")
    
    model1 = get_model(config.model)
    model1 = load_checkpoint(model1, args.model1, device=args.device)
    print(f"✓ Loaded model 1: {args.model1}")
    
    model2 = get_model(config.model)
    model2 = load_checkpoint(model2, args.model2, device=args.device)
    print(f"✓ Loaded model 2: {args.model2}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Loading Dataset: {args.dataset}")
    print(f"{'='*60}")
    
    val_transform = create_val_transform()
    
    if args.dataset == "labeled":
        dataset = LabeledVesselDataset(
            image_root=config.paths.labeled_dir,
            mask_dir=config.paths.labeled_masks_dir,
            transform=val_transform,
            target_size=config.data.image_size
        )
        model1_name = "Không Mean Teacher"
        model2_name = "Có Mean Teacher"
    else:
        dataset = VideoDataset(
            image_root="data/video_frames",
            transform=val_transform,
            target_size=config.data.image_size
        )
        model1_name = "Không MT (Unlabeled)"
        model2_name = "Mean Teacher (Unlabeled)"
    
    print(f"✓ Loaded {len(dataset)} samples")
    
    # Visualize
    print(f"\n{'='*60}")
    print(f"Visualizing {args.num_samples} samples...")
    print(f"{'='*60}")
    
    fig, avg1, avg2 = visualize_comparison(
        model1, model1_name,
        model2, model2_name,
        dataset,
        device=args.device,
        num_samples=args.num_samples
    )
    
    # Save
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {args.output}")
    
    # Print summary
    if avg1 and avg2:
        print(f"\n{'='*60}")
        print("TỔNG KẾT METRICS:")
        print(f"{'='*60}")
        print(f"\n{model1_name}:")
        print(f"  Dice: {avg1['dice']:.4f}")
        print(f"  IoU:  {avg1['iou']:.4f}")
        print(f"  Pixel Acc: {avg1['pixel_acc']:.4f}")
        
        print(f"\n{model2_name}:")
        print(f"  Dice: {avg2['dice']:.4f}")
        print(f"  IoU:  {avg2['iou']:.4f}")
        print(f"  Pixel Acc: {avg2['pixel_acc']:.4f}")
        
        print(f"\n{'='*60}")
        print("KHÁC BIỆT:")
        print(f"{'='*60}")
        print(f"  Dice: {avg2['dice'] - avg1['dice']:+.4f} ({'+' if avg2['dice'] > avg1['dice'] else ''}{((avg2['dice']/avg1['dice'] - 1) * 100):.2f}%)")
        print(f"  IoU:  {avg2['iou'] - avg1['iou']:+.4f} ({'+' if avg2['iou'] > avg1['iou'] else ''}{((avg2['iou']/avg1['iou'] - 1) * 100):.2f}%)")
        print(f"  Pixel Acc: {avg2['pixel_acc'] - avg1['pixel_acc']:+.4f}")
        
        if avg2['dice'] > avg1['dice']:
            print(f"\n🏆 {model2_name} tốt hơn về Dice và IoU!")
        elif avg1['dice'] > avg2['dice']:
            print(f"\n🏆 {model1_name} tốt hơn về Dice và IoU!")
        else:
            print(f"\n🤝 Hai models tương đương!")

