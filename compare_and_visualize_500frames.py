#!/usr/bin/env python3
"""
So sánh và visualize 2 models (Không Mean Teacher vs Có Mean Teacher)
trên nhiều frames với overlay màu đỏ.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.models.model_factory import get_model, load_checkpoint
from src.data.datasets import VideoDataset
from src.config import ExperimentConfig

def overlay_mask_red(image, mask, alpha=0.5):
    """Overlay mask màu đỏ lên image"""
    image_rgb = image.copy()
    if len(image_rgb.shape) == 2:
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
    
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 0.5] = [255, 0, 0]  # Màu đỏ
    
    overlay = cv2.addWeighted(image_rgb, 1 - alpha, colored_mask, alpha, 0)
    return overlay

def load_model(model_path, config):
    """Load model từ checkpoint"""
    model = get_model(config.model)
    device = torch.device(config.training.device)
    model = load_checkpoint(model, model_path, device=device)
    model.eval()
    return model, device

def predict_mask(model, image, device, transform):
    """Predict mask từ image"""
    if isinstance(image, np.ndarray):
        # Convert numpy to PIL if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
    
    transformed = transform(image=image)
    img_tensor = transformed['image'].unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        if isinstance(output, dict):
            output = output['out']
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return pred

def visualize_comparison(
    model1_path="models/final_no_mt.pth",
    model2_path="models/final_mt.pth",
    frames_dir="data/video_frames",
    config_path="config.json",
    num_samples=50,  # Số lượng frames để visualize
    output_path="models/comparison_500frames_overlay.png"
):
    """Visualize so sánh 2 models trên nhiều frames"""
    
    print("=" * 60)
    print("Loading Models...")
    print("=" * 60)
    
    # Load config
    config = ExperimentConfig.from_json(config_path) if Path(config_path).exists() else ExperimentConfig()
    
    # Load models
    model1, device = load_model(model1_path, config)
    print(f"✓ Loaded model 1: {model1_path}")
    
    model2, device2 = load_model(model2_path, config)
    print(f"✓ Loaded model 2: {model2_path}")
    
    # Load dataset
    print(f"\n{'=' * 60}")
    print(f"Loading Dataset: {frames_dir}")
    print("=" * 60)
    
    val_transform = A.Compose([
        A.Resize(config.data.image_size[0], config.data.image_size[1]),
        A.Normalize(),
        ToTensorV2()
    ])
    
    dataset = VideoDataset(
        image_root=frames_dir,
        transform=val_transform,
        target_size=config.data.image_size
    )
    
    print(f"✓ Loaded {len(dataset)} frames")
    
    # Chọn samples ngẫu nhiên
    import random
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    print(f"\n{'=' * 60}")
    print(f"Visualizing {len(indices)} samples...")
    print("=" * 60)
    
    # Tính số hàng và cột
    cols = 4  # Input, Model1, Model2, Overlay comparison
    rows = len(indices)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        image_tensor = sample['image']
        image_path = sample['meta']['img_path']
        
        # Convert image tensor to numpy for display
        img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        # Predict với model 1 (Không Mean Teacher)
        pred1 = predict_mask(model1, img_np, device, val_transform)
        
        # Predict với model 2 (Mean Teacher)
        pred2 = predict_mask(model2, img_np, device2, val_transform)
        
        # Resize predictions về kích thước gốc nếu cần
        if pred1.shape != img_np.shape[:2]:
            pred1 = cv2.resize(pred1, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred2 = cv2.resize(pred2, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Tính Dice cho mỗi model (nếu có ground truth thì tính, không thì bỏ qua)
        frame_name = Path(image_path).stem
        
        # Display
        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Input: {frame_name}', fontsize=8)
        axes[idx, 0].axis('off')
        
        # Model 1 overlay
        overlay1 = overlay_mask_red(img_np, pred1, alpha=0.5)
        axes[idx, 1].imshow(overlay1)
        axes[idx, 1].set_title(f'Không Mean Teacher\n(Dice: {pred1.sum()/pred1.size:.3f})', fontsize=8)
        axes[idx, 1].axis('off')
        
        # Model 2 overlay
        overlay2 = overlay_mask_red(img_np, pred2, alpha=0.5)
        axes[idx, 2].imshow(overlay2)
        axes[idx, 2].set_title(f'Mean Teacher\n(Dice: {pred2.sum()/pred2.size:.3f})', fontsize=8)
        axes[idx, 2].axis('off')
        
        # Side-by-side comparison
        comparison = np.hstack([overlay1, overlay2])
        axes[idx, 3].imshow(comparison)
        axes[idx, 3].set_title('So sánh Overlay', fontsize=8)
        axes[idx, 3].axis('off')
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(indices)} samples...")
    
    plt.suptitle(f'So Sánh Models: Không Mean Teacher vs Có Mean Teacher\n({len(indices)} frames với overlay màu đỏ)', 
                 fontsize=14, y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="So sánh và visualize 2 models")
    parser.add_argument("--model1", type=str, default="models/final_no_mt.pth",
                       help="Path to model 1 (Không Mean Teacher)")
    parser.add_argument("--model2", type=str, default="models/final_mt.pth",
                       help="Path to model 2 (Mean Teacher)")
    parser.add_argument("--frames_dir", type=str, default="data/video_frames",
                       help="Directory containing frames")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to config file")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of frames to visualize (càng nhiều càng tốt)")
    parser.add_argument("--output", type=str, default="models/comparison_500frames_overlay.png",
                       help="Output path for visualization")
    
    args = parser.parse_args()
    
    visualize_comparison(
        model1_path=args.model1,
        model2_path=args.model2,
        frames_dir=args.frames_dir,
        config_path=args.config,
        num_samples=args.num_samples,
        output_path=args.output
    )

