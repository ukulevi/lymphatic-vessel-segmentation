"""
Script để visualize predictions của baseline và final models
Hiển thị cả raw và cleaned predictions để so sánh
"""
import os
import torch
import json
from src.models.model_factory import get_model, load_checkpoint
from src.config import ExperimentConfig
from src.data.datasets import LabeledVesselDataset
from src.utils.augment import create_val_transform
from src.visualization import visualize_predictions

def visualize_model_predictions(model_path, output_path, config):
    """Visualize predictions của một model"""
    print(f"\n{'='*70}")
    print(f"Đang load model từ: {model_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model: {model_path}")
        return None
    
    # Load model
    model = get_model(config.model)
    model = load_checkpoint(model, model_path, device=config.training.device)
    model.eval()
    
    # Create dataset
    dataset = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=create_val_transform(),
        target_size=config.data.image_size
    )
    
    print(f"Dataset có {len(dataset)} samples")
    print(f"Đang tạo visualization...")
    
    # Visualize with post-processing
    fig = visualize_predictions(
        model,
        dataset,
        num_samples=4,
        device=config.training.device,
        use_post_process=True
    )
    
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Đã lưu visualization vào: {output_path}")
    
    return fig

def main():
    # Load config
    with open("config.json", "r") as f:
        config_dict = json.load(f)
    config = ExperimentConfig.from_dict(config_dict)
    
    # Visualize baseline model
    baseline_path = os.path.join(config.paths.model_dir, "baseline.pth")
    baseline_output = os.path.join(config.paths.model_dir, "baseline_predictions.png")
    
    if os.path.exists(baseline_path):
        print(f"\n📊 VISUALIZING BASELINE MODEL")
        visualize_model_predictions(baseline_path, baseline_output, config)
    else:
        print(f"⚠️ Không tìm thấy baseline model: {baseline_path}")
    
    # Visualize final model
    final_path = os.path.join(config.paths.model_dir, "final.pth")
    final_output = os.path.join(config.paths.model_dir, "final_predictions.png")
    
    if os.path.exists(final_path):
        print(f"\n📊 VISUALIZING FINAL MODEL")
        visualize_model_predictions(final_path, final_output, config)
    else:
        print(f"⚠️ Không tìm thấy final model: {final_path}")
    
    print(f"\n{'='*70}")
    print("✓ Hoàn thành visualization!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

