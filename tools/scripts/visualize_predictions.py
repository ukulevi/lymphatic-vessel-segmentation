"""
Script to visualize predictions of baseline & final models
Display both raw và cleaned predictions for comparation
"""
import os
import torch
import json
from src.models.model_factory import get_model, load_checkpoint
from src.config import ExperimentConfig
from src.data.datasets import LabeledVesselDataset
from src.utils.augment import create_val_transform
from src.visualization import visualize_predictions

def visualize_model_predictions(model_path, output_path, config, device):
    """Visualize predictions of a model"""
    print(f"\n{'='*70}")
    print(f"Loading model from: {model_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(model_path):
        print(f"❌ Cannot detect model: {model_path}")
        return None
    
    # Load model
    model = get_model(config.model)
    model = load_checkpoint(model, model_path, device=device)
    model.eval()
    
    # Create dataset
    dataset = LabeledVesselDataset(
        image_root=config.paths.labeled_dir,
        mask_dir=config.paths.labeled_masks_dir,
        transform=create_val_transform(),
        target_size=config.data.image_size
    )
    
    print(f"Dataset has {len(dataset)} samples")
    print(f"Creating visualization...")
    
    # Visualize with post-processing
    fig = visualize_predictions(
        model,
        dataset,
        num_samples=4,
        device=device,
        use_post_process=True
    )
    
    # Save figure
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Save visualization to: {output_path}")
    
    return fig

def main():
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA (NVIDIA GPU)")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    config_files = [f for f in os.listdir('.') if f.startswith('config_stage') and f.endswith('.json')]
    if not config_files:
        print("No config_stage*.json files found in the root directory. Please ensure configuration files are present.")
        exit(1)

    for config_file in sorted(config_files):
        print(f"\n{'='*70}")
        print(f"Processing configuration from {config_file}")
        print(f"{'='*70}")
        
        try:
            config = ExperimentConfig.from_json_file(config_file)
            print(f"Loaded configuration for type: {config.type}")

            # Visualize baseline model
            baseline_path = os.path.join(config.paths.model_dir, "baseline.pth")
            baseline_output = os.path.join(config.paths.model_dir, "baseline_predictions.png")
            
            if os.path.exists(baseline_path):
                print(f"\n📊 VISUALIZING BASELINE MODEL for {config.type}")
                visualize_model_predictions(baseline_path, baseline_output, config, device)
            else:
                print(f"⚠️ Cannot detect baseline model: {baseline_path}")
            
            # Visualize final model
            model_name_base = "final"
            model_path = os.path.join(config.paths.model_dir, f"{model_name_base}.pth")
            output_path = os.path.join(config.paths.model_dir, f"{model_name_base}_predictions.png")
            
            if os.path.exists(model_path):
                print(f"\n📊 VISUALIZING FINAL MODEL for {config.type}")
                visualize_model_predictions(model_path, output_path, config, device)
            else:
                print(f"⚠️ Cannot detect final model: {model_path}")

        except Exception as e:
            print(f"Error processing {config_file}: {e}")

    print(f"\n{'='*70}")
    print("✓ Complete visualization for all detected models!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()