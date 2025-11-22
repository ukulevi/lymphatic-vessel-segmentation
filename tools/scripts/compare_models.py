import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

# Assuming src/visualization.py and src/config.py are importable
# Add the project root to the Python path for imports to work
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.visualization import plot_training_curves, visualize_evaluation_table

def compare_models():
    parser = argparse.ArgumentParser(description="Compare training results of two models.")
    parser.add_argument(
        "--log-dir1",
        type=str,
        required=True,
        help="Path to the first model's log directory (containing metrics.csv)."
    )
    parser.add_argument(
        "--log-dir2",
        type=str,
        required=True,
        help="Path to the second model's log directory (containing metrics.csv)."
    )
    parser.add_argument(
        "--name1",
        type=str,
        default="Model A",
        help="Name for the first model in plots/tables."
    )
    parser.add_argument(
        "--name2",
        type=str,
        default="Model B",
        help="Name for the second model in plots/tables."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_comparisons",
        help="Directory to save comparison plots and tables."
    )
    parser.add_argument(
        "--metrics-to-plot",
        nargs='+',
        default=['loss', 'iou', 'dice', 'pixel_acc', 'boundary_f1'],
        help="List of metrics to plot (e.g., 'loss iou dice')."
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metrics
    metrics_path1 = os.path.join(args.log_dir1, "metrics.csv")
    metrics_path2 = os.path.join(args.log_dir2, "metrics.csv")

    if not os.path.exists(metrics_path1):
        raise FileNotFoundError(f"metrics.csv not found in {args.log_dir1}")
    if not os.path.exists(metrics_path2):
        raise FileNotFoundError(f"metrics.csv not found in {args.log_dir2}")

    df1 = pd.read_csv(metrics_path1)
    df2 = pd.read_csv(metrics_path2)

    # Generate comparison plot
    plot_save_path = os.path.join(args.output_dir, f"comparison_curves_{args.name1}_vs_{args.name2}.png")
    plot_training_curves(
        df_main=df1, 
        df_compare=df2, 
        model_labels=(args.name1, args.name2),
        save_path=plot_save_path,
        metrics_to_plot=args.metrics_to_plot
    )
    print(f"Comparison plot saved to {plot_save_path}")

    # Generate comparison table
    table_save_path = os.path.join(args.output_dir, f"comparison_table_{args.name1}_vs_{args.name2}.txt")
    visualize_evaluation_table(
        df_main=df1, 
        df_compare=df2, 
        model_labels=(args.name1, args.name2), 
        save_path=table_save_path
    )


if __name__ == "__main__":
    compare_models()
