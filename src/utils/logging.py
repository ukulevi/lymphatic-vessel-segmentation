"""
Logging utilities for tracking training progress
"""
import os
import json
import csv
import logging as _pylogging
from datetime import datetime

class TrainingLogger:
    """Log training metrics and config to CSV/JSON"""
    
    def __init__(self, log_dir, experiment_name=None):
        """
        Args:
            log_dir: Directory to save logs
            experiment_name: Optional name, defaults to timestamp
        """
        self.log_dir = log_dir
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # Create log directory
        self.experiment_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize metrics log
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        self.metrics_fields = None
        
    def log_config(self, config):
        """Save experiment configuration"""
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
            
    def log_metrics(self, metrics):
        """Append a single metrics row to CSV.
        If an 'epoch' key is present, it will be kept; otherwise a timestamp will be used in 'step'.
        """
        if "step" not in metrics and "epoch" not in metrics:
            metrics = {"step": datetime.now().isoformat(), **metrics}
        
        # Initialize CSV file with headers if needed
        if self.metrics_fields is None:
            self.metrics_fields = list(metrics.keys())
            with open(self.metrics_file, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_fields)
                writer.writeheader()
                
        # Append metrics
        with open(self.metrics_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_fields)
            writer.writerow(metrics)
            
    def log_model(self, model_path, metrics=None):
        """
        Log saved model path and optional metrics
        Args:
            model_path: Path to saved model weights
            metrics: Optional dict of metrics for this model
        """
        models_file = os.path.join(self.experiment_dir, "models.jsonl")
        entry = {
            "path": model_path,
            "step": datetime.now().isoformat()
        }
        if metrics:
            entry["metrics"] = metrics
            
        with open(models_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
            
    def get_best_model(self, metric="val_loss", mode="min"):
        """
        Get path to best model according to metric
        Args:
            metric: Metric name to compare
            mode: "min" or "max"
        Returns:
            Path to best model, or None if no models logged
        """
        models_file = os.path.join(self.experiment_dir, "models.jsonl")
        if not os.path.exists(models_file):
            return None
            
        best_path = None
        best_value = float("inf") if mode == "min" else float("-inf")
        
        with open(models_file) as f:
            for line in f:
                entry = json.loads(line)
                if "metrics" not in entry:
                    continue
                    
                value = entry["metrics"].get(metric)
                if value is None:
                    continue
                    
                if mode == "min" and value < best_value:
                    best_value = value
                    best_path = entry["path"]
                elif mode == "max" and value > best_value:
                    best_value = value
                    best_path = entry["path"]
                    
        return best_path

    def log_message(self, message: str):
        """Append a plain text message to a log file and also print it."""
        txt_file = os.path.join(self.experiment_dir, "events.log")
        try:
            with open(txt_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}\t{message}\n")
        except Exception:
            pass
        print(message)


def get_logger(name: str) -> _pylogging.Logger:
    """Lightweight project-wide logger factory.
    Ensures consistent formatting and avoids duplicate handlers on repeated calls.
    """
    logger = _pylogging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(_pylogging.INFO)
        handler = _pylogging.StreamHandler()
        formatter = _pylogging.Formatter('[%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    return logger

    
    
    
    
    
    
    
    
    