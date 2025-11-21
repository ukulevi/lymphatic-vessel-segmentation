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
    
    def __init__(self, base_log_dir: str):
        """
        Args:
            base_log_dir: The base directory (e.g., 'logs/Human') where the timestamped log folder will be created.
        """
        self.base_log_dir = base_log_dir
        self.experiment_dir = self._create_log_dir()
        
        # Initialize metrics log
        self.metrics_file = os.path.join(self.experiment_dir, "metrics.csv")
        self.metrics_fields = None
            
    def log_config(self, config):
        """Save experiment configuration"""
        config_file = os.path.join(self.experiment_dir, "config.json")
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)

    def _create_log_dir(self) -> str:
        """Create a timestamped directory for the current run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(self.base_log_dir, timestamp)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
            
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

    def log_message(self, message: str):
        """Append a plain text message to a log file and also print it."""
        txt_file = os.path.join(self.experiment_dir, "events.log")
        try:
            with open(txt_file, "a", encoding="utf-8") as f:
                f.write(f"{datetime.now().isoformat()}\t{message}\n")
        except Exception:
            pass
        print(message)

    def log_model_path(self, model_path: str):
        """
        Logs the path of a saved model to models.jsonl.
        This helps the plotting script identify the model type.
        """
        if not self.experiment_dir:
            return
        
        models_log_file = os.path.join(self.experiment_dir, "models.jsonl")
        log_entry = {"path": model_path.replace("\\", "/")} # Use forward slashes for consistency
        
        try:
            with open(models_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Could not write to models.jsonl: {e}")


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