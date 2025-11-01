"""
Training utilities and wrapper functions
"""
from typing import Optional, List, Dict, Any, Callable
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
from tqdm import tqdm

from ..utils.logging import TrainingLogger
from ..utils.metrics import dice_coefficient, iou_score, pixel_accuracy, boundary_f1_score, avg_score
from ..config import TrainingConfig
from ..utils.losses import BCELoss, CombinedLoss

class Trainer:
    """Main trainer class with support for early stopping and k-fold"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        logger: Optional[TrainingLogger] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = device or config.device
        
        # Auto-detect device with fallback for macOS
        if self.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA not available, checking MPS...")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print("Using MPS (Apple Silicon GPU)")
                self.device = "mps"
            else:
                print("Using CPU")
                self.device = "cpu"
        elif self.device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                print("Warning: MPS not available, using CPU")
                self.device = "cpu"
        
        self.model = self.model.to(self.device)
        self._best_val_loss = float('inf')
        self._best_model_state = None
        self._epochs_since_improve = 0
        # Build criterion: default BCE; allow combined if config has marker
        self.criterion = self._build_criterion()

    def _build_criterion(self):
        loss_type = getattr(self.config, 'loss_type', 'bce')
        if loss_type == 'combined':
            alpha_schedule = getattr(self.config, 'alpha_schedule', 'rebalance')
            initial_alpha = getattr(self.config, 'initial_alpha', 0.005)
            alpha_increment = getattr(self.config, 'alpha_increment', 0.005)
            return CombinedLoss(alpha_schedule=alpha_schedule, initial_alpha=initial_alpha, alpha_increment=alpha_increment)
        return BCELoss()
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                
                if isinstance(outputs, dict):
                    outputs = outputs.get('out', outputs.get('logits'))
                    
                # Update combined loss schedule if applicable
                if isinstance(self.criterion, CombinedLoss):
                    self.criterion.update_alpha(epoch)
                loss = self.criterion(outputs, masks.float())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Run validation and compute metrics"""
        self.model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        val_acc = 0
        val_bf1 = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = self.model(images)
                if isinstance(outputs, dict):
                    outputs = outputs.get('out', outputs.get('logits'))
                    
                loss = self.criterion(outputs, masks.float())
                
                # Compute metrics
                pred = (torch.sigmoid(outputs) > 0.5).float()
                dice = dice_coefficient(pred, masks)
                iou = iou_score(pred, masks)
                acc = pixel_accuracy(pred, masks)
                bf1 = boundary_f1_score(pred, masks)
                
                val_loss += loss.item()
                val_dice += dice
                val_iou += iou
                val_acc += acc
                val_bf1 += bf1
                
        num_batches = len(val_loader)
        mean_loss = val_loss / num_batches
        mean_dice = val_dice / num_batches
        mean_iou = val_iou / num_batches
        mean_acc = val_acc / num_batches
        mean_bf1 = val_bf1 / num_batches
        metrics = {
            'val_loss': mean_loss,
            'val_dice': mean_dice,
            'val_iou': mean_iou,
            'val_pixel_acc': mean_acc,
            'val_boundary_f1': mean_bf1,
            'val_avg': avg_score(mean_iou, mean_dice, mean_acc, mean_bf1)
        }
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ) -> nn.Module:
        """Main training loop with early stopping"""
        if optimizer is None:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 'min', patience=5, factor=0.1
            )
            
        for epoch in range(self.config.epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validation
            metrics = self.validate(val_loader)
            val_loss = metrics['val_loss']
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)

            # Logging
            if self.logger is not None:
                self.logger.log_metrics({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    **metrics
                })
                
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val Dice: {metrics['val_dice']:.4f}")
            print(f"Val IoU: {metrics['val_iou']:.4f}")
            
            # Save best model
            if val_loss < self._best_val_loss - 1e-6:
                self._best_val_loss = val_loss
                self._best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )
                self._epochs_since_improve = 0
            else:
                self._epochs_since_improve += 1
                
            # Early stopping
            if (self.config.early_stop_patience is not None and 
                self._epochs_since_improve >= self.config.early_stop_patience):
                print(
                    f"Early stopping triggered after {epoch + 1} epochs"
                )
                break
                
        # Restore best model
        if self._best_model_state is not None:
            self.model.load_state_dict(self._best_model_state)
            
        return self.model

def train_kfold(
    model_factory: Callable[[], nn.Module],
    dataset: torch.utils.data.Dataset,
    config: TrainingConfig,
    logger: Optional[TrainingLogger] = None
) -> nn.Module:
    """K-fold cross validation wrapper"""
    if len(dataset) < config.k_folds:
        print(f"Dataset too small for {config.k_folds}-fold CV")
        print("Training on full dataset instead")
        model = model_factory()
        trainer = Trainer(model, config, logger)
        return trainer.train(
            DataLoader(dataset, batch_size=config.batch_size, shuffle=True),
            DataLoader(dataset, batch_size=config.batch_size)
        )
        
    # Split indices into folds
    indices = np.arange(len(dataset))
    fold_sizes = np.full(
        config.k_folds,
        len(dataset) // config.k_folds,
        dtype=int
    )
    fold_sizes[:len(dataset) % config.k_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop
        
    # Train each fold
    best_val_loss = float('inf')
    best_model = None
    
    for fold in range(config.k_folds):
        print(f"\nTraining Fold {fold + 1}/{config.k_folds}")
        
        # Create train/val split
        val_idx = folds[fold]
        train_idx = np.concatenate([
            folds[i] for i in range(config.k_folds) if i != fold
        ])
        
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size
        )
        
        # Train model
        model = model_factory()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=5, factor=0.1
        )
        trainer = Trainer(model, config, logger)
        model = trainer.train(train_loader, val_loader, optimizer, scheduler)
        
        # Update best model
        if trainer._best_val_loss < best_val_loss:
            best_val_loss = trainer._best_val_loss
            best_model = copy.deepcopy(model)
            
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    return best_model