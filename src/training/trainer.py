import copy
from typing import Optional, Dict, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from tqdm import tqdm

from src.config import TrainingConfig, ModelConfig
from src.utils.logging import TrainingLogger
from src.utils.metrics import (
    avg_score,
    boundary_f1_score,
    dice_coefficient,
    iou_score,
    pixel_accuracy,
)
from src.training.losses import BCELoss, CombinedLoss, DeepSupervisionLoss


class Trainer:
    """Main trainer class with support for early stopping and k-fold"""
    
    def __init__(
        self,
        model: nn.Module,
        model_config: ModelConfig,
        config: TrainingConfig,
        logger: Optional[TrainingLogger] = None,
        device: Optional[str] = None
    ):
        self.model = model
        self.model_config = model_config
        self.config = config
        self.logger = logger
        self.device = device or config.device
        
        # Auto-detect device
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        
        # Fallback for specified but unavailable devices
        elif self.device == "cuda" and not torch.cuda.is_available():
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
            # Check for alpha schedule params in training config
            alpha_schedule = getattr(self.config, 'alpha_schedule', 'rebalance')
            initial_alpha = getattr(self.config, 'initial_alpha', 0.005)
            alpha_increment = getattr(self.config, 'alpha_increment', 0.005)
            criterion = CombinedLoss(alpha_schedule=alpha_schedule, initial_alpha=initial_alpha, alpha_increment=alpha_increment)
        else:
            criterion = BCELoss()

        # Check if deep supervision is enabled in the model config
        deep_supervision_enabled = (
            self.model_config.params and
            self.model_config.params.get('deep_supervision', False)
        )

        # Apply deep supervision loss wrapper if the model is 'cto' or if deep supervision is explicitly enabled
        if self.model_config.name == 'cto' or deep_supervision_enabled:
            # Get aux_weights from config, with a fallback to the default in DeepSupervisionLoss
            aux_weights = getattr(self.config, 'aux_weights', None)
            if aux_weights:
                criterion = DeepSupervisionLoss(criterion, aux_weights=aux_weights)
            else:
                criterion = DeepSupervisionLoss(criterion)
        
        return criterion
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int
    ) -> float:
        """Train one epoch"""
        self.model.train()

        # Update alpha for CombinedLoss if applicable
        if isinstance(self.criterion, CombinedLoss) or \
           (isinstance(self.criterion, DeepSupervisionLoss) and isinstance(self.criterion.criterion, CombinedLoss)):
            inner_loss = self.criterion.criterion if isinstance(self.criterion, DeepSupervisionLoss) else self.criterion
            inner_loss.update_alpha(epoch)

        total_loss = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks.float())
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader, model_to_validate: Optional[nn.Module] = None) -> Dict[str, float]:
        """Run validation and compute metrics"""
        model = model_to_validate if model_to_validate is not None else self.model
        model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                outputs = model(images)

                # For deep supervision, use the main output (index 0) for validation
                if isinstance(outputs, (list, tuple)):
                    final_output = outputs[0]
                else:
                    final_output = outputs

                # During validation, we only care about the loss on the main output.
                # Use the inner criterion to avoid applying deep supervision weights.
                if isinstance(self.criterion, DeepSupervisionLoss):
                    loss = self.criterion.criterion(final_output, masks.float())
                else:
                    loss = self.criterion(final_output, masks.float())

                total_loss += loss.item()
                
                # Collect predictions and targets for metrics
                probs = torch.sigmoid(final_output)
                preds = (probs > 0.5).float()
                all_preds.append(preds)
                all_targets.append(masks)
                
        num_batches = len(val_loader)
        if num_batches == 0:
            print("Warning: Validation loader is empty. Skipping validation metrics.")
            return {
                'val_loss': float('inf'),
                'val_dice': 0,
                'val_iou': 0,
                'val_pixel_acc': 0,
                'val_boundary_f1': 0,
                'val_avg': 0
            }
        
        # Compute metrics over all batches
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        dice = dice_coefficient(all_preds, all_targets)
        iou = iou_score(all_preds, all_targets)
        pixel_acc = pixel_accuracy(all_preds, all_targets)
        boundary_f1 = boundary_f1_score(all_preds, all_targets)
        avg_sc = avg_score(iou, dice, pixel_acc, boundary_f1)
        
        metrics = {
            'val_loss': total_loss / num_batches,
            'val_dice': dice,
            'val_iou': iou,
            'val_pixel_acc': pixel_acc,
            'val_boundary_f1': boundary_f1,
            'val_avg': avg_sc
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
            train_metrics = self.train_epoch(train_loader, optimizer, epoch)
            
            # Validation
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['val_loss']
            
            # Scheduler step
            if scheduler is not None:
                scheduler.step(val_loss)

            # Prepare metrics for logging
            log_metrics = {'epoch': epoch + 1}
            if isinstance(train_metrics, dict):
                log_metrics.update(train_metrics)
                train_loss_display = train_metrics.get('loss', 0.0)
            else: # is a float
                log_metrics['train_loss'] = train_metrics
                train_loss_display = train_metrics
            log_metrics.update(val_metrics)

            # Logging
            if self.logger is not None:
                self.logger.log_metrics(log_metrics)
                
            print(f"Epoch {epoch+1}/{self.config.epochs}")
            print(f"Train Loss: {train_loss_display:.4f}")
            print(f"Valid Loss: {val_loss:.4f}")
            print(f"Dice Coef: {val_metrics['val_dice']:.4f}")
            print(f"IoU Score: {val_metrics['val_iou']:.4f}")
            print(f"Pixel Acc: {val_metrics['val_pixel_acc']:.4f}")
            print(f"BF1 Score: {val_metrics['val_boundary_f1']:.4f}")
            print(f"Avg Score: {val_metrics['val_avg']:.4f}")
            
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
    model_config: ModelConfig,
    config: TrainingConfig,
    logger: Optional[TrainingLogger] = None
) -> nn.Module:
    """K-fold cross validation wrapper"""
    if config.k_folds <= 1 or len(dataset) < config.k_folds:
        print(f"K-fold not applicable (k={config.k_folds}, dataset_size={len(dataset)}). Using a simple train/validation split.")
        
        # Create a simple train/val split
        val_split = 0.1
        val_len = max(1, int(val_split * len(dataset)))
        train_len = len(dataset) - val_len
        
        # Add this check to prevent error when dataset is too small
        if train_len == 0:
            train_subset = dataset
            val_subset = dataset
        else:
            train_subset, val_subset = random_split(dataset, [train_len, val_len])
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.batch_size,
            shuffle=True
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=config.batch_size
        )
        
        model = model_factory()
        trainer = Trainer(model, model_config, config, logger)
        # call trainer.train directly without recreating optimizer/scheduler
        # because they are default-created in trainer.train if not provided.
        return trainer.train(
            train_loader=train_loader,
            val_loader=val_loader
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
        trainer = Trainer(model, model_config, config, logger)
        model = trainer.train(train_loader, val_loader, optimizer, scheduler)
        
        # Update best model
        if trainer._best_val_loss < best_val_loss:
            best_val_loss = trainer._best_val_loss
            best_model = copy.deepcopy(model)
            
        # Clean up
        del model
        torch.cuda.empty_cache()
        
    return best_model
