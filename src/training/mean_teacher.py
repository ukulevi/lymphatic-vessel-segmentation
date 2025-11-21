"""
Mean Teacher Implementation cho Stage 3
Student model được train, Teacher model là EMA của Student
Consistency loss giữa student và teacher predictions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from typing import Optional, Dict
from tqdm import tqdm
from torch.utils.data import DataLoader

from ..utils.logging import TrainingLogger
from src.utils.metrics import (
    dice_coefficient, iou_score, pixel_accuracy, boundary_f1_score, avg_score
)
from src.training.losses import BCELoss, CombinedLoss, DeepSupervisionLoss
from src.config import ModelConfig, TrainingConfig
from src.training.trainer import Trainer

class MeanTeacherTrainer(Trainer):
    """
    Trainer for Mean Teacher semi-supervised learning.
    """
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        model_config: ModelConfig,
        config: TrainingConfig,
        logger: TrainingLogger,
        device: str = "cuda"
    ):
        # Initialize the base Trainer with the student model
        super().__init__(student_model, model_config, config, logger, device)
        
        # Mean Teacher specific attributes
        self.student_model = self.model  # Rename for clarity
        self.teacher_model = teacher_model
        self.optimizer = optimizer
        self.teacher_model = self.teacher_model.to(self.device)

        self.consistency_loss = nn.MSELoss()

        self.consistency_rampup = config.consistency_rampup
        self.consistency_weight = config.consistency_weight
        self.ema_decay = config.ema_decay

    def update_teacher(self, alpha=None):
        """
        Update teacher model bằng EMA của student

        Teacher = alpha * Teacher + (1 - alpha) * Student
        """
        if alpha is None:
            alpha = self.ema_decay

        # Update teacher weights bằng EMA
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher_model.parameters(), self.student_model.parameters()):
                teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1 - alpha)
    
    def get_consistency_weight(self, epoch):
        """
        Ramp-up consistency weight từ 0 đến consistency_weight
        """
        if epoch < self.consistency_rampup:
            # Ramp-up: epoch 0 → 0, epoch 1 → consistency_weight/rampup, ...
            # Tránh chia cho 0 khi epoch = 0
            if epoch == 0:
                return 0.0
            return self.consistency_weight * (epoch / self.consistency_rampup)
        return self.consistency_weight

    def train_epoch(self, train_loaders, optimizer, epoch):
        """
        Train một epoch với Mean Teacher

        Args:
            train_loaders: A tuple of (labeled_loader, unlabeled_loader)
            optimizer: The optimizer for the student model.
            epoch: Current epoch number
        """
        labeled_loader, unlabeled_loader = train_loaders

        self.student_model.train()
        self.teacher_model.eval()  # Teacher luôn ở eval mode
        
        total_loss = 0.0
        total_supervised_loss = 0.0
        total_consistency_loss = 0.0
        
        # Consistency weight cho epoch này
        consistency_weight = self.get_consistency_weight(epoch)
        
        # Update alpha for CombinedLoss if applicable (once per epoch)
        if isinstance(self.criterion, CombinedLoss) or \
           (isinstance(self.criterion, DeepSupervisionLoss) and isinstance(self.criterion.criterion, CombinedLoss)):
            inner_loss = self.criterion.criterion if isinstance(self.criterion, DeepSupervisionLoss) else self.criterion
            inner_loss.update_alpha(epoch)

        # Iterate qua labeled và unlabeled data
        # Ensure we can loop through the smaller dataset if datasets have different lengths
        if len(labeled_loader) > len(unlabeled_loader):
            unlabeled_iter = iter(torch.utils.data.dataloader.DataLoader(unlabeled_loader.dataset, batch_size=unlabeled_loader.batch_size, shuffle=True, num_workers=unlabeled_loader.num_workers, pin_memory=True))
            labeled_iter = iter(labeled_loader)
            num_batches = len(labeled_loader)
        else:
            labeled_iter = iter(torch.utils.data.dataloader.DataLoader(labeled_loader.dataset, batch_size=labeled_loader.batch_size, shuffle=True, num_workers=labeled_loader.num_workers, pin_memory=True))
            unlabeled_iter = iter(unlabeled_loader)
            num_batches = len(unlabeled_loader)
        
        # Fallback if one loader is missing
        if not labeled_loader:
            num_batches = len(unlabeled_loader)
        if not unlabeled_loader:
            num_batches = len(labeled_loader)
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in pbar:
            # Initialize losses
            has_labeled = False
            has_unlabeled = False
            
            # 1. Process labeled data (supervised loss)
            supervised_loss = None
            
            if labeled_loader:
                try:
                    labeled_batch = next(labeled_iter)
                    if isinstance(labeled_batch, dict):
                        images = labeled_batch['image'].to(self.device)
                        masks = labeled_batch['mask'].to(self.device)
                    elif isinstance(labeled_batch, (list, tuple)) and len(labeled_batch) >= 2:
                        images, masks = labeled_batch[0], labeled_batch[1]
                        images = images.to(self.device)
                        masks = masks.to(self.device)
                    else:
                        continue
                    
                    # Student prediction
                    student_pred = self.student_model(images)
                    
                    # Supervised loss
                    supervised_loss = self.criterion(student_pred, masks.float())
                    has_labeled = True
                    
                except StopIteration:
                    continue # Should not happen with the new iterator logic
            
            # 2. Process unlabeled data (consistency loss)
            consistency_loss = None
            
            if unlabeled_loader and consistency_weight > 0:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                    if isinstance(unlabeled_batch, dict):
                        images_unlabeled = unlabeled_batch['image'].to(self.device)
                    elif isinstance(unlabeled_batch, (list, tuple)) and len(unlabeled_batch) > 0:
                        images_unlabeled = unlabeled_batch[0].to(self.device)
                    else:
                        continue
                    
                    # Student: weak augmentation (đã apply trong dataset)
                    student_pred_unlabeled = self.student_model(images_unlabeled)
                    if isinstance(student_pred_unlabeled, (list, tuple)):
                        student_pred_unlabeled = student_pred_unlabeled[0] # Use main output
                    student_prob = torch.sigmoid(student_pred_unlabeled)
                    
                    # Teacher: same image với no_grad
                    with torch.no_grad():
                        teacher_pred_unlabeled = self.teacher_model(images_unlabeled)
                        if isinstance(teacher_pred_unlabeled, (list, tuple)):
                            teacher_pred_unlabeled = teacher_pred_unlabeled[0] # Use main output
                        teacher_prob = torch.sigmoid(teacher_pred_unlabeled)
                    
                    # Consistency loss: MSE giữa student và teacher probabilities
                    consistency_loss = F.mse_loss(student_prob, teacher_prob)
                    has_unlabeled = True
                    
                except StopIteration:
                    continue # Should not happen with the new iterator logic
            
            # 3. Total loss (chỉ backward nếu có ít nhất 1 loss)
            if has_labeled or has_unlabeled:
                if supervised_loss is not None and consistency_loss is not None:
                    total_loss_batch = supervised_loss + consistency_weight * consistency_loss
                elif supervised_loss is not None:
                    total_loss_batch = supervised_loss
                elif consistency_loss is not None:
                    total_loss_batch = consistency_weight * consistency_loss
                else:
                    continue
                
                # 4. Backward và update student
                optimizer.zero_grad()
                total_loss_batch.backward()
                optimizer.step()

                # 5. Update teacher (EMA) - sau mỗi batch
                self.update_teacher()
                
                # 6. Logging
                total_loss += total_loss_batch.item()
                if supervised_loss is not None:
                    total_supervised_loss += supervised_loss.item()
                if consistency_loss is not None:
                    total_consistency_loss += consistency_loss.item()
                
                pbar.set_postfix({
                    'loss': total_loss_batch.item(),
                    'sup': supervised_loss.item() if supervised_loss is not None else 0.0,
                    'cons': consistency_loss.item() if consistency_loss is not None else 0.0,
                    'lambda': consistency_weight
                })
            
        
        # Average losses
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_supervised = total_supervised_loss / num_batches if num_batches > 0 else 0.0
        avg_consistency = total_consistency_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'supervised_loss': avg_supervised,
            'consistency_loss': avg_consistency,
            'consistency_weight': consistency_weight
        }
    
    def validate(self, val_loader):
        """
        Validate với teacher model (teacher thường tốt hơn student)
        """
        self.teacher_model.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Skip if batch is empty or invalid
                if not batch:
                    continue

                if isinstance(batch, dict):
                    images = batch['image'].to(self.device)
                    masks = batch['mask'].to(self.device)
                elif isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    images, masks = batch[0], batch[1]
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                else:
                    continue
                
                # Dùng teacher để validate
                predictions = self.teacher_model(images)
                
                # For deep supervision, use the main output for validation
                if isinstance(predictions, (list, tuple)):
                    main_prediction = predictions[0]
                else:
                    main_prediction = predictions
                
                # During validation, use the inner criterion to avoid deep supervision weights.
                if isinstance(self.criterion, DeepSupervisionLoss):
                    loss = self.criterion.criterion(main_prediction, masks.float())
                else:
                    loss = self.criterion(main_prediction, masks.float())

                total_loss += loss.item()
                
                # Collect predictions và targets cho metrics
                probs = torch.sigmoid(main_prediction)
                preds = (probs > 0.5).float()
                all_preds.append(preds)
                all_targets.append(masks)
        
        # Compute metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        dice = dice_coefficient(all_preds, all_targets)
        iou = iou_score(all_preds, all_targets)
        pixel_acc = pixel_accuracy(all_preds, all_targets)
        boundary_f1 = boundary_f1_score(all_preds, all_targets)
        avg_sc = avg_score(iou, dice, pixel_acc, boundary_f1)  # Đúng thứ tự: iou, dice, pixel_acc, boundary_f1
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_dice': dice,
            'val_iou': iou,
            'val_pixel_acc': pixel_acc,
            'val_boundary_f1': boundary_f1,
            'val_avg': avg_sc
        }
    
    def train(self, labeled_loader: DataLoader, unlabeled_loader: Optional[DataLoader], val_loader: DataLoader, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> nn.Module:
        """
        Override the main training loop to handle Mean Teacher's specific epoch training.
        This method now prepares the arguments and calls the base `Trainer.train` method,
        which handles the main epoch loop, validation, logging, and early stopping.
        """
        # The base `train` method expects `train_loader` as the first argument.
        # We pass a tuple of loaders, and our overridden `train_epoch` will know how to handle it.
        train_loaders = (labeled_loader, unlabeled_loader)
        
        # Call the parent's train method
        super().train(train_loaders, val_loader, self.optimizer, scheduler)
        
        return self.teacher_model  # Trả về teacher model (tốt hơn student)