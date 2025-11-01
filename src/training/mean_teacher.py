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

from ..utils.logging import TrainingLogger
from ..utils.metrics import dice_coefficient, iou_score, pixel_accuracy, boundary_f1_score, avg_score
from ..utils.losses import CombinedLoss
from ..config import TrainingConfig


class MeanTeacherTrainer:
    """
    Mean Teacher Trainer:
    - Student: Model được train với gradient
    - Teacher: EMA của Student (không train trực tiếp)
    - Consistency Loss: MSE giữa student và teacher predictions trên unlabeled data
    """
    
    def __init__(self, student_model, teacher_model, optimizer, config: TrainingConfig, logger: Optional[TrainingLogger] = None, device: Optional[str] = None):
        self.student = student_model
        self.teacher = teacher_model
        
        # Initialize teacher với weights của student
        self.update_teacher(alpha=1.0)  # Copy student weights
        
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        
        # Device handling
        self.device = device or config.device
        if self.device == "cuda" and not torch.cuda.is_available():
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        elif self.device == "mps":
            if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
                self.device = "cpu"
        
        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device)
        
        # EMA decay (thường 0.99 hoặc 0.999)
        self.ema_decay = getattr(config, 'ema_decay', 0.99)
        
        # Consistency weight (ramp-up từ 0)
        self.consistency_weight = getattr(config, 'consistency_weight', 1.0)
        self.consistency_rampup = getattr(config, 'consistency_rampup', 100)
        
        # Loss function
        self.criterion = CombinedLoss()
        
        # Best model tracking
        self._best_val_loss = float('inf')
        self._best_model_state = None
        self._epochs_since_improve = 0
    
    def update_teacher(self, alpha=None):
        """
        Update teacher model bằng EMA của student
        
        Teacher = alpha * Teacher + (1 - alpha) * Student
        """
        if alpha is None:
            alpha = self.ema_decay
        
        # Update teacher weights bằng EMA
        with torch.no_grad():
            for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
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
    
    def train_epoch(self, labeled_loader, unlabeled_loader, epoch):
        """
        Train một epoch với Mean Teacher
        
        Args:
            labeled_loader: DataLoader cho labeled data
            unlabeled_loader: DataLoader cho unlabeled data (có thể None)
            epoch: Current epoch number
        """
        self.student.train()
        self.teacher.eval()  # Teacher luôn ở eval mode
        
        total_loss = 0.0
        total_supervised_loss = 0.0
        total_consistency_loss = 0.0
        
        # Consistency weight cho epoch này
        consistency_weight = self.get_consistency_weight(epoch)
        
        # Iterate qua labeled và unlabeled data
        labeled_iter = iter(labeled_loader) if labeled_loader else None
        unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader else None
        
        num_batches = len(labeled_loader) if labeled_loader else 0
        if unlabeled_loader:
            num_batches = max(num_batches, len(unlabeled_loader))
        
        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}")
        
        for batch_idx in pbar:
            # Initialize losses
            has_labeled = False
            has_unlabeled = False
            
            # 1. Process labeled data (supervised loss)
            supervised_loss = None
            
            if labeled_iter:
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
                    student_pred = self.student(images)
                    if isinstance(student_pred, dict):
                        student_pred = student_pred.get('out', student_pred.get('logits', student_pred))
                    
                    # Update loss schedule
                    if isinstance(self.criterion, CombinedLoss):
                        self.criterion.update_alpha(epoch)
                    
                    # Supervised loss
                    supervised_loss = self.criterion(student_pred, masks.float())
                    has_labeled = True
                    
                except StopIteration:
                    pass
            
            # 2. Process unlabeled data (consistency loss)
            consistency_loss = None
            
            if unlabeled_iter and consistency_weight > 0:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                    if isinstance(unlabeled_batch, dict):
                        images_unlabeled = unlabeled_batch['image'].to(self.device)
                    elif isinstance(unlabeled_batch, (list, tuple)) and len(unlabeled_batch) > 0:
                        images_unlabeled = unlabeled_batch[0].to(self.device)
                    else:
                        continue
                    
                    # Student: weak augmentation (đã apply trong dataset)
                    student_pred_unlabeled = self.student(images_unlabeled)
                    if isinstance(student_pred_unlabeled, dict):
                        student_pred_unlabeled = student_pred_unlabeled.get('out', student_pred_unlabeled.get('logits', student_pred_unlabeled))
                    student_prob = torch.sigmoid(student_pred_unlabeled)
                    
                    # Teacher: same image với no_grad
                    with torch.no_grad():
                        teacher_pred_unlabeled = self.teacher(images_unlabeled)
                        if isinstance(teacher_pred_unlabeled, dict):
                            teacher_pred_unlabeled = teacher_pred_unlabeled.get('out', teacher_pred_unlabeled.get('logits', teacher_pred_unlabeled))
                        teacher_prob = torch.sigmoid(teacher_pred_unlabeled)
                    
                    # Consistency loss: MSE giữa student và teacher probabilities
                    consistency_loss = F.mse_loss(student_prob, teacher_prob)
                    has_unlabeled = True
                    
                except StopIteration:
                    pass
            
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
                self.optimizer.zero_grad()
                total_loss_batch.backward()
                self.optimizer.step()
                
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
        self.teacher.eval()
        
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
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
                predictions = self.teacher(images)
                if isinstance(predictions, dict):
                    predictions = predictions.get('out', predictions.get('logits', predictions))
                
                # Update loss schedule
                if isinstance(self.criterion, CombinedLoss):
                    self.criterion.update_alpha(0)  # Epoch 0 cho validation
                
                loss = self.criterion(predictions, masks.float())
                total_loss += loss.item()
                
                # Collect predictions và targets cho metrics
                probs = torch.sigmoid(predictions)
                preds = (probs > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(masks.cpu().numpy())
        
        # Compute metrics
        import numpy as np
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        dice = dice_coefficient(all_preds, all_targets)
        iou = iou_score(all_preds, all_targets)
        pixel_acc = pixel_accuracy(all_preds, all_targets)
        boundary_f1 = boundary_f1_score(all_preds, all_targets)
        avg_sc = avg_score(iou, dice, pixel_acc, boundary_f1)  # Đúng thứ tự: iou, dice, pixel_acc, boundary_f1
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'dice': dice,
            'iou': iou,
            'pixel_acc': pixel_acc,
            'boundary_f1': boundary_f1,
            'avg_score': avg_sc
        }
    
    def train(self, labeled_loader, unlabeled_loader, val_loader, scheduler=None):
        """
        Full training loop với Mean Teacher
        """
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(labeled_loader, unlabeled_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Logging
            if self.logger:
                # Log metrics (tương tự Trainer)
                metrics_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'val_dice': val_metrics['dice'],
                    'val_iou': val_metrics['iou'],
                    'val_pixel_acc': val_metrics['pixel_acc'],
                    'val_boundary_f1': val_metrics['boundary_f1'],
                    'val_avg_score': val_metrics['avg_score'],
                    'supervised_loss': train_metrics['supervised_loss'],
                    'consistency_loss': train_metrics['consistency_loss'],
                    'consistency_weight': train_metrics['consistency_weight']
                }
                self.logger.log_metrics(metrics_dict)
            
            # Early stopping
            if val_metrics['loss'] < self._best_val_loss:
                self._best_val_loss = val_metrics['loss']
                self._best_model_state = deepcopy(self.teacher.state_dict())
                self._epochs_since_improve = 0
            else:
                self._epochs_since_improve += 1
                if self._epochs_since_improve >= self.config.early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                  f"Train Loss: {train_metrics['loss']:.4f}, "
                  f"Val Loss: {val_metrics['loss']:.4f}, "
                  f"Val Dice: {val_metrics['dice']:.4f}, "
                  f"Consistency: {train_metrics['consistency_loss']:.4f} (λ={train_metrics['consistency_weight']:.3f})")
        
        # Load best model
        if self._best_model_state is not None:
            self.teacher.load_state_dict(self._best_model_state)
        
        return self.teacher  # Trả về teacher model (tốt hơn student)


def create_mean_teacher_models(base_model):
    """
    Tạo student và teacher models từ base model
    
    Args:
        base_model: Model architecture (UNet++ với ResNet34)
    
    Returns:
        student_model: Model để train
        teacher_model: Copy của student (sẽ được update bằng EMA)
    """
    # Student: Model được train
    student_model = base_model
    
    # Teacher: Copy của student (không train)
    teacher_model = deepcopy(base_model)
    
    # Freeze teacher (không update bằng gradient)
    for param in teacher_model.parameters():
        param.requires_grad = False
    
    return student_model, teacher_model

