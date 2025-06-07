import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from micro_module import MicroModule
from pytree_utils import to_device, print_batch_info, validate_batch_format


class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate flattened size: 28->14->7, so 7*7*64
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch_size, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch_size, 64, 7, 7)
        x = self.dropout1(x)
        
        x = x.view(-1, 7 * 7 * 64)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class MNISTModule(MicroModule):
    """MNIST Classification Module using the micro framework"""
    
    def __init__(self, learning_rate: float = 1e-3, num_classes: int = 10, debug: bool = False):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.debug = debug  # Enable debug printing
        
        # For tracking validation metrics
        self.validation_predictions = []
        self.validation_targets = []
        
    def configure_model(self) -> nn.Module:
        """Configure and return the CNN model"""
        return SimpleCNN(num_classes=self.num_classes)
        
    def configure_optimizers(self):
        """Configure and return the optimizer"""
        return Adam(self.model.parameters(), lr=self.learning_rate)
        
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Training step logic for MNIST
        """
        # Validate batch format (optional, can be disabled for performance)
        if self.debug and batch_idx == 0:
            validate_batch_format(batch)
            print_batch_info(batch, "training_batch")
        
        # Move batch to device using utils
        batch = to_device(batch, self.device)
        
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        
        # Calculate accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss,
            'n_tokens': len(images),  # Use batch size as "tokens" for MNIST
            'accuracy': accuracy.item()
        }
        
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step logic for MNIST
        """
        # Move batch to device using utils
        
        images = batch['image']
        labels = batch['label']
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(images)
            loss = F.cross_entropy(logits, labels)
            
            # Store predictions and targets for metrics calculation
            predictions = torch.argmax(logits, dim=1)
            
            # Move to CPU for sklearn metrics
            self.validation_predictions.extend(predictions.cpu().numpy())
            self.validation_targets.extend(labels.cpu().numpy())
            
            # Calculate batch accuracy
            accuracy = (predictions == labels).float().mean()
        
        return {
            'loss': loss,
            'n_tokens': len(images),
            'accuracy': accuracy.item()
        }
    
    def on_validation_epoch_end(self, dataset) -> Optional[Dict[str, Any]]:
        """
        Calculate additional metrics at the end of validation epoch
        """
        if not self.validation_predictions or not self.validation_targets:
            return None
            
        # Calculate overall accuracy
        accuracy = accuracy_score(self.validation_targets, self.validation_predictions)
        
        # Calculate per-class accuracy (optional, for detailed analysis)
        unique_classes = np.unique(self.validation_targets)
        per_class_acc = {}
        
        for cls in unique_classes:
            mask = np.array(self.validation_targets) == cls
            if np.sum(mask) > 0:  # Avoid division by zero
                cls_predictions = np.array(self.validation_predictions)[mask]
                cls_accuracy = np.mean(cls_predictions == cls)
                per_class_acc[f'class_{int(cls)}_acc'] = cls_accuracy
        
        # Clear predictions for next validation
        self.validation_predictions.clear()
        self.validation_targets.clear()
        
        # Return metrics
        metrics = {
            'val_accuracy': accuracy,
            'val_avg_class_acc': np.mean(list(per_class_acc.values())) if per_class_acc else 0.0
        }
        
        # Add per-class accuracies (limit to avoid too much logging)
        metrics.update(per_class_acc)
        
        return metrics 