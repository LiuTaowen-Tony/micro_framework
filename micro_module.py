from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import Optimizer

class MicroModule(ABC):
    """Base class for Lightning-like modules"""
    
    def __init__(self):
        self.trainer = None
        self.device = None
        self._model = None
        self.datamodule = None  # Will be set by trainer
        
    @property
    def model(self):
        return self._model
        
    @model.setter 
    def model(self, value):
        self._model = value
        
    @abstractmethod
    def configure_model(self) -> nn.Module:
        """Configure and return the model"""
        pass
        
    @abstractmethod
    def configure_optimizers(self) -> Optimizer:
        """Configure and return the optimizer"""
        pass
        
    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Training step logic
        Returns: Dict with 'loss', 'n_tokens', and optionally other metrics
        """
        pass
        
    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Validation step logic  
        Returns: Dict with 'loss', 'n_tokens', and optionally other metrics
        """
        pass

    def on_validation_epoch_end(self, dataset) -> Optional[Dict[str, Any]]:
        """
        Called at the end of validation epoch to compute additional metrics
        Returns: Dict with metric names and values, or None
        """
        return None
