from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, Union
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
        
    def log(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):
        """
        Log metrics to trainer's logging system
        
        Args:
            metrics: Dictionary of metric names and values to log
            step: Optional step number. If None, uses trainer's global_step
        """
        if self.trainer is None:
            print(f"Warning: No trainer available for logging. Metrics: {metrics}")
            return
            
        # Use trainer's global_step if no step is provided
        log_step = step if step is not None else getattr(self.trainer, 'global_step', 0)
        
        # Forward the log call to trainer's logging system
        if hasattr(self.trainer, 'run') and self.trainer.run is not None:
            # Only log if this is the main process (rank 0)
            if getattr(self.trainer, 'is_main', True):
                self.trainer.run.log(metrics, step=log_step)
        else:
            print(f"Warning: Trainer logging not initialized. Metrics: {metrics}")
        
    def print_log(self, message: str, step: Optional[int] = None):
        """
        Print log message with step information (only on main process)
        
        Args:
            message: Message to print
            step: Optional step number. If None, uses trainer's global_step
        """
        if self.trainer is None or not getattr(self.trainer, 'is_main', True):
            return
            
        log_step = step if step is not None else getattr(self.trainer, 'global_step', 0)
        print(f"[rank0] step {log_step:06d} | {message}")
        
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
