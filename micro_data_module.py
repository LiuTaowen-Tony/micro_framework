import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union, Callable
import torch
from torch.utils.data import Dataset, DataLoader, Subset, DistributedSampler
import torch.distributed as dist


class MicroDataModule(ABC):
    """Base class for Lightning-like data modules with automatic distributed handling"""
    
    def __init__(self, args, batch_size: int = 32, num_workers: int = 4):
        self.args = args
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # These will be set by the trainer during distributed setup
        self.rank = 0
        self.world_size = 1
        self.is_distributed = False
        
    def setup_distributed_info(self, rank: int, world_size: int, is_distributed: bool):
        """Called by trainer to set distributed information"""
        self.rank = rank
        self.world_size = world_size
        self.is_distributed = is_distributed
    
    @abstractmethod
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        pass

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        pass
    
    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        pass
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return test dataloader (optional)"""
        return None
    
    def get_dataset(self, split: str = "train") -> Optional[Dataset]:
        """Get dataset for a specific split"""
        return None

