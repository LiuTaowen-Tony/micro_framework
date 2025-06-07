import os
from typing import Optional, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler

from micro_data_module import MicroDataModule


class MNISTDataModule(MicroDataModule):
    """MNIST DataModule for the micro framework"""
    
    def __init__(
        self, 
        args,
        data_dir: str = './data',
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: float = 0.1,
        download: bool = True
    ):
        super().__init__(args, batch_size, num_workers)
        self.data_dir = data_dir
        self.val_split = val_split
        self.download = download
        
        # Datasets will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
        ])
        
    def prepare_data(self):
        """Download MNIST dataset if needed"""
        # This runs on main process only
        datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=self.download
        )
        datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=self.download
        )
    
    def setup(self, stage: Optional[str] = None):
        """Setup datasets for different stages"""
        # Prepare data (download if needed)
        self.prepare_data()
        
        if stage == 'fit' or stage is None:
            # Load full training dataset
            mnist_full = datasets.MNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform
            )
            
            # Split training dataset into train and validation
            val_size = int(len(mnist_full) * self.val_split)
            train_size = len(mnist_full) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                mnist_full, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(getattr(self.args, 'seed', 42))
            )
            
        if stage == 'test' or stage is None:
            # Load test dataset
            self.test_dataset = datasets.MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform
            )
    
    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create DataLoader with proper configuration"""
        # Create sampler if distributed
        sampler = None
        if self.is_distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                seed=getattr(self.args, 'seed', 42)
            )
            shuffle = False  # Don't shuffle when using DistributedSampler
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch):
        """Custom collate function to format batch correctly"""
        images, labels = zip(*batch)
        images = torch.stack(images)
        labels = torch.tensor(labels)
        
        return {
            'image': images,
            'label': labels
        }
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader"""
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is None. Make sure to call setup() first.")
        return self._create_dataloader(self.train_dataset, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader"""
        if self.val_dataset is None:
            raise RuntimeError("val_dataset is None. Make sure to call setup() first.")
        return self._create_dataloader(self.val_dataset, shuffle=False)
    
    def test_dataloader(self) -> Optional[DataLoader]:
        """Return test dataloader"""
        if self.test_dataset is None:
            return None
        return self._create_dataloader(self.test_dataset, shuffle=False)
    
    def get_dataset(self, split: str = "train") -> Optional[Dataset]:
        """Get dataset for a specific split"""
        if split == "train":
            return self.train_dataset
        elif split == "val":
            return self.val_dataset
        elif split == "test":
            return self.test_dataset
        else:
            return None 