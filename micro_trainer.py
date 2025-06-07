import os
import functools
import random
from typing import Optional, Dict, Any, Union
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import wandb
import dataclasses

from micro_module import MicroModule
from micro_dist_strategy import DEFAULT_FSDP, DistributedStrategy, FSDPStrategy
from micro_data_module import MicroDataModule
from pytree_utils import to_device


class Trainer:
    """Lightning-like Trainer for distributed training with large model support"""
    
    def __init__(
        self,
        args,
        trainer_args,
        project_name: str = "",
        entity: Optional[str] = None,
        # Distributed strategy configuration
        strategy: Optional[DistributedStrategy] = None,
        replace_sampler_ddp: bool = True
    ):
        self.args = args
        self.trainer_args = trainer_args
        self.project_name = project_name
        self.entity = entity
        self.replace_sampler_ddp = replace_sampler_ddp
        
        # Use FSDP strategy by default if none provided
        self.strategy = strategy
        if strategy is None:
            self.strategy = DEFAULT_FSDP
        
        # Internal state
        self.model = None
        self.optimizer = None
        self.micro_module = None
        self.device = None
        self.local_rank = None
        self.is_main = False
        self.run = None
        
        # Training state
        self.global_step = 0
        self.token_accum = 0
        self.loss_accum = 0.0
        os.environ["WANDB_CONSOLE"] = "wrap"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_float32_matmul_precision("medium")
        
    def setup_distributed(self):
        """Initialize distributed training"""
        dist.init_process_group("nccl")
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        self.is_main = self.local_rank == 0
        
    def setup_logging(self):
        """Initialize wandb logging"""
        if self.is_main and self.project_name:
            self.run = wandb.init(
                project=self.project_name,
                entity=self.entity,
                config=dataclasses.asdict(self.args),
            )
        else:
            class _Dummy:
                def log(self, *_, **__): ...
            self.run = _Dummy()
            
    def setup_model(self, micro_module: MicroModule):
        """Setup model using the distributed strategy"""
        self.micro_module = micro_module
        micro_module.trainer = self
        micro_module.device = self.device
        
        # Setup base model
        base_model = micro_module.configure_model()
        
        # Use strategy to wrap the model for distributed training
        self.model = self.strategy.setup_model(base_model, self.device, self.is_main)
        micro_module.model = self.model
        
        # Setup optimizer AFTER model wrapping
        self.optimizer = micro_module.configure_optimizers()
        
        if self.is_main:
            print("📊 Memory after optimizer setup:")
            print(torch.cuda.memory_summary(device=self.device, abbreviated=True))
        
    def _is_distributed(self) -> bool:
        """Check if we're in distributed mode"""
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
    
    def _create_dataloader_with_distributed_sampler(
        self, 
        original_dataloader: DataLoader, 
        shuffle: bool = False
    ) -> DataLoader:
        """Create new DataLoader with DistributedSampler, preserving all original settings"""
        distributed_sampler = DistributedSampler(
            original_dataloader.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=getattr(self.args, 'seed', 0)
        )
        
        return DataLoader(
            original_dataloader.dataset,
            batch_size=original_dataloader.batch_size,
            sampler=distributed_sampler,
            num_workers=original_dataloader.num_workers,
            collate_fn=original_dataloader.collate_fn,
            pin_memory=original_dataloader.pin_memory,
            drop_last=original_dataloader.drop_last,
            timeout=original_dataloader.timeout,
            worker_init_fn=original_dataloader.worker_init_fn,
            multiprocessing_context=original_dataloader.multiprocessing_context,
            generator=original_dataloader.generator,
            prefetch_factor=original_dataloader.prefetch_factor,
            persistent_workers=original_dataloader.persistent_workers
        )
    
    def _auto_add_distributed_sampler(self, dataloader: DataLoader, shuffle: bool = False) -> DataLoader:
        """Automatically add DistributedSampler to DataLoader if in distributed mode"""
        if not self._is_distributed() or not self.replace_sampler_ddp:
            return dataloader
        
        # Check if already has a DistributedSampler
        if isinstance(dataloader.sampler, DistributedSampler):
            return dataloader
        
        # Create new DataLoader with DistributedSampler
        new_dataloader = self._create_dataloader_with_distributed_sampler(dataloader, shuffle)
        
        if self.is_main:
            print(f"🔄 Automatically replaced sampler with DistributedSampler")
            
        return new_dataloader
        
    def fit(
        self, 
        micro_module: MicroModule, 
        datamodule: MicroDataModule,
    ):
        """
        Main training loop with automatic distributed sampler handling
        
        Args:
            micro_module: The module to train
            datamodule: DataModule instance
        """
        # Setup
        self.setup_distributed()
        self.setup_logging()
        self.setup_model(micro_module)
        
        # Setup datamodule with distributed info (Lightning-like)
        if self._is_distributed():
            datamodule.setup_distributed_info(
                rank=dist.get_rank(),
                world_size=dist.get_world_size(),
                is_distributed=True
            )
        
        # Setup datamodule
        datamodule.setup()
        
        # Get dataloaders and automatically add distributed samplers
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()
        train_dataset = datamodule.get_dataset("train")
        
        # Lightning-like automatic distributed sampler replacement
        train_loader = self._auto_add_distributed_sampler(train_loader, shuffle=True)
        val_loader = self._auto_add_distributed_sampler(val_loader, shuffle=False)
        
        micro_module.datamodule = datamodule
        
        # Seeding
        random.seed(self.args.seed + self.local_rank)
        torch.manual_seed(self.args.seed + self.local_rank)
        
        # Training loop
        self.model.train()
        
        while self.global_step < self.args.max_steps:
            # Set epoch for distributed sampler (Lightning-like automatic handling)
            if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(self.global_step)
            
            if self.is_main:
                print(f"[rank0] step {self.global_step:06d}")

            for batch_idx, batch in enumerate(tqdm(train_loader)):
                if self.global_step >= self.args.max_steps:
                    break

                # Training step
                batch = to_device(batch, self.device)
                step_output = micro_module.training_step(batch, batch_idx)
                n_tok = step_output['n_tokens']
                
                self.token_accum += n_tok
                self.loss_accum += step_output['loss']

                # Gradient accumulation step
                if (batch_idx + 1) % self.args.accumulate_grad_batches == 0:
                    self._optimizer_step()
                    
                    # Logging
                    if self.global_step % self.trainer_args.log_every_n_steps == 0 and self.is_main:
                        self.run.log({"train_loss": self.loss_accum / self.token_accum}, step=self.global_step)
                    
                    # Reset accumulators
                    self._reset_accumulators()
                    self.global_step += 1

                    # Validation
                    if self.global_step % self.args.val_check_interval == 0 and self.global_step > 0:
                        self._run_validation(val_loader, train_dataset, datamodule)

        # Cleanup
        self._save_model()
        self._cleanup()
        
    def _optimizer_step(self):
        """Perform optimizer step with gradient normalization"""
        # Normalize gradients by number of active tokens
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.div_(self.token_accum)
                
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        
    def _reset_accumulators(self):
        """Reset training accumulators"""
        self.token_accum = 0
        self.loss_accum = 0.0
        
    def _run_validation(self, val_loader, train_dataset, datamodule: Optional[MicroDataModule] = None):
        """Run validation and additional metrics"""
        self.model.eval()
        
        # Standard validation
        val_loss = self._evaluate(val_loader)
        if self.is_main:
            self.run.log({"val_loss": val_loss}, step=self.global_step)
            print(f"[rank0] step {self.global_step:06d} | val_loss={val_loss:.4f}")
        
        # Additional metrics from module - add error handling
        try:
            # Use datamodule if available, otherwise fall back to train_dataset
            dataset_for_metrics = train_dataset
            if datamodule is not None:
                dataset_for_metrics = datamodule.get_dataset("train")
            
            additional_metrics = self.micro_module.on_validation_epoch_end(dataset_for_metrics)
            if self.is_main and additional_metrics:
                self.run.log(additional_metrics, step=self.global_step)
                for key, value in additional_metrics.items():
                    print(f"[rank0] step {self.global_step:06d} | {key}={value:.4f}")
        except Exception as e:
            if self.is_main:
                print(f"[rank0] Warning: Error computing additional metrics: {e}")
        
        self.model.train()
        
    def _evaluate(self, val_loader) -> float:
        """Run evaluation on validation set"""
        total_loss = torch.zeros(1, device=self.device)
        total_tokens = torch.zeros(1, device=self.device)

        with torch.no_grad():
            for batch in tqdm(val_loader):
                step_output = self.micro_module.validation_step(batch, 0)
                total_loss += step_output['loss']
                total_tokens += step_output['n_tokens']

        # All-reduce across processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

        return (total_loss / total_tokens).item()
        
    def _save_model(self):
        """Save model using the distributed strategy"""
        self.strategy.save_model(self.model, self.args.output_path, self.is_main)
            
    def _cleanup(self):
        """Cleanup distributed resources"""
        dist.barrier()
        dist.destroy_process_group() 