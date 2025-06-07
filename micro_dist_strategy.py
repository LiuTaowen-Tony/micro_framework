import os
import functools
from abc import ABC, abstractmethod
from typing import Optional
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy
from torch.distributed.fsdp import ShardingStrategy, CPUOffload, BackwardPrefetch, MixedPrecision
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


class DistributedStrategy(ABC):
    """Base class for distributed training strategies"""
    
    @abstractmethod
    def setup_model(self, model: nn.Module, device: torch.device, is_main: bool) -> nn.Module:
        """Setup and wrap the model for distributed training"""
        pass
    
    @abstractmethod
    def save_model(self, model: nn.Module, path: str, is_main: bool):
        """Save model with strategy-specific considerations"""
        pass


class FSDPStrategy(DistributedStrategy):
    """FSDP (Fully Sharded Data Parallel) strategy for large model training"""
    
    def __init__(
        self,
        sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
        # cpu_offload: bool = False,
        # activation_checkpointing: bool = False,
        auto_wrap_min_params: int = 1e6,  # Wrap layers with at least 1M parameters
        use_size_based_wrap: bool = True,
        precision: str = "fp32"
    ):
        self.sharding_strategy = sharding_strategy
        # self.cpu_offload = cpu_offload
        # self.activation_checkpointing = activation_checkpointing
        self.auto_wrap_min_params = auto_wrap_min_params
        self.use_size_based_wrap = use_size_based_wrap
        self.precision = precision
        
    def _get_fsdp_auto_wrap_policy(self):
        """Get the appropriate auto wrap policy based on configuration"""
        if self.use_size_based_wrap:
            # Size-based wrapping: wrap layers with at least N parameters
            return functools.partial(
                size_based_auto_wrap_policy, 
                min_num_params=self.auto_wrap_min_params
            )
        else:
            # Transformer-based wrapping: wrap specific transformer layers
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={LlamaDecoderLayer},
            )
    
    def _get_fsdp_config(self, device: torch.device):
        """Get FSDP configuration for large models"""
        # CPU offload configuration
        # cpu_offload_config = None
        # if self.cpu_offload:
        #     cpu_offload_config = CPUOffload(offload_params=True)
        
        # # Mixed precision configuration
        # mixed_precision_policy = None
        # if self.precision == "bf16-mixed":
        #     mixed_precision_policy = MixedPrecision(
        #         param_dtype=torch.bfloat16,
        #         reduce_dtype=torch.bfloat16,
        #         buffer_dtype=torch.bfloat16,
        #     )
        # elif self.precision == "fp16-mixed":
        #     mixed_precision_policy = MixedPrecision(
        #         param_dtype=torch.float16,
        #         reduce_dtype=torch.float16,
        #         buffer_dtype=torch.float16,
        #     )
        
        return {
            'sharding_strategy': self.sharding_strategy,
            # 'cpu_offload': cpu_offload_config,
            # 'mixed_precision': mixed_precision_policy,
            'backward_prefetch': BackwardPrefetch.BACKWARD_PRE,
            'device_id': device,
            'auto_wrap_policy': self._get_fsdp_auto_wrap_policy(),
        }
    
    # def _apply_activation_checkpointing(self, model):
    #     """Apply activation checkpointing to save memory"""
    #     if not self.activation_checkpointing:
    #         return
            
    #     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    #         checkpoint_wrapper,
    #         CheckpointImpl,
    #         apply_activation_checkpointing,
    #     )
        
    #     def check_fn(submodule):
    #         # Apply checkpointing to transformer layers
    #         return isinstance(submodule, LlamaDecoderLayer)
        
    #     apply_activation_checkpointing(
    #         model, 
    #         checkpoint_wrapper_fn=checkpoint_wrapper,
    #         check_fn=check_fn
    #     )
        
    def setup_model(self, model: nn.Module, device: torch.device, is_main: bool) -> nn.Module:
        """Setup model with FSDP wrapping for large models"""
        if is_main:
            print("📊 Model memory before FSDP:")
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
            
            # Print model size info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"📈 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Move to device BEFORE FSDP wrapping for better memory management
        model.to(device)
        
        # Apply activation checkpointing before FSDP
        # self._apply_activation_checkpointing(model)
        
        # Get FSDP configuration
        fsdp_config = self._get_fsdp_config(device)
        
        if is_main:
            print(f"🔧 FSDP Configuration:")
            print(f"   Sharding Strategy: {self.sharding_strategy}")
            # print(f"   CPU Offload: {self.cpu_offload}")
            # print(f"   Activation Checkpointing: {self.activation_checkpointing}")
            print(f"   Auto Wrap Min Params: {self.auto_wrap_min_params:,}")
        
        # FSDP wrapping
        wrapped_model = FSDP(model, **fsdp_config)
        
        if is_main:
            print("📊 Model memory after FSDP:")
            print(torch.cuda.memory_summary(device=device, abbreviated=True))
        
        return wrapped_model
        
    def save_model(self, model: nn.Module, path: str, is_main: bool):
        """Save model state with FSDP support"""
        if is_main:
            # For FSDP models, we need to use the FSDP-specific state dict
            with FSDP.state_dict_type(model, FSDP.StateDictType.FULL_STATE_DICT):
                state_dict = model.state_dict()
                torch.save(state_dict, path)
            print(f"[rank0] Model saved to {path}")


DEFAULT_FSDP = FSDPStrategy(
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_min_params=1e6,
    use_size_based_wrap=True,
    precision="fp32"
)
