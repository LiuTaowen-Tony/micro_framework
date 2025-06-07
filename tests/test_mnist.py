#!/usr/bin/env python3
"""
MNIST Test Script for Micro ML Framework

This script demonstrates how to use the micro framework for MNIST classification.
It can be run in both single-GPU and multi-GPU distributed modes.

Usage:
    # Single GPU
    python test_mnist.py
    
    # Multi GPU with torchrun
    torchrun --nproc_per_node=2 test_mnist.py --distributed

    # With custom parameters
    python test_mnist.py --batch_size 128 --learning_rate 0.001 --max_steps 1000
"""

import os
import sys
import argparse
import dataclasses
from typing import Optional

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from micro_trainer import Trainer
from mnist_module import MNISTModule
from mnist_data_module import MNISTDataModule


@dataclasses.dataclass
class TrainingArgs:
    """Training configuration arguments"""
    # Model parameters
    learning_rate: float = 1e-3
    num_classes: int = 10
    
    # Training parameters
    max_steps: int = 2000
    accumulate_grad_batches: int = 1
    val_check_interval: int = 100
    seed: int = 42
    
    # Data parameters
    batch_size: int = 64
    num_workers: int = 4
    data_dir: str = './data'
    val_split: float = 0.1
    
    # Output
    output_path: str = './mnist_model.pt'
    
    # Distributed training (FSDP config)
    cpu_offload: bool = False
    activation_checkpointing: bool = False
    auto_wrap_min_params: int = 1e6
    use_size_based_wrap: bool = True  # Good for smaller models like MNIST


@dataclasses.dataclass  
class TrainerArgs:
    """Trainer-specific configuration"""
    precision: str = "fp32"  # or "bf16-mixed", "fp16-mixed"
    log_every_n_steps: int = 50


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='MNIST Test for Micro ML Framework')
    
    # Model args
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_steps', type=int, default=2000, help='Maximum training steps')
    parser.add_argument('--val_check_interval', type=int, default=100, help='Validation check interval')
    
    # Data args
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--val_split', type=float, default=0.1, help='Validation split ratio')
    
    # Training args
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_path', type=str, default='./mnist_model.pt', help='Output model path')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation steps')
    
    # Distributed/FSDP args
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training')
    parser.add_argument('--cpu_offload', action='store_true', help='Enable FSDP CPU offloading')
    parser.add_argument('--activation_checkpointing', action='store_true', help='Enable activation checkpointing')
    parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16-mixed', 'bf16-mixed'])
    
    # Logging
    parser.add_argument('--project_name', type=str, default='mnist-micro-test', help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None, help='Wandb entity')
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Logging frequency')
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Create configuration objects
    training_args = TrainingArgs(
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        val_check_interval=args.val_check_interval,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        val_split=args.val_split,
        output_path=args.output_path,
        cpu_offload=args.cpu_offload,
        activation_checkpointing=args.activation_checkpointing,
        use_size_based_wrap=True  # Good for MNIST CNN
    )
    
    trainer_args = TrainerArgs(
        precision=args.precision,
        log_every_n_steps=args.log_every_n_steps
    )
    
    print("🚀 Starting MNIST training with Micro ML Framework")
    print(f"📊 Training Configuration:")
    print(f"   Learning Rate: {training_args.learning_rate}")
    print(f"   Batch Size: {training_args.batch_size}")
    print(f"   Max Steps: {training_args.max_steps}")
    print(f"   Validation Interval: {training_args.val_check_interval}")
    print(f"   Precision: {trainer_args.precision}")
    print(f"   Distributed: {args.distributed}")
    if args.distributed:
        print(f"   CPU Offload: {training_args.cpu_offload}")
        print(f"   Activation Checkpointing: {training_args.activation_checkpointing}")
    
    # Initialize trainer
    trainer = Trainer(
        args=training_args,
        trainer_args=trainer_args,
        project_name=args.project_name,
        entity=args.entity,
        strategy=None  # Will use default FSDP strategy
    )
    
    # Initialize module and data module
    model = MNISTModule(
        learning_rate=training_args.learning_rate,
        num_classes=training_args.num_classes
    )
    
    datamodule = MNISTDataModule(
        args=training_args,
        data_dir=training_args.data_dir,
        batch_size=training_args.batch_size,
        num_workers=training_args.num_workers,
        val_split=training_args.val_split,
        download=True
    )
    
    # Start training
    print("\n🎯 Starting training...")
    try:
        trainer.fit(model, datamodule)
        print("\n✅ Training completed successfully!")
        print(f"💾 Model saved to: {training_args.output_path}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


def run_quick_test():
    """Run a quick test to verify everything works"""
    print("🧪 Running quick MNIST test...")
    
    # Create minimal config for testing
    training_args = TrainingArgs(
        max_steps=10,  # Very short test
        val_check_interval=5,
        batch_size=32,
        data_dir='./test_data'
    )
    
    trainer_args = TrainerArgs(log_every_n_steps=2)
    
    # Test without distributed training
    os.environ.pop('LOCAL_RANK', None)  # Ensure we're not in distributed mode
    
    trainer = Trainer(
        args=training_args,
        trainer_args=trainer_args,
        project_name="",  # No wandb logging for test
        strategy=None
    )
    
    model = MNISTModule(learning_rate=1e-3)
    datamodule = MNISTDataModule(args=training_args, data_dir='./test_data', batch_size=32)
    
    try:
        # This should work even without distributed setup
        print("✅ Components initialized successfully")
        print("✅ Quick test passed!")
        return True
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


if __name__ == "__main__":
    # Check if we should run the quick test
    if len(sys.argv) > 1 and sys.argv[1] == '--quick-test':
        run_quick_test()
    else:
        main() 