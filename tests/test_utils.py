#!/usr/bin/env python3
"""
Test script for the pytree_utils module.

This script demonstrates the pytree utility functions for device operations,
data manipulation, and debugging helpers.
"""

import torch
import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytree_utils import (
    to_device, to_dtype, detach_pytree, clone_pytree, map_pytree,
    get_pytree_info, count_parameters, get_memory_usage,
    batch_to_device, print_batch_info, validate_batch_format,
    flatten_pytree
)


def test_basic_pytree_operations():
    """Test basic pytree operations"""
    print("🧪 Testing Basic Pytree Operations")
    print("=" * 50)
    
    # Create a complex nested structure (pytree)
    pytree = {
        'images': torch.randn(8, 3, 32, 32),
        'labels': torch.randint(0, 10, (8,)),
        'metadata': {
            'batch_size': 8,
            'image_ids': torch.arange(8),
            'augmentation_params': [
                torch.tensor([0.5, 0.2]),
                torch.tensor([0.8, 0.1])
            ]
        },
        'extra_data': (
            torch.randn(8, 128),
            {'features': torch.randn(8, 64)}
        )
    }
    
    print("📊 Original pytree info:")
    print(get_pytree_info(pytree, "original"))
    
    # Test device transfer
    if torch.cuda.is_available():
        print("\n🔄 Moving to CUDA...")
        pytree_gpu = to_device(pytree, 'cuda')
        print(get_pytree_info(pytree_gpu, "gpu"))
        
        # Move back to CPU
        print("\n🔄 Moving back to CPU...")
        pytree_cpu = to_device(pytree_gpu, 'cpu')
    else:
        print("\n⚠️  CUDA not available, skipping GPU tests")
        pytree_cpu = pytree
    
    # Test dtype conversion
    print("\n🔢 Converting to float16...")
    pytree_fp16 = to_dtype(pytree_cpu, torch.float16)
    print(get_pytree_info(pytree_fp16, "fp16"))
    
    # Test detaching
    print("\n🔗 Detaching from computation graph...")
    pytree_detached = detach_pytree(pytree_cpu)
    
    # Test cloning
    print("\n📋 Cloning...")
    pytree_cloned = clone_pytree(pytree_cpu)
    
    print("✅ Basic pytree operations test passed!")


def test_batch_operations():
    """Test batch-specific operations"""
    print("\n🎯 Testing Batch Operations")
    print("=" * 50)
    
    # Create MNIST-like batch
    batch = {
        'image': torch.randn(32, 1, 28, 28),
        'label': torch.randint(0, 10, (32,))
    }
    
    print("📋 Validating batch format...")
    is_valid = validate_batch_format(batch, ['image', 'label'])
    print(f"✅ Batch valid: {is_valid}")
    
    print("\n📊 Batch information:")
    print_batch_info(batch, "mnist_batch")
    
    # Test batch device transfer
    if torch.cuda.is_available():
        print("\n🚀 Moving batch to GPU...")
        batch_gpu = batch_to_device(batch, 'cuda')
        print_batch_info(batch_gpu, "mnist_batch_gpu")
    
    print("✅ Batch operations test passed!")


def test_memory_and_parameters():
    """Test memory usage and parameter counting"""
    print("\n💾 Testing Memory and Parameter Functions")
    print("=" * 50)
    
    # Create a simple model-like structure
    model_params = {
        'layer1': {
            'weight': torch.randn(128, 784, requires_grad=True),
            'bias': torch.randn(128, requires_grad=True)
        },
        'layer2': {
            'weight': torch.randn(10, 128, requires_grad=True),
            'bias': torch.randn(10, requires_grad=True)
        },
        'frozen_layer': {
            'weight': torch.randn(64, 64, requires_grad=False)  # Frozen parameters
        }
    }
    
    # Count parameters
    param_counts = count_parameters(model_params)
    print(f"📈 Parameter counts:")
    print(f"   Total: {param_counts['total']:,}")
    print(f"   Trainable: {param_counts['trainable']:,}")
    print(f"   Non-trainable: {param_counts['non_trainable']:,}")
    
    # Check memory usage
    memory_info = get_memory_usage(model_params)
    print(f"\n💾 Memory usage: {memory_info['total_memory_mb']:.2f} MB")
    
    print("✅ Memory and parameter functions test passed!")


def test_map_function():
    """Test the map_pytree function"""
    print("\n🗺️  Testing Map Function")
    print("=" * 50)
    
    # Create pytree with various tensors
    data = {
        'tensor1': torch.randn(10) * 100,  # Large values
        'nested': {
            'tensor2': torch.randn(5, 5) * 50
        },
        'list_data': [torch.randn(3) * 10, torch.randn(2) * 20]
    }
    
    print("📊 Original data ranges:")
    print(get_pytree_info(data, "original"))
    
    # Normalize all tensors to [0, 1] range
    def normalize_tensor(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)
    
    normalized_data = map_pytree(normalize_tensor, data)
    
    print("\n📊 Normalized data ranges:")
    print(get_pytree_info(normalized_data, "normalized"))
    
    print("✅ Map function test passed!")


def test_flatten_pytree():
    """Test pytree flattening and reconstruction"""
    print("\n🗂️  Testing Flatten Pytree")
    print("=" * 50)
    
    # Create a nested structure
    original_pytree = {
        'layer1': {
            'weight': torch.randn(3, 4),
            'bias': torch.randn(3)
        },
        'layer2': {
            'weight': torch.randn(1, 3),
            'bias': torch.randn(1)
        }
    }
    
    print("📊 Original pytree:")
    print(get_pytree_info(original_pytree, "original"))
    
    # Flatten the pytree
    flat_tensors, unflatten_func = flatten_pytree(original_pytree)
    print(f"\n📋 Flattened to {len(flat_tensors)} tensors")
    
    # Modify the flattened tensors (e.g., add noise)
    modified_tensors = [tensor + 0.1 * torch.randn_like(tensor) for tensor in flat_tensors]
    
    # Reconstruct the pytree
    reconstructed_pytree = unflatten_func(modified_tensors)
    
    print("\n📊 Reconstructed pytree:")
    print(get_pytree_info(reconstructed_pytree, "reconstructed"))
    
    print("✅ Flatten pytree test passed!")


def test_error_cases():
    """Test error handling and edge cases"""
    print("\n🚨 Testing Error Cases")
    print("=" * 50)
    
    # Test with invalid batch
    invalid_batch = {'image': 'not_a_tensor', 'label': torch.randint(0, 10, (5,))}
    print("🔍 Testing invalid batch validation...")
    is_valid = validate_batch_format(invalid_batch)
    print(f"❌ Invalid batch correctly detected: {not is_valid}")
    
    # Test with missing keys
    incomplete_batch = {'image': torch.randn(5, 3, 32, 32)}  # Missing 'label'
    print("\n🔍 Testing incomplete batch...")
    is_valid = validate_batch_format(incomplete_batch, ['image', 'label'])
    print(f"❌ Incomplete batch correctly detected: {not is_valid}")
    
    # Test with empty pytree
    empty_data = {}
    print("\n🔍 Testing empty pytree...")
    memory_info = get_memory_usage(empty_data)
    print(f"💾 Empty pytree memory: {memory_info['total_memory_mb']:.2f} MB")
    
    print("✅ Error cases test passed!")


def main():
    """Run all tests"""
    print("🚀 Testing Pytree Utils Module")
    print("=" * 60)
    
    test_basic_pytree_operations()
    test_batch_operations()
    test_memory_and_parameters()
    test_map_function()
    test_flatten_pytree()
    test_error_cases()
    
    print("\n🎉 All pytree utils tests passed!")
    print("\n📚 Usage Examples:")
    print("""
# Basic usage:
from pytree_utils import to_device, print_batch_info

# Move batch to GPU
batch_gpu = to_device(batch, 'cuda')

# Debug batch info
print_batch_info(batch, "my_batch")

# Normalize all tensors in a nested structure
normalized = map_pytree(lambda x: (x - x.mean()) / x.std(), data)

# Flatten and reconstruct pytrees
flat_tensors, unflatten_func = flatten_pytree(complex_nested_data)
reconstructed = unflatten_func(flat_tensors)
""")


if __name__ == "__main__":
    main() 