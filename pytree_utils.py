"""
Pytree utility functions for the micro ML framework.

This module provides common utilities for handling pytrees (nested data structures),
device operations, and other helper functions.

A pytree is a nested structure of containers (like dicts, lists, tuples) with 
tensor/array leaves. This is a common pattern in ML frameworks.
"""

import torch
from typing import Any, Union, Dict, List, Tuple, Optional, Callable
import numpy as np


# Type hint for pytree structures
PyTree = Union[
    torch.Tensor,
    np.ndarray,
    Dict[str, Any],
    List[Any],
    Tuple[Any, ...],
    Any
]


def to_device(pytree: PyTree, device: Union[torch.device, str]) -> PyTree:
    """
    Move a pytree (nested structure of tensors) to the specified device.
    
    Args:
        pytree: Nested structure containing tensors, arrays, dicts, lists, tuples
        device: Target device (e.g., 'cuda', 'cpu', torch.device('cuda:0'))
    
    Returns:
        Pytree with all tensors moved to the specified device
    
    Examples:
        >>> batch = {'images': torch.randn(32, 3, 224, 224), 'labels': torch.randint(0, 10, (32,))}
        >>> batch_gpu = to_device(batch, 'cuda')
        
        >>> nested_data = {'data': [torch.randn(10), {'inner': torch.randn(5)}]}
        >>> nested_gpu = to_device(nested_data, torch.device('cuda:0'))
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    def _to_device_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, np.ndarray):
            # Convert numpy array to tensor and move to device
            return torch.from_numpy(obj).to(device)
        elif isinstance(obj, dict):
            return {key: _to_device_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_to_device_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_device_recursive(item) for item in obj)
        else:
            # For non-tensor types (strings, ints, etc.), return as is
            return obj
    
    return _to_device_recursive(pytree)


def to_dtype(pytree: PyTree, dtype: torch.dtype) -> PyTree:
    """
    Convert all tensors in a pytree to the specified dtype.
    
    Args:
        pytree: Nested structure containing tensors
        dtype: Target dtype (e.g., torch.float32, torch.float16)
    
    Returns:
        Pytree with all tensors converted to the specified dtype
    """
    def _to_dtype_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype)
        elif isinstance(obj, dict):
            return {key: _to_dtype_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_to_dtype_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_to_dtype_recursive(item) for item in obj)
        else:
            return obj
    
    return _to_dtype_recursive(pytree)


def detach_pytree(pytree: PyTree) -> PyTree:
    """
    Detach all tensors in a pytree from the computation graph.
    
    Args:
        pytree: Nested structure containing tensors
    
    Returns:
        Pytree with all tensors detached
    """
    def _detach_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return obj.detach()
        elif isinstance(obj, dict):
            return {key: _detach_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_detach_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_detach_recursive(item) for item in obj)
        else:
            return obj
    
    return _detach_recursive(pytree)


def clone_pytree(pytree: PyTree) -> PyTree:
    """
    Clone all tensors in a pytree.
    
    Args:
        pytree: Nested structure containing tensors
    
    Returns:
        Pytree with all tensors cloned
    """
    def _clone_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        elif isinstance(obj, dict):
            return {key: _clone_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_clone_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_clone_recursive(item) for item in obj)
        else:
            return obj
    
    return _clone_recursive(pytree)


def map_pytree(func: Callable[[torch.Tensor], torch.Tensor], pytree: PyTree) -> PyTree:
    """
    Apply a function to all tensors in a pytree.
    
    Args:
        func: Function to apply to each tensor
        pytree: Nested structure containing tensors
    
    Returns:
        Pytree with function applied to all tensors
    
    Example:
        >>> # Normalize all tensors
        >>> normalized = map_pytree(lambda x: (x - x.mean()) / x.std(), batch)
    """
    def _map_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return func(obj)
        elif isinstance(obj, dict):
            return {key: _map_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [_map_recursive(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(_map_recursive(item) for item in obj)
        else:
            return obj
    
    return _map_recursive(pytree)


def get_pytree_info(pytree: PyTree, name: str = "pytree") -> str:
    """
    Get information about tensors in a pytree (shapes, dtypes, devices).
    
    Args:
        pytree: Nested structure containing tensors
        name: Name prefix for the pytree
    
    Returns:
        String containing information about all tensors
    """
    info_lines = []
    
    def _get_info_recursive(obj, path: str):
        if isinstance(obj, torch.Tensor):
            info_lines.append(
                f"{path}: shape={obj.shape}, dtype={obj.dtype}, device={obj.device}, "
                f"requires_grad={obj.requires_grad}"
            )
        elif isinstance(obj, np.ndarray):
            info_lines.append(
                f"{path}: shape={obj.shape}, dtype={obj.dtype} (numpy array)"
            )
        elif isinstance(obj, dict):
            for key, value in obj.items():
                _get_info_recursive(value, f"{path}.{key}")
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                _get_info_recursive(item, f"{path}[{i}]")
    
    _get_info_recursive(pytree, name)
    return "\n".join(info_lines)


def count_parameters(pytree: PyTree) -> Dict[str, int]:
    """
    Count the number of parameters in tensors within a pytree.
    
    Args:
        pytree: Nested structure containing tensors
    
    Returns:
        Dictionary with total, trainable, and non-trainable parameter counts
    """
    total_params = 0
    trainable_params = 0
    
    def _count_recursive(obj):
        nonlocal total_params, trainable_params
        if isinstance(obj, torch.Tensor):
            num_params = obj.numel()
            total_params += num_params
            if obj.requires_grad:
                trainable_params += num_params
        elif isinstance(obj, dict):
            for value in obj.values():
                _count_recursive(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _count_recursive(item)
    
    _count_recursive(pytree)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_memory_usage(pytree: PyTree, device: Optional[Union[torch.device, str]] = None) -> Dict[str, float]:
    """
    Get memory usage of tensors in a pytree.
    
    Args:
        pytree: Nested structure containing tensors
        device: Specific device to check (if None, checks all tensors)
    
    Returns:
        Dictionary with memory usage in MB
    """
    if isinstance(device, str):
        device = torch.device(device)
    
    total_memory = 0.0
    
    def _get_memory_recursive(obj):
        nonlocal total_memory
        if isinstance(obj, torch.Tensor):
            if device is None or obj.device == device:
                # Calculate memory in bytes, then convert to MB
                memory_bytes = obj.numel() * obj.element_size()
                total_memory += memory_bytes / (1024 * 1024)  # Convert to MB
        elif isinstance(obj, dict):
            for value in obj.values():
                _get_memory_recursive(value)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _get_memory_recursive(item)
    
    _get_memory_recursive(pytree)
    
    return {
        'total_memory_mb': total_memory,
        'device': str(device) if device else 'all_devices'
    }


# Convenience functions for common operations
def batch_to_device(batch: Dict[str, torch.Tensor], device: Union[torch.device, str]) -> Dict[str, torch.Tensor]:
    """
    Convenience function to move a batch (dictionary of tensors) to device.
    This is a specialized version of to_device for the common case of batches.
    
    Args:
        batch: Dictionary containing tensors (typical ML batch format)
        device: Target device
    
    Returns:
        Batch with all tensors moved to device
    """
    return to_device(batch, device)


def print_batch_info(batch: Dict[str, Any], name: str = "batch") -> None:
    """
    Print information about a batch for debugging.
    
    Args:
        batch: Dictionary containing batch data
        name: Name to display for the batch
    """
    print(f"\n📊 {name.upper()} INFO:")
    print(get_pytree_info(batch, name))
    
    memory_info = get_memory_usage(batch)
    print(f"💾 Memory usage: {memory_info['total_memory_mb']:.2f} MB")


def validate_batch_format(batch: Dict[str, Any], expected_keys: Optional[List[str]] = None) -> bool:
    """
    Validate that a batch has the expected format.
    
    Args:
        batch: Batch to validate
        expected_keys: List of expected keys (if None, uses common ML keys)
    
    Returns:
        True if batch is valid, False otherwise
    """
    if expected_keys is None:
        expected_keys = ['image', 'label']  # Default for MNIST
    
    if not isinstance(batch, dict):
        print(f"❌ Batch should be a dictionary, got {type(batch)}")
        return False
    
    for key in expected_keys:
        if key not in batch:
            print(f"❌ Missing expected key '{key}' in batch")
            return False
        
        if not isinstance(batch[key], torch.Tensor):
            print(f"❌ Batch['{key}'] should be a tensor, got {type(batch[key])}")
            return False
    
    return True


# Additional utility functions for pytrees
def flatten_pytree(pytree: PyTree) -> Tuple[List[torch.Tensor], Callable]:
    """
    Flatten a pytree into a list of tensors and return a function to reconstruct it.
    
    Args:
        pytree: Nested structure to flatten
    
    Returns:
        Tuple of (flat_tensors, unflatten_func)
    """
    tensors = []
    
    def _extract_tensors(obj, path=[]):
        if isinstance(obj, torch.Tensor):
            tensors.append((obj, path))
        elif isinstance(obj, dict):
            for key, value in obj.items():
                _extract_tensors(value, path + [('dict', key)])
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                _extract_tensors(item, path + [('list', i)])
        elif isinstance(obj, tuple):
            for i, item in enumerate(obj):
                _extract_tensors(item, path + [('tuple', i)])
    
    _extract_tensors(pytree)
    
    flat_tensors = [tensor for tensor, _ in tensors]
    paths = [path for _, path in tensors]
    
    def unflatten_func(new_tensors):
        if len(new_tensors) != len(paths):
            raise ValueError(f"Expected {len(paths)} tensors, got {len(new_tensors)}")
        
        # Reconstruct the pytree
        result = {}
        for tensor, path in zip(new_tensors, paths):
            current = result
            for i, (container_type, key) in enumerate(path[:-1]):
                if key not in current:
                    next_type, _ = path[i + 1]
                    if next_type == 'dict':
                        current[key] = {}
                    elif next_type in ['list', 'tuple']:
                        current[key] = []
                    else:
                        current[key] = {}
                current = current[key]
            
            if path:
                _, final_key = path[-1]
                current[final_key] = tensor
        
        return result
    
    return flat_tensors, unflatten_func 