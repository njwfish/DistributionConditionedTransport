#!/usr/bin/env python3
"""
Script to recursively inspect the structure of a PyTorch file.
Loads the file and prints out types, keys, and shapes of the first element.
"""

import torch
import numpy as np
from typing import Any, Dict, List, Tuple, Union


def get_shape_info(obj: Any) -> str:
    """Get shape information for tensor-like objects."""
    if hasattr(obj, 'shape'):
        return f"shape: {obj.shape}"
    elif hasattr(obj, '__len__'):
        try:
            length = len(obj)
            return f"length: {length}"
        except:
            return "length: unknown"
    else:
        return "no shape/length info"


def recursive_inspect(obj: Any, prefix: str = "", max_depth: int = 10, current_depth: int = 0) -> None:
    """
    Recursively inspect an object and print its structure.
    
    Args:
        obj: Object to inspect
        prefix: String prefix for indentation
        max_depth: Maximum recursion depth
        current_depth: Current recursion depth
    """
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth {max_depth} reached)")
        return
    
    obj_type = type(obj).__name__
    
    # Handle different types of objects
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        print(f"{prefix}Type: {obj_type}, {get_shape_info(obj)}, dtype: {obj.dtype}")
        
    elif isinstance(obj, dict):
        print(f"{prefix}Type: {obj_type}, {get_shape_info(obj)}")
        for key, value in obj.items():
            print(f"{prefix}  Key: '{key}'")
            recursive_inspect(value, prefix + "    ", max_depth, current_depth + 1)
            
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}Type: {obj_type}, {get_shape_info(obj)}")
        if len(obj) > 0:
            print(f"{prefix}  First element:")
            recursive_inspect(obj[0], prefix + "    ", max_depth, current_depth + 1)
        if len(obj) > 1:
            print(f"{prefix}  ... and {len(obj) - 1} more elements of similar structure")
            
    elif isinstance(obj, (int, float, str, bool)):
        print(f"{prefix}Type: {obj_type}, Value: {obj}")
        
    else:
        # For other types, try to get basic info
        print(f"{prefix}Type: {obj_type}, {get_shape_info(obj)}")
        
        # If it has attributes that look like data containers, explore them
        if hasattr(obj, '__dict__'):
            attrs = [attr for attr in dir(obj) if not attr.startswith('_')]
            if attrs:
                print(f"{prefix}  Attributes: {attrs[:5]}{'...' if len(attrs) > 5 else ''}")


def main():
    """Main function to load and inspect the PyTorch file."""
    file_path = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/data/spikeprot0430/virus_tokenized_data_from_gde.pt"
    
    print(f"Loading PyTorch file: {file_path}")
    print("=" * 60)
    
    try:
        # Load the PyTorch file
        data = torch.load(file_path, map_location='cpu')
        print(f"Successfully loaded file!")
        print(f"Root object type: {type(data).__name__}")
        print("=" * 60)
        
        # If it's a list or tuple, inspect the first element
        if isinstance(data, (list, tuple)) and len(data) > 0:
            print(f"Container has {len(data)} elements. Inspecting first element:")
            print("-" * 40)
            recursive_inspect(data[0])
        else:
            print("Inspecting root object:")
            print("-" * 40)
            recursive_inspect(data)
            
    except Exception as e:
        print(f"Error loading file: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
