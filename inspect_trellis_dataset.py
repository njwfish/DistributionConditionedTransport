#!/usr/bin/env python3
"""
Script to instantiate trellis_dataset and inspect the first element
"""

import sys
import os
import numpy as np
import torch

# Add the project root to path
sys.path.insert(0, '/orcd/data/omarabu/001/paolo/CoupledDistributionEmbeddings')

from datasets.mfm_trellis import trellis_dataset

def print_separator(title=""):
    """Print a visual separator"""
    print("\n" + "="*80)
    if title:
        print(f" {title}")
        print("="*80)
    print()

def main():
    print_separator("INSTANTIATING TRELLIS_DATASET")
    
    # Instantiate the dataset with default parameters
    dataset = trellis_dataset(
        split_name='pdo21',  # default split
        set_size=32,
        seed=0
    )
    
    print_separator("DATASET INFORMATION")
    print(f"Number of base samples: {dataset.num_samples}")
    print(f"Dataset length (pairs): {len(dataset)} = {dataset.num_samples}^2")
    print(f"Split name: {dataset.split_name}")
    print(f"Control treatments: {dataset.control}")
    print(f"Treatments: {dataset.treatment}")
    print(f"Cultures: {dataset.culture}")
    print(f"Cell types: {dataset.cell_type}")
    print(f"\nNote: The dataset now creates all possible pairs of (source, target) samples.")
    print(f"Linear index i maps to pair (i // n, i % n) where n = num_samples")
    
    print_separator("FIRST ELEMENT - DETAILED INFORMATION")
    
    # Get the first element (now returns a dictionary with torch tensors)
    first_element = dataset[0]
    
    # Extract from dictionary (converting torch tensors to numpy for inspection)
    x0 = first_element['source_samples']
    x1 = first_element['target_samples']
    cell_cond = first_element['cell_cond']
    treat_cond = first_element['treat_cond']
    patient = first_element['patient']
    culture = first_element['culture']
    source_idx = first_element['source_idx']
    target_idx = first_element['target_idx']
    
    # Check if tensors are torch tensors and convert to numpy for display
    is_torch = isinstance(x0, torch.Tensor)
    if is_torch:
        x0_np = x0.numpy()
        x1_np = x1.numpy()
        cell_cond_np = cell_cond.numpy()
        treat_cond_np = treat_cond.numpy()
    else:
        x0_np = x0
        x1_np = x1
        cell_cond_np = cell_cond
        treat_cond_np = treat_cond
    
    print_separator("Element Components")
    print(f"Linear Index: 0")
    print(f"Source Index: {source_idx}")
    print(f"Target Index: {target_idx}")
    print(f"Patient: {patient}")
    print(f"Culture: {culture}")
    print(f"Data type: {'torch.Tensor' if is_torch else 'numpy.ndarray'}")
    
    print_separator("Source Distribution (x0)")
    print(f"Shape: {x0_np.shape}")
    print(f"Data type: {x0_np.dtype}")
    print(f"Min value: {x0_np.min():.6f}")
    print(f"Max value: {x0_np.max():.6f}")
    print(f"Mean: {x0_np.mean():.6f}")
    print(f"Std: {x0_np.std():.6f}")
    print(f"\nFirst 3 samples (first 10 features):")
    print(x0_np[:3, :10])
    
    print_separator("Target Distribution (x1)")
    print(f"Shape: {x1_np.shape}")
    print(f"Data type: {x1_np.dtype}")
    print(f"Min value: {x1_np.min():.6f}")
    print(f"Max value: {x1_np.max():.6f}")
    print(f"Mean: {x1_np.mean():.6f}")
    print(f"Std: {x1_np.std():.6f}")
    print(f"\nFirst 3 samples (first 10 features):")
    print(x1_np[:3, :10])
    
    print_separator("Cell Condition (one-hot encoding)")
    print(f"Shape: {cell_cond_np.shape}")
    print(f"Data type: {cell_cond_np.dtype}")
    print(f"Cell types: {dataset.cell_type}")
    print(f"\nFirst 5 samples:")
    print(cell_cond_np[:5])
    print(f"\nCell type distribution:")
    for i, cell_type in enumerate(dataset.cell_type):
        count = np.sum(cell_cond_np[:, i])
        print(f"  {cell_type}: {int(count)} cells")
    
    print_separator("Treatment Condition (one-hot encoding)")
    print(f"Shape: {treat_cond_np.shape}")
    print(f"Data type: {treat_cond_np.dtype}")
    print(f"Treatments: {dataset.treatment}")
    print(f"\nFirst 3 samples:")
    print(treat_cond_np[:3])
    treatment_idx = np.argmax(treat_cond_np[0])
    print(f"\nTreatment applied: {dataset.treatment[treatment_idx]}")
    
    print_separator("INDEXING SCHEME EXAMPLES")
    print("Showing how linear indices map to (source_idx, target_idx) pairs:\n")
    n = len(dataset.samples)
    example_indices = [0, 1, n, n+1, len(dataset)-1]
    for linear_idx in example_indices:
        if linear_idx < len(dataset):
            i = linear_idx // n
            j = linear_idx % n
            print(f"  Linear index {linear_idx:4d} -> pair ({i:2d}, {j:2d})")
    
    print_separator("SUMMARY")
    print(f"Total base samples: {len(dataset.samples)}")
    print(f"Total dataset size (pairs): {len(dataset)} = {len(dataset.samples)}^2")
    print(f"Set size (subsampled): {dataset.set_size}")
    print(f"\nThis element (index 0):")
    print(f"  Source sample index: {source_idx}")
    print(f"  Target sample index: {target_idx}")
    print(f"  Total cells in source (x0): {x0_np.shape[0]} (subsampled from original)")
    print(f"  Total cells in target (x1): {x1_np.shape[0]} (subsampled from original)")
    print(f"  Feature dimension: {x0_np.shape[1]}")
    print(f"  Patient: {patient}")
    print(f"  Culture type: {culture}")
    print(f"  Treatment: {dataset.treatment[treatment_idx]}")
    print(f"  Returned as: {'torch.Tensor' if is_torch else 'numpy.ndarray'}")
    print()

if __name__ == "__main__":
    main()

