#!/usr/bin/env python
"""
Test script to verify train_predictor_bool functionality in SnapMMDUnified dataset.
This script instantiates the GoM dataset and prints train_predictor_bool, source_idx, 
and target_idx for all possible pairs.
"""

import sys
import os
import numpy as np

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.snapMMD_unified import SnapMMDUnified

def main():
    # Instantiate the dataset for GoM
    print("Instantiating SnapMMDUnified dataset for GoM...")
    dataset = SnapMMDUnified(
        dataset_name='GoM',
        testing_method='forecast',
        seed=42,
        set_size=32,
        ot_coupling=False
    )
    
    print(f"Dataset length (total pairs): {len(dataset)}")
    print(f"Number of time points: {dataset.data.shape[0]}")
    print(f"\n{'Index':<8} {'Source':<8} {'Target':<8} {'Train Predictor':<18} {'Delta (T-S)'}")
    print("-" * 70)
    
    # Track statistics
    train_predictor_count = 0
    
    # Iterate through all elements
    for idx in range(len(dataset)):
        # Get the item (this will randomly sample, but we only care about the indices)
        item = dataset[idx]
        
        source_idx = item['source_idx']
        target_idx = item['target_idx']
        train_predictor_bool = item['train_predictor_bool']
        delta = target_idx - source_idx
        
        # Track count
        if train_predictor_bool:
            train_predictor_count += 1
        
        # Print the information
        print(f"{idx:<8} {source_idx:<8} {target_idx:<8} {str(train_predictor_bool):<18} {delta}")
    
    print("-" * 70)
    print(f"\nSummary:")
    print(f"Total pairs: {len(dataset)}")
    print(f"Pairs with train_predictor_bool=True: {train_predictor_count}")
    print(f"Pairs with train_predictor_bool=False: {len(dataset) - train_predictor_count}")
    print(f"\nNote: train_predictor_bool is True when (target_idx - source_idx) == 1")
    print(f"      This means only consecutive time step pairs are used for training.")

if __name__ == "__main__":
    main()

