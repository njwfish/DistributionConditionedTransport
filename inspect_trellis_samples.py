"""
Script to inspect the structure of self.samples in trellis_dataset
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.trellis import trellis_dataset
import numpy as np


def inspect_sample(sample, sample_idx):
    """Print information about a single sample from self.samples"""
    print(f"\n{'='*60}")
    print(f"Sample {sample_idx}:")
    print(f"{'='*60}")
    
    # Unpack the sample tuple (same structure as in __getitem__)
    culture, x0, x1, cell_cond, treat_cond, patient = sample
    
    print(f"\nculture: {repr(culture)}")
    print(f"  - Type: {type(culture).__name__}")
    if hasattr(culture, '__len__'):
        print(f"  - Length: {len(culture)}")
    
    print(f"\nx0 (source):")
    print(f"  - Type: {type(x0).__name__}")
    print(f"  - Shape: {x0.shape if isinstance(x0, np.ndarray) else 'N/A'}")
    print(f"  - Dtype: {x0.dtype if isinstance(x0, np.ndarray) else 'N/A'}")
    
    print(f"\nx1 (target):")
    print(f"  - Type: {type(x1).__name__}")
    print(f"  - Shape: {x1.shape if isinstance(x1, np.ndarray) else 'N/A'}")
    print(f"  - Dtype: {x1.dtype if isinstance(x1, np.ndarray) else 'N/A'}")
    
    print(f"\ncell_cond:")
    print(f"  - Type: {type(cell_cond).__name__}")
    print(f"  - Shape: {cell_cond.shape if isinstance(cell_cond, np.ndarray) else 'N/A'}")
    print(f"  - Dtype: {cell_cond.dtype if isinstance(cell_cond, np.ndarray) else 'N/A'}")
    
    print(f"\ntreat_cond:")
    print(f"  - Type: {type(treat_cond).__name__}")
    print(f"  - Shape: {treat_cond.shape if isinstance(treat_cond, np.ndarray) else 'N/A'}")
    print(f"  - Dtype: {treat_cond.dtype if isinstance(treat_cond, np.ndarray) else 'N/A'}")
    
    print(f"\npatient: {repr(patient)}")
    print(f"  - Type: {type(patient).__name__}")
    if hasattr(patient, '__len__'):
        print(f"  - Length: {len(patient)}")


def main():
    print("Initializing trellis_dataset...")
    
    # Create dataset instance with default parameters
    dataset = trellis_dataset(
        split_name='pdo21',
        split_mode='train',
        set_size=32,
        seed=0,
        ot_coupling=False
    )
    
    print(f"\nDataset initialized successfully!")
    print(f"Total number of samples in self.samples: {len(dataset.samples)}")
    print(f"Total dataset length (__len__): {len(dataset)} (samples^2 for all pairs)")
    
    # Inspect first 3 samples
    num_to_inspect = min(3, len(dataset.samples))
    
    for i in range(num_to_inspect):
        inspect_sample(dataset.samples[i], i)
    
    print(f"\n{'='*60}")
    print("Inspection complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
