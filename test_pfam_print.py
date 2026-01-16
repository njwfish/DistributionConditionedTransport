#!/usr/bin/env python3
"""
Simple script to test PfamDataset and view family counts
"""

import sys
import os
import time

# Add the parent directory to path so we can import datasets
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.pfams import PfamDataset

if __name__ == "__main__":
    print("Instantiating PfamDataset...")
    print("=" * 60)
    
    # Create the dataset - set tokenize=True to force re-reading the data
    # Adjust data_dir if your pfam data is in a different location
    t1 = time.time()
    dataset = PfamDataset(
        data_dir='data/pfam',  # Adjust this path if needed
        data_file="pfam_tokenized_data_test_1000.pt",
        set_size=16,
        tokenize=True,  # Force re-tokenization to see the print output
        start_line=33233246,
        lines_to_read=10**9,
        max_pfams=1000,
        max_length=128,
    )
    t2 = time.time()
    print(f"Time taken: {t2 - t1} seconds")
    print("=" * 60)
    print(f"\nDataset created successfully!")
    print(f"Total items in dataset: {len(dataset)}")
