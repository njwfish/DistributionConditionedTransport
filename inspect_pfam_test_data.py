#!/usr/bin/env python
"""
Simple script to inspect the pfam_tokenized_data_test.pt file.
Prints overview statistics about the data.
"""

import torch
import os

def inspect_pfam_data(file_path):
    """Load and inspect the tokenized Pfam data."""
    
    print(f"Loading data from: {file_path}")
    print("=" * 60)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    # Load the data
    data = torch.load(file_path, weights_only=False)
    
    # Basic statistics
    print(f"\nNumber of Pfam families: {len(data)}")
    print("\nData structure overview:")
    print("-" * 60)
    
    if len(data) > 0:
        # Examine first element
        first_elem = data[0]
        print(f"\nFirst element keys: {list(first_elem.keys())}")
        print(f"  - pfam: {first_elem.get('pfam', 'N/A')}")
        
        # Check samples structure
        if 'samples' in first_elem:
            samples = first_elem['samples']
            print(f"\n  - samples keys: {list(samples.keys())}")
            
            # Print shapes of tokenized data
            for key in samples.keys():
                shape = samples[key].shape
                dtype = samples[key].dtype
                print(f"      {key}: shape={shape}, dtype={dtype}")
        
        # Check raw texts
        if 'raw_texts' in first_elem:
            num_texts = len(first_elem['raw_texts'])
            print(f"\n  - raw_texts: {num_texts} sequences")
            if num_texts > 0:
                first_seq = first_elem['raw_texts'][0]
                print(f"      Example (first 80 chars): {first_seq[:80]}...")
    
    # Statistics across all families
    print("\n" + "=" * 60)
    print("Statistics across all families:")
    print("-" * 60)
    
    family_names = []
    num_sequences = []
    seq_lengths = []
    
    for elem in data:
        pfam = elem.get('pfam', 'unknown')
        family_names.append(pfam)
        
        if 'samples' in elem:
            # Number of sequences in this family
            num_seqs = elem['samples']['esm_input_ids'].shape[0]
            num_sequences.append(num_seqs)
            
            # Sequence length (from attention mask)
            seq_len = elem['samples']['esm_input_ids'].shape[1]
            seq_lengths.append(seq_len)
    
    print(f"\nFamily names: {family_names[:10]}{'...' if len(family_names) > 10 else ''}")
    
    if num_sequences:
        print(f"\nSequences per family:")
        print(f"  Min: {min(num_sequences)}")
        print(f"  Max: {max(num_sequences)}")
        print(f"  Mean: {sum(num_sequences)/len(num_sequences):.1f}")
        print(f"  Total sequences: {sum(num_sequences)}")
    
    if seq_lengths:
        print(f"\nMax sequence length (tokenized): {seq_lengths[0]}")
    
    # Show distribution of families
    print("\n" + "=" * 60)
    print("Detailed family breakdown:")
    print("-" * 60)
    for i, elem in enumerate(data):
        pfam = elem.get('pfam', f'family_{i}')
        if 'samples' in elem:
            num_seqs = elem['samples']['esm_input_ids'].shape[0]
            print(f"  [{i}] {pfam}: {num_seqs} sequences")
    
    print("\n" + "=" * 60)
    print("Summary complete!")
    print("=" * 60)


if __name__ == '__main__':
    # Default path - can be changed as needed
    data_file = 'data/pfam/pfam_tokenized_data_10000.pt'
    
    # Get absolute path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_file)
    
    inspect_pfam_data(data_path)
