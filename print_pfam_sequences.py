#!/usr/bin/env python3
"""
Simple script to load the PFAM dataset and print the first 10 sequences for each family.
"""

import torch
import argparse


def print_pfam_sequences(data_path: str = 'data/pfam/pfam_tokenized_data.pt', 
                         num_seqs: int = 10):
    """
    Load the PFAM dataset and print sequences for each family.
    
    Args:
        data_path: Path to the tokenized PFAM dataset
        num_seqs: Number of sequences to print per family
    """
    print(f"Loading dataset from {data_path}...")
    data = torch.load(data_path)
    print(f"Loaded {len(data)} PFAM families\n")
    print("=" * 80)
    
    for family_data in data:
        pfam = family_data['pfam']
        raw_texts = family_data['raw_texts']
        total_seqs = len(raw_texts)
        
        print(f"\nPFAM: {pfam}")
        print(f"Total sequences: {total_seqs}")
        print("-" * 80)
        
        # Print first num_seqs sequences
        for i, seq in enumerate(raw_texts[:num_seqs]):
            print(f"  [{i+1}/{min(num_seqs, total_seqs)}] Length: {len(seq):3d} | {seq}")
        
        if total_seqs > num_seqs:
            print(f"  ... ({total_seqs - num_seqs} more sequences not shown)")
        
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Print sequences from the PFAM dataset'
    )
    parser.add_argument(
        '--data_path', 
        type=str, 
        default='data/pfam/pfam_tokenized_data.pt',
        help='Path to the tokenized PFAM dataset'
    )
    parser.add_argument(
        '--num_seqs', 
        type=int, 
        default=10,
        help='Number of sequences to print per family'
    )
    args = parser.parse_args()
    
    print_pfam_sequences(args.data_path, args.num_seqs)


if __name__ == "__main__":
    main()
