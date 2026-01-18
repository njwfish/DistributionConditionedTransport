#!/usr/bin/env python3
"""
Script to load a .pt file containing pfam data, print the number of pfams,
and optionally save a subset containing only the first N elements.

Usage:
    python subset_pfam_data.py <input_file.pt> [N]
    
Examples:
    # Just print the number of pfams
    python subset_pfam_data.py data/pfam/pfam_tokenized_data.pt
    
    # Create a subset with first 100 pfams
    python subset_pfam_data.py data/pfam/pfam_tokenized_data.pt 100
"""

import argparse
import os
import sys
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='Load a pfam .pt file, print the number of pfams, and optionally save a subset.'
    )
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the input .pt file'
    )
    parser.add_argument(
        'N',
        type=int,
        nargs='?',
        default=None,
        help='Number of pfams to keep in the subset (optional)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        sys.exit(1)
    
    # Load the .pt file
    print(f"Loading {args.input_file}...")
    try:
        data = torch.load(args.input_file, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        sys.exit(1)
    
    # Check that data is a list
    if not isinstance(data, list):
        print(f"Error: Expected data to be a list, but got {type(data).__name__}")
        sys.exit(1)
    
    # Print the number of pfams
    num_pfams = len(data)
    print(f"Number of pfams: {num_pfams}")
    
    # If N is provided, save a subset
    if args.N is not None:
        if args.N <= 0:
            print(f"Error: N must be positive, got {args.N}")
            sys.exit(1)
        
        if args.N > num_pfams:
            print(f"Warning: N ({args.N}) is greater than the number of pfams ({num_pfams}).")
            print(f"Saving all {num_pfams} pfams instead.")
            subset_size = num_pfams
        else:
            subset_size = args.N
        
        # Create subset
        subset_data = data[:subset_size]
        
        # Generate output filename
        input_path = Path(args.input_file)
        output_filename = f"{input_path.stem}_{subset_size}{input_path.suffix}"
        output_path = input_path.parent / output_filename
        
        # Save the subset
        print(f"Saving subset with {subset_size} pfams to {output_path}...")
        try:
            torch.save(subset_data, output_path)
            print(f"Successfully saved subset to {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
