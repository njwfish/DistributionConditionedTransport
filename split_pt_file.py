"""
Script to split a .pt tokenized data file into train and test sets.
- Train: first 200 elements
- Test: remaining elements
"""

import torch
import argparse
import os


def split_pt_file(input_path: str, train_size: int = 200):
    """
    Load a .pt file and split it into train and test sets.
    
    Args:
        input_path: Path to the input .pt file
        train_size: Number of elements for the training set (default: 200)
    """
    # Load the data
    print(f"Loading {input_path}...")
    data = torch.load(input_path)
    
    total_elements = len(data)
    print(f"Total elements: {total_elements}")
    
    if total_elements <= train_size:
        print(f"Warning: Data has only {total_elements} elements, which is <= train_size ({train_size})")
        print("Train set will contain all elements, test set will be empty.")
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"Train set size: {len(train_data)}")
    print(f"Test set size: {len(test_data)}")
    
    # Generate output filenames
    base_name = input_path.rsplit('.pt', 1)[0]
    train_path = f"{base_name}_train.pt"
    test_path = f"{base_name}_test.pt"
    
    # Save the splits
    print(f"Saving train set to {train_path}...")
    torch.save(train_data, train_path)
    
    print(f"Saving test set to {test_path}...")
    torch.save(test_data, test_path)
    
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a .pt file into train and test sets")
    parser.add_argument("input_file", type=str, help="Path to the input .pt file")
    parser.add_argument("--train_size", type=int, default=200, 
                        help="Number of elements for the training set (default: 200)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File {args.input_file} does not exist")
        exit(1)
    
    if not args.input_file.endswith('.pt'):
        print(f"Warning: Input file does not have .pt extension")
    
    split_pt_file(args.input_file, args.train_size)
