#!/usr/bin/env python3

import argparse
import sys
from typing import Optional, List, Set, Dict

import torch


def normalize_sequence(sequence: str) -> str:
    """Normalize a sequence by stripping whitespace and converting to uppercase."""
    return sequence.strip().upper()


def process_dataset_simple(input_path: str, limit: Optional[int] = None) -> None:
    """
    Process the dataset and count completely new sequences at each time point.
    Also check for sequences with issues: containing X, wrong length, or non-standard amino acids.
    
    A sequence is considered "new" if it hasn't been seen at any previous time point.
    """
    # Load the dataset
    dataset = torch.load(input_path, map_location="cpu")
    if not isinstance(dataset, list):
        raise TypeError(f"Expected dataset to be a list, got {type(dataset)}")
    
    # Define the 20 standard amino acids
    STANDARD_AMINO_ACIDS = set('ARNDCQEGHILKMFPSTWYV')
    
    # Keep track of all sequences we've seen so far
    seen_sequences: Set[str] = set()
    
    # Track sequences with issues
    sequences_with_x: Set[str] = set()
    sequences_wrong_length: Dict[str, int] = {}  # sequence -> its length
    sequences_non_standard_aa: Dict[str, Set[str]] = {}  # sequence -> set of non-standard amino acids
    
    # Determine how many items to process
    max_items = len(dataset) if limit is None else min(limit, len(dataset))
    
    print("Processing dataset...")
    print("Format: time_point | total_sequences | new_sequences | cumulative_unique")
    print("-" * 70)
    
    for i in range(max_items):
        item = dataset[i]
        
        # Handle non-dict items gracefully
        if not isinstance(item, dict):
            print(f"{i:8} | skipping non-dict item")
            continue
            
        # Get time and sequences from the item
        time_value = item.get("time", "")
        time_str = str(time_value) if time_value is not None else f"item_{i}"
        
        raw_texts = item.get("raw_texts", [])
        if not isinstance(raw_texts, (list, tuple)):
            raw_texts = []
        
        # Filter to only string sequences and normalize them
        sequences_this_timepoint = []
        for seq in raw_texts:
            if isinstance(seq, str):
                normalized = normalize_sequence(seq)
                if normalized:  # Skip empty sequences
                    sequences_this_timepoint.append(normalized)
                    
                    # Check for sequences containing X
                    if 'X' in normalized:
                        sequences_with_x.add(normalized)
                    
                    # Check for sequences with wrong length
                    if len(normalized) != 1000:
                        sequences_wrong_length[normalized] = len(normalized)
                    
                    # Check for non-standard amino acids
                    seq_chars = set(normalized)
                    non_standard = seq_chars - STANDARD_AMINO_ACIDS
                    if non_standard:
                        sequences_non_standard_aa[normalized] = non_standard
        
        # Count how many sequences are completely new (not seen before)
        new_sequences = 0
        for seq in sequences_this_timepoint:
            if seq not in seen_sequences:
                new_sequences += 1
                seen_sequences.add(seq)
        
        total_this_timepoint = len(sequences_this_timepoint)
        cumulative_unique = len(seen_sequences)
        
        # Print results
        print(f"{time_str:8} | {total_this_timepoint:15} | {new_sequences:13} | {cumulative_unique:17}")
    
    # Print summary of sequence issues
    print("\n" + "=" * 70)
    print("SEQUENCE QUALITY SUMMARY")
    print("=" * 70)
    
    print(f"\nSequences containing 'X': {len(sequences_with_x)}")
    if sequences_with_x:
        print("Examples (showing first 3):")
        for i, seq in enumerate(sorted(sequences_with_x)):
            if i >= 3:
                break
            x_positions = [j for j, char in enumerate(seq) if char == 'X']
            print(f"  - Length {len(seq)}, X at positions: {x_positions}")
    
    print(f"\nSequences with length != 1000: {len(sequences_wrong_length)}")
    if sequences_wrong_length:
        length_counts = {}
        for seq, length in sequences_wrong_length.items():
            length_counts[length] = length_counts.get(length, 0) + 1
        print("Length distribution:")
        for length in sorted(length_counts.keys()):
            print(f"  - Length {length}: {length_counts[length]} sequences")
    
    print(f"\nSequences with non-standard amino acids: {len(sequences_non_standard_aa)}")
    if sequences_non_standard_aa:
        all_non_standard = set()
        for non_standard_chars in sequences_non_standard_aa.values():
            all_non_standard.update(non_standard_chars)
        print(f"Non-standard characters found: {sorted(all_non_standard)}")
        print("Examples (showing first 3):")
        for i, (seq, non_standard_chars) in enumerate(sorted(sequences_non_standard_aa.items())):
            if i >= 3:
                break
            print(f"  - Length {len(seq)}, contains: {sorted(non_standard_chars)}")
    
    print(f"\nTotal unique sequences processed: {len(seen_sequences)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Count completely new protein sequences at each time point and check for sequence quality issues (X characters, wrong length, non-standard amino acids)"
    )
    parser.add_argument(
        "input_path",
        help="Path to the .pt dataset file"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Process only the first N time points"
    )
    
    args = parser.parse_args()
    
    process_dataset_simple(
        input_path=args.input_path,
        limit=args.max_items
    )


if __name__ == "__main__":
    main()
