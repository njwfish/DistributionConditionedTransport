#!/usr/bin/env python3
"""
Script to check all full_data_unit.tsv files for invalid DNA sequences.
Validates that all sequences in the 'sequence' column contain only A, C, T, G.
"""

import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


def is_valid_dna(sequence):
    """
    Check if a DNA sequence contains only valid nucleotides (A, C, T, G).
    
    Args:
        sequence: DNA sequence string
    
    Returns:
        Tuple (is_valid, invalid_chars) where:
        - is_valid: Boolean indicating if sequence is valid
        - invalid_chars: Set of invalid characters found (empty if valid)
    """
    if pd.isna(sequence) or sequence == '':
        return True, set()  # Empty/NaN sequences are considered valid
    
    # Find all characters that are NOT A, C, T, or G
    invalid_chars = set(re.findall(r'[^ACTG]', str(sequence).upper()))
    
    return len(invalid_chars) == 0, invalid_chars


def check_file(file_path):
    """
    Check a single TSV file for invalid DNA sequences and track sequence lengths.
    
    Args:
        file_path: Path to the TSV file
    
    Returns:
        Dictionary with results:
        - 'total_sequences': Total number of sequences
        - 'invalid_count': Number of invalid sequences
        - 'invalid_chars': Set of all invalid characters found
        - 'invalid_rows': List of (row_index, sequence, invalid_chars) tuples
        - 'min_length': Minimum sequence length
        - 'max_length': Maximum sequence length
        - 'min_length_row': Row index of shortest sequence
        - 'max_length_row': Row index of longest sequence
    """
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        return {'error': str(e)}
    
    if 'sequence' not in df.columns:
        return {'error': 'Column "sequence" not found'}
    
    results = {
        'total_sequences': len(df),
        'invalid_count': 0,
        'invalid_chars': set(),
        'invalid_rows': [],
        'min_length': float('inf'),
        'max_length': 0,
        'min_length_row': None,
        'max_length_row': None
    }
    
    # Check each sequence
    for idx, sequence in enumerate(df['sequence']):
        is_valid, invalid_chars = is_valid_dna(sequence)
        
        if not is_valid:
            results['invalid_count'] += 1
            results['invalid_chars'].update(invalid_chars)
            # Store only first 10 invalid sequences per file to avoid memory issues
            if len(results['invalid_rows']) < 10:
                results['invalid_rows'].append((idx, str(sequence)[:100], invalid_chars))
        
        # Track sequence length
        if pd.notna(sequence) and sequence != '':
            seq_len = len(str(sequence))
            if seq_len < results['min_length']:
                results['min_length'] = seq_len
                results['min_length_row'] = idx
            if seq_len > results['max_length']:
                results['max_length'] = seq_len
                results['max_length_row'] = idx
    
    return results


def find_full_data_files(root_dir):
    """
    Find all full_data_unit.tsv files in the directory structure.
    
    Args:
        root_dir: Root directory to search
    
    Returns:
        List of Path objects for all full_data_unit.tsv files
    """
    root_path = Path(root_dir)
    return sorted(root_path.rglob("full_data_unit.tsv"))


def extract_subject_time(file_path):
    """
    Extract subject (patient) ID and time from file path.
    
    Args:
        file_path: Path to the TSV file
    
    Returns:
        Tuple (subject_id, time) or (None, None) if not found
    """
    parts = file_path.parts
    subject_id = None
    time = None
    
    for part in parts:
        if part.startswith("subject="):
            subject_id = part.split("=")[1]
        elif part.startswith("time="):
            time = part.split("=")[1]
    
    return subject_id, time


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    tcr_data_dir = script_dir / "tcr_dataset" / "tcr_data"
    
    # Check if directory exists
    if not tcr_data_dir.exists():
        print(f"Error: Directory {tcr_data_dir} does not exist!")
        return
    
    print(f"Searching for full_data_unit.tsv files in: {tcr_data_dir}\n")
    
    # Find all files
    tsv_files = find_full_data_files(tcr_data_dir)
    
    if not tsv_files:
        print("No full_data_unit.tsv files found!")
        return
    
    print(f"Found {len(tsv_files)} files to check\n")
    
    # Statistics
    total_files_checked = 0
    total_files_with_errors = 0
    total_sequences = 0
    total_invalid_sequences = 0
    all_invalid_chars = set()
    files_with_issues = []
    
    # Track sequence lengths globally
    global_min_length = float('inf')
    global_max_length = 0
    min_seq_info = None  # (file_path, row, subject, time, length)
    max_seq_info = None  # (file_path, row, subject, time, length)
    
    # Check each file
    for file_path in tqdm(tsv_files, desc="Checking files"):
        results = check_file(file_path)
        
        if 'error' in results:
            print(f"\n⚠ Error reading {file_path.relative_to(tcr_data_dir)}: {results['error']}")
            continue
        
        total_files_checked += 1
        total_sequences += results['total_sequences']
        total_invalid_sequences += results['invalid_count']
        
        if results['invalid_count'] > 0:
            total_files_with_errors += 1
            all_invalid_chars.update(results['invalid_chars'])
            files_with_issues.append((file_path, results))
        
        # Track global min/max sequence lengths
        subject_id, time = extract_subject_time(file_path)
        
        if results['min_length'] < global_min_length:
            global_min_length = results['min_length']
            min_seq_info = (file_path, results['min_length_row'], subject_id, time, results['min_length'])
        
        if results['max_length'] > global_max_length:
            global_max_length = results['max_length']
            max_seq_info = (file_path, results['max_length_row'], subject_id, time, results['max_length'])
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total files checked: {total_files_checked}")
    print(f"Total sequences checked: {total_sequences:,}")
    print(f"Files with invalid sequences: {total_files_with_errors}")
    print(f"Total invalid sequences: {total_invalid_sequences:,}")
    
    # Print sequence length statistics
    print(f"\n{'='*70}")
    print("SEQUENCE LENGTH STATISTICS")
    print(f"{'='*70}")
    
    if min_seq_info:
        file_path, row, subject, time, length = min_seq_info
        print(f"\nShortest sequence:")
        print(f"  Length: {length:,} bp")
        print(f"  Patient: {subject}")
        print(f"  Time: {time}")
        print(f"  Row: {row} (0-indexed, row {row + 1} in file including header)")
        print(f"  File: {file_path.relative_to(tcr_data_dir)}")
    
    if max_seq_info:
        file_path, row, subject, time, length = max_seq_info
        print(f"\nLongest sequence:")
        print(f"  Length: {length:,} bp")
        print(f"  Patient: {subject}")
        print(f"  Time: {time}")
        print(f"  Row: {row} (0-indexed, row {row + 1} in file including header)")
        print(f"  File: {file_path.relative_to(tcr_data_dir)}")
    
    if total_invalid_sequences > 0:
        print(f"\n{'='*70}")
        print("INVALID SEQUENCES")
        print(f"{'='*70}")
        print(f"\nInvalid characters found: {sorted(all_invalid_chars)}")
        print(f"\n{'='*70}")
        print("FILES WITH ISSUES")
        print(f"{'='*70}")
        
        for file_path, results in files_with_issues:
            rel_path = file_path.relative_to(tcr_data_dir)
            print(f"\n📁 {rel_path}")
            print(f"   Invalid sequences: {results['invalid_count']:,} / {results['total_sequences']:,} "
                  f"({100 * results['invalid_count'] / results['total_sequences']:.2f}%)")
            print(f"   Invalid characters: {sorted(results['invalid_chars'])}")
            
            if results['invalid_rows']:
                print(f"   First few invalid sequences:")
                for idx, seq, chars in results['invalid_rows'][:3]:
                    print(f"      Row {idx}: {seq[:80]}... (invalid: {sorted(chars)})")
    else:
        print(f"\n{'='*70}")
        print("✓ All sequences are valid! All contain only A, C, T, G.")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()

