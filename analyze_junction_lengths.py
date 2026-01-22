#!/usr/bin/env python3
"""
Script to analyze junction_aa sequence lengths from TCR dataset TSV files.
Computes statistics (min, max, median, mean) per file and overall,
and plots a histogram of sequence length distribution.
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_junction_lengths(base_dir="tcr_dataset/tcr_data"):
    """
    Load all .tsv files from base_dir (recursively), extract junction_aa column,
    and compute statistics on sequence lengths.
    """
    # Find all .tsv files recursively
    tsv_pattern = os.path.join(base_dir, "**", "*.tsv")
    tsv_files = glob.glob(tsv_pattern, recursive=True)
    
    if not tsv_files:
        print(f"No .tsv files found in {base_dir}")
        return
    
    print(f"Found {len(tsv_files)} TSV files\n")
    print("=" * 80)
    
    # Store all lengths for overall statistics
    all_lengths = []
    file_stats = []
    
    for tsv_file in sorted(tsv_files):
        try:
            # Load TSV file
            df = pd.read_csv(tsv_file, sep='\t', usecols=['junction_aa'])
            
            # Drop any NaN values and convert to string
            sequences = df['junction_aa'].dropna().astype(str)
            
            # Compute lengths
            lengths = sequences.str.len().values
            
            if len(lengths) == 0:
                print(f"File: {tsv_file}")
                print("  No valid sequences found\n")
                continue
            
            # Compute statistics for this file
            min_len = int(np.min(lengths))
            max_len = int(np.max(lengths))
            median_len = float(np.median(lengths))
            mean_len = float(np.mean(lengths))
            
            # Store for later
            all_lengths.extend(lengths)
            file_stats.append({
                'file': tsv_file,
                'count': len(lengths),
                'min': min_len,
                'max': max_len,
                'median': median_len,
                'mean': mean_len
            })
            
            # Print per-file statistics
            relative_path = os.path.relpath(tsv_file, base_dir)
            print(f"File: {relative_path}")
            print(f"  Sequences: {len(lengths):,}")
            print(f"  Min length:    {min_len}")
            print(f"  Max length:    {max_len}")
            print(f"  Median length: {median_len:.1f}")
            print(f"  Mean length:   {mean_len:.2f}")
            print()
            
        except Exception as e:
            print(f"Error processing {tsv_file}: {e}\n")
    
    # Overall statistics
    if all_lengths:
        all_lengths = np.array(all_lengths)
        
        print("=" * 80)
        print("OVERALL STATISTICS (across all files)")
        print("=" * 80)
        print(f"Total files processed: {len(file_stats)}")
        print(f"Total sequences:       {len(all_lengths):,}")
        print(f"Min length:            {int(np.min(all_lengths))}")
        print(f"Max length:            {int(np.max(all_lengths))}")
        print(f"Median length:         {np.median(all_lengths):.1f}")
        print(f"Mean length:           {np.mean(all_lengths):.2f}")
        print(f"Std deviation:         {np.std(all_lengths):.2f}")
        print()
        
        # Plot histogram
        plt.figure(figsize=(12, 6))
        
        # Compute histogram bins
        min_len = int(np.min(all_lengths))
        max_len = int(np.max(all_lengths))
        bins = np.arange(min_len, max_len + 2) - 0.5  # Center bins on integers
        
        plt.hist(all_lengths, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
        
        plt.xlabel('Sequence Length (amino acids)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of junction_aa Sequence Lengths\n(TCR Dataset)', fontsize=14)
        
        # Add vertical lines for mean and median
        mean_val = np.mean(all_lengths)
        median_val = np.median(all_lengths)
        
        plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {mean_val:.1f}')
        plt.axvline(median_val, color='green', linestyle='-.', linewidth=2, 
                    label=f'Median: {median_val:.1f}')
        
        plt.legend(fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        # Add text box with summary stats
        textstr = f'n = {len(all_lengths):,}\nMin = {min_len}\nMax = {max_len}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.95, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
                       verticalalignment='top', horizontalalignment='right', bbox=props)
        
        plt.tight_layout()
        
        # Save figure
        output_file = 'junction_aa_length_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Histogram saved to: {output_file}")
        
        plt.close()
        
    return all_lengths, file_stats


if __name__ == "__main__":
    analyze_junction_lengths()
