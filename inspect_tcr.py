import pandas as pd
from collections import Counter
import glob
import os

# Standard amino acids
standard_amino_acids = set('ACDEFGHIKLMNPQRSTVWY')

def process_file(file_path):
    """Process a single TSV file and return statistics."""
    try:
        df = pd.read_csv(file_path, sep='\t')
        
        # Get the junction_aa column
        junction_aa = df['junction_aa']
        
        # Total number of elements before filtering
        total_before = len(junction_aa)
        
        # Filter to only keep sequences with standard amino acids
        junction_aa_filtered = junction_aa[junction_aa.apply(lambda x: set(str(x)).issubset(standard_amino_acids))]
        
        # Calculate statistics
        filtered_count = total_before - len(junction_aa_filtered)
        unique_count = junction_aa_filtered.nunique()
        
        # Sequence length statistics
        lengths = junction_aa_filtered.str.len()
        
        if len(lengths) > 0:
            min_len = lengths.min()
            max_len = lengths.max()
            median_len = lengths.median()
            mean_len = lengths.mean()
        else:
            min_len = max_len = median_len = mean_len = None
        
        # Top 5 most common elements
        counts = Counter(junction_aa_filtered)
        top_5 = counts.most_common(5)
        
        return {
            'total_before': total_before,
            'filtered_count': filtered_count,
            'total_after': len(junction_aa_filtered),
            'unique_count': unique_count,
            'min_len': min_len,
            'max_len': max_len,
            'median_len': median_len,
            'mean_len': mean_len,
            'top_5': top_5
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Find all TSV files recursively
tsv_files = glob.glob("tcr_dataset/**/*.tsv", recursive=True)
print(f"Found {len(tsv_files)} TSV files\n")

# Track overall min and max
overall_min = float('inf')
overall_max = float('-inf')

# Process each file
for file_path in sorted(tsv_files):
    print("=" * 80)
    print(f"File: {file_path}")
    print("=" * 80)
    
    stats = process_file(file_path)
    
    if stats:
        print(f"Total number of elements (before filtering): {stats['total_before']}")
        print(f"Sequences removed by filtering: {stats['filtered_count']}")
        print(f"Total number of elements (after filtering): {stats['total_after']}")
        print(f"Number of unique elements: {stats['unique_count']}")
        
        if stats['min_len'] is not None:
            print(f"\nSequence length statistics:")
            print(f"  Min: {stats['min_len']}")
            print(f"  Max: {stats['max_len']}")
            print(f"  Median: {stats['median_len']}")
            print(f"  Mean: {stats['mean_len']:.2f}")
            
            # Update overall min/max
            overall_min = min(overall_min, stats['min_len'])
            overall_max = max(overall_max, stats['max_len'])
        
        print(f"\nTop 5 most common elements:")
        for element, count in stats['top_5']:
            print(f"  {element}: {count}")
    
    print()

# Print overall statistics
print("=" * 80)
print("OVERALL STATISTICS (across all files)")
print("=" * 80)
if overall_min != float('inf'):
    print(f"Shortest sequence length (after filtering): {overall_min}")
    print(f"Longest sequence length (after filtering): {overall_max}")
else:
    print("No valid sequences found across all files")