#!/usr/bin/env python3
"""
Script to count total lines and unique Pfam families in Pfam-A.fasta.gz
"""
import gzip
import os

def count_pfam_stats(filepath='data/pfam/Pfam-A-filtered.fasta.gz'):
    """Count total lines and unique Pfam families in the gzipped FASTA file."""
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return
    
    total_lines = 0
    pfam_families = set()
    sequence_count = 0
    
    print(f"Reading {filepath}...")
    
    with gzip.open(filepath, 'rt') as f:
        for line in f:
            total_lines += 1
            
            if line.startswith('>'):
                # Extract pfam family from header (same logic as pfams_random.py)
                # Format: >... PF00001.1;
                fam = line.split()[-1].split(';')[0]
                pfam_families.add(fam)
                sequence_count += 1
            
            # Progress indicator every 10 million lines
            if total_lines % 10_000_000 == 0:
                print(f"  Processed {total_lines:,} lines, {len(pfam_families):,} unique families so far...")
    
    print("\n" + "="*50)
    print("PFAM FILE STATISTICS")
    print("="*50)
    print(f"Total lines:           {total_lines:,}")
    print(f"Total sequences:       {sequence_count:,}")
    print(f"Unique Pfam families:  {len(pfam_families):,}")
    print("="*50)

if __name__ == '__main__':
    count_pfam_stats()
