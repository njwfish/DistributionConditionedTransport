#!/usr/bin/env python3
"""
Script to concatenate TSV files within each subject/time/repertoire directory.
For each subject/time combination, concatenates all TSV files from the repertoire
subdirectory, saves the result as full_data_unit.tsv in the subject/time directory,
and removes the repertoire subdirectory.
"""

import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm


def find_repertoire_directories(root_dir):
    """
    Find all repertoire directories within subject/time directories.
    
    Args:
        root_dir: Root directory to search (tcr_dataset/tcr_data)
    
    Returns:
        List of tuples (repertoire_dir, output_dir) where:
        - repertoire_dir: Path to the repertoire directory containing TSV files
        - output_dir: Path to the subject/time directory where output should be saved
    """
    root_path = Path(root_dir)
    repertoire_dirs = []
    
    # Find all directories matching the pattern subject=*/time=*/repertoire=*
    for subject_dir in sorted(root_path.glob("subject=*")):
        if not subject_dir.is_dir():
            continue
        
        for time_dir in sorted(subject_dir.glob("time=*")):
            if not time_dir.is_dir():
                continue
            
            # Find repertoire subdirectories
            for repertoire_dir in sorted(time_dir.glob("repertoire=*")):
                if repertoire_dir.is_dir():
                    repertoire_dirs.append((repertoire_dir, time_dir))
    
    return repertoire_dirs


def concatenate_and_cleanup(repertoire_dir, output_dir):
    """
    Concatenate all TSV files in repertoire_dir, save to output_dir, and remove repertoire_dir.
    
    Args:
        repertoire_dir: Directory containing TSV files to concatenate
        output_dir: Directory where full_data_unit.tsv should be saved
    
    Returns:
        Tuple (success, num_rows, num_files) or (False, 0, 0) on error
    """
    # Find all TSV files in the repertoire directory
    tsv_files = sorted(repertoire_dir.glob("*.tsv"))
    
    if not tsv_files:
        print(f"  Warning: No TSV files found in {repertoire_dir}")
        return False, 0, 0
    
    # List to store dataframes
    dfs = []
    
    # Read each file
    for tsv_file in tsv_files:
        try:
            df = pd.read_csv(tsv_file, sep='\t')
            dfs.append(df)
        except Exception as e:
            print(f"  Error reading {tsv_file}: {e}")
            continue
    
    if not dfs:
        print(f"  Warning: No data could be read from {repertoire_dir}")
        return False, 0, 0
    
    # Concatenate all dataframes
    full_data = pd.concat(dfs, ignore_index=True)
    
    # Save to output file
    output_file = output_dir / "full_data_unit.tsv"
    full_data.to_csv(output_file, sep='\t', index=False)
    
    # Remove the repertoire directory
    try:
        shutil.rmtree(repertoire_dir)
    except Exception as e:
        print(f"  Warning: Could not remove {repertoire_dir}: {e}")
    
    return True, len(full_data), len(tsv_files)


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    tcr_data_dir = script_dir / "tcr_dataset" / "tcr_data"
    
    # Check if input directory exists
    if not tcr_data_dir.exists():
        print(f"Error: Directory {tcr_data_dir} does not exist!")
        return
    
    print(f"Searching for repertoire directories in: {tcr_data_dir}\n")
    
    # Find all repertoire directories
    repertoire_dirs = find_repertoire_directories(tcr_data_dir)
    
    if not repertoire_dirs:
        print("No repertoire directories found!")
        return
    
    print(f"Found {len(repertoire_dirs)} repertoire directories to process\n")
    
    # Process each repertoire directory
    total_rows = 0
    total_files = 0
    successful = 0
    
    for repertoire_dir, output_dir in tqdm(repertoire_dirs, desc="Processing directories"):
        success, num_rows, num_files = concatenate_and_cleanup(repertoire_dir, output_dir)
        
        if success:
            successful += 1
            total_rows += num_rows
            total_files += num_files
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successfully processed: {successful}/{len(repertoire_dirs)} directories")
    print(f"Total TSV files concatenated: {total_files}")
    print(f"Total data rows: {total_rows}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

