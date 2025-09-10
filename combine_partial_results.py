#!/usr/bin/env python3
"""
Script to combine partial aggregated data files from parallel jobs into final result.
This script merges all the partial_aggregated_data_job_*.pt files and performs the final aggregation.
"""

import torch
import gc
import argparse
import os
from pathlib import Path

def load_partial_files(total_jobs, data_dir="data/spikeprot0430"):
    """Load all partial aggregated data files from parallel jobs."""
    print(f"Loading partial results from {total_jobs} jobs...")
    
    merged_aggregated_data = {}
    loaded_jobs = []
    
    for job_id in range(total_jobs):
        with open(f"auxillary_log_job.log", "a") as f:
            f.write(f"Loading job {job_id}\n")
        partial_file = os.path.join(data_dir, f"partial_aggregated_data_job_{job_id}.pt")
        
        if not os.path.exists(partial_file):
            print(f"Warning: {partial_file} not found, skipping job {job_id}")
            continue
        
        try:
            print(f"Loading {partial_file}")
            partial_data = torch.load(partial_file)
            loaded_jobs.append(job_id)
            
            # Merge into main dictionary
            for time, elements in partial_data.items():
                if time not in merged_aggregated_data:
                    merged_aggregated_data[time] = []
                merged_aggregated_data[time].extend(elements)
            
            # Clean up
            del partial_data
            gc.collect()
            
        except Exception as e:
            print(f"Error loading {partial_file}: {e}")
            continue
    
    print(f"Successfully loaded {len(loaded_jobs)} out of {total_jobs} jobs")
    print(f"Loaded jobs: {loaded_jobs}")
    print(f"Found {len(merged_aggregated_data)} unique time entries across all jobs")
    
    return merged_aggregated_data

def perform_final_aggregation(aggregated_data, min_sequences=1):
    """Perform the final aggregation step: concatenate raw_texts and stack tensors for each time."""
    print("Performing final aggregation...")
    
    final_aggregated_data = {}
    
    for time, elements in aggregated_data.items():
        total_sequences = sum(len(el['raw_texts']) for el in elements)
        print(f"Time '{time}': {len(elements)} chunks, {total_sequences} sequences")
        
        if len(elements) == 0:
            continue
            
        # Concatenate all raw_texts lists
        all_raw_texts = []
        for el in elements:
            all_raw_texts.extend(el['raw_texts'])
        
        # Apply minimum sequences filter
        if len(all_raw_texts) < min_sequences:
            print(f"  -> Filtering out '{time}': only {len(all_raw_texts)} sequences (min required: {min_sequences})")
            continue
        
        # Stack all tensors for each key in samples
        stacked_samples = {}
        sample_keys = elements[0]['samples'].keys()  # Get keys from first element
        
        for key in sample_keys:
            # Collect all tensors for this key across all elements
            tensors_to_stack = []
            for el in elements:
                tensors_to_stack.append(el['samples'][key])
            
            # Stack them along dimension 0 (batch dimension)
            stacked_samples[key] = torch.cat(tensors_to_stack, dim=0)
        
        # Create the final aggregated entry for this time
        final_aggregated_data[time] = {
            'samples': stacked_samples,
            'time': time,
            'raw_texts': all_raw_texts
        }
        
        # Print detailed shape information for all tensor keys
        print(f"  -> Final aggregated: {len(all_raw_texts)} sequences")
        for key in sorted(sample_keys):
            print(f"     {key}: {stacked_samples[key].shape}")
    
    return final_aggregated_data


def print_summary_statistics(final_data_list):
    """Print summary statistics for the final aggregated data."""
    print("\n" + "="*60)
    print("FINAL SUMMARY STATISTICS")
    print("="*60)
    
    total_time = len(final_data_list)
    total_sequences = sum(len(data['raw_texts']) for data in final_data_list)
    
    print(f"Total time entries: {total_time}")
    print(f"Total sequences across all time: {total_sequences}")
    
    if final_data_list:
        # Get tensor shape info from first entry
        first_entry = final_data_list[0]
        sample_keys = list(first_entry['samples'].keys())
        print(f"Sample keys: {sample_keys}")
        
        for key in sample_keys:
            shape = first_entry['samples'][key].shape
            dtype = first_entry['samples'][key].dtype
            print(f"  {key}: shape template {shape}, dtype {dtype}")
        
        print(f"\nAll times in the dataset (sorted chronologically):")
        for data in final_data_list:
            time = data['time']
            num_sequences = len(data['raw_texts'])
            print(f"  {time}: {num_sequences} sequences")
            # Print tensor shapes for this time
            for key in sorted(sample_keys):
                print(f"    {key}: {data['samples'][key].shape}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Combine partial aggregated data files')
    parser.add_argument('--total_jobs', type=int, required=True,
                       help='Total number of jobs that were submitted')
    parser.add_argument('--data_dir', type=str, default="data/spikeprot0430",
                       help='Directory containing partial data files (default: data/spikeprot0430)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file path (default: data_dir/filtered_aggregated_data.pt)')
    parser.add_argument('--min_sequences', type=int, default=1,
                       help='Minimum number of raw_texts required to keep a time point (default: 1)')

    args = parser.parse_args()
    
    # Set default output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(args.data_dir, "filtered_aggregated_data.pt")
    
    # Ensure data directory exists
    os.makedirs(args.data_dir, exist_ok=True)
    
    print(f"Combining results from {args.total_jobs} jobs")
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output_file}")
    
    # Step 1: Load all partial files
    merged_aggregated_data = load_partial_files(args.total_jobs, args.data_dir)
    
    if not merged_aggregated_data:
        print("Error: No data loaded from partial files. Check if jobs completed successfully.")
        return 1
    
    # Step 2: Perform final aggregation with filtering
    final_aggregated_data = perform_final_aggregation(merged_aggregated_data, args.min_sequences)
    
    # Clean up intermediate aggregation data to free memory
    del merged_aggregated_data
    gc.collect()
    
    # Step 3: Convert dictionary to sorted list by time
    print(f"\nConverting to sorted list by time...")
    final_data_list = []
    
    # Sort by time (yyyy-mm format) - string sorting works for this format
    for time in sorted(final_aggregated_data.keys()):
        final_data_list.append(final_aggregated_data[time])
    
    # Clean up dictionary
    del final_aggregated_data
    gc.collect()
    
    # Optional: Clear GPU cache if using CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Step 4: Save the final aggregated data as a list
    print(f"\nSaving final aggregated data (as sorted list) to {args.output_file}")
    torch.save(final_data_list, args.output_file)
    print(f"Successfully saved final aggregated data as sorted list!")
    
    # Step 5: Print summary statistics
    print_summary_statistics(final_data_list)

    print(f"\n✅ Combination complete! Final file: {args.output_file}")
    
    return 0

if __name__ == "__main__":
    exit(main())
