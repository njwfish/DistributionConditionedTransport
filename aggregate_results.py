#!/usr/bin/env python3
"""
Aggregate evaluation results by computing mean and std across splits.

- IID: Average across replicas-1 and replicas-2
- OOD: Average across pdo21, pdo27, pdo75

Prints results for each metric with ours, mfm, and oracle side by side.

Usage:
    python aggregate_results.py --generator swd
    python aggregate_results.py --generator energy
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description='Aggregate evaluation results across splits.')
    parser.add_argument('--generator', type=str, required=True, choices=['swd', 'energy', 'fm'],
                        help='Generator model to aggregate results for (swd, energy, or fm)')
    args = parser.parse_args()
    
    generator = args.generator
    
    # Read the CSV file
    csv_path = Path(__file__).parent / f'evaluation_results_{generator}.csv'
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run parse_evaluation_logs.py --generator {generator} first.")
        return
    
    # Store data as: data[method][metric][split] = model_mean
    data = defaultdict(lambda: defaultdict(dict))
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            metric = row['metric']
            split = row['split']
            model_mean = float(row['model_mean'])
            data[method][metric][split] = model_mean
    
    # Define split groups
    iid_splits = ['replicas-1', 'replicas-2']
    ood_splits = ['pdo21', 'pdo27', 'pdo75']
    
    methods = ['ours', 'mfm', 'oracle']
    metrics = ['mmd_energy', 'mmd_rbf', 'swd']
    
    # Filter methods to only those with data
    available_methods = [m for m in methods if m in data and len(data[m]) > 0]
    if not available_methods:
        print("Error: No data found for any method.")
        return
    
    print(f"Found data for methods: {', '.join(available_methods)}")
    
    # Compute aggregated results
    # results[method][metric] = {'iid_mean': ..., 'iid_std': ..., 'ood_mean': ..., 'ood_std': ...}
    results = defaultdict(lambda: defaultdict(dict))
    
    for method in available_methods:
        for metric in metrics:
            # Get values for IID splits
            iid_values = [data[method][metric][s] for s in iid_splits if s in data[method][metric]]
            if iid_values:
                results[method][metric]['iid_mean'] = np.mean(iid_values)
                results[method][metric]['iid_std'] = np.std(iid_values, ddof=0)  # population std
            
            # Get values for OOD splits
            ood_values = [data[method][metric][s] for s in ood_splits if s in data[method][metric]]
            if ood_values:
                results[method][metric]['ood_mean'] = np.mean(ood_values)
                results[method][metric]['ood_std'] = np.std(ood_values, ddof=0)  # population std
    
    # Print results
    print()
    print("=" * 120)
    print(f"AGGREGATED EVALUATION RESULTS ({generator.upper()} generator)")
    print("=" * 120)
    print()
    print("IID = average across replicas-1, replicas-2")
    print("OOD = average across pdo21, pdo27, pdo75")
    print()
    
    # Determine column width based on number of methods
    col_width = 30
    
    for metric in metrics:
        print("-" * 120)
        print(f"METRIC: {metric.upper()}")
        print("-" * 120)
        print()
        
        # Header
        header = f"{'Split Type':<12} |"
        subheader = f"{'':<12} |"
        for method in available_methods:
            header += f" {method.upper():<{col_width}} |"
            subheader += f" {'Mean':<14} {'Std':<14} |"
        print(header)
        print(subheader)
        print("-" * (14 + (col_width + 3) * len(available_methods)))
        
        # IID results
        iid_line = f"{'IID':<12} |"
        for method in available_methods:
            if 'iid_mean' in results[method][metric]:
                iid_line += f" {results[method][metric]['iid_mean']:<14.6f} {results[method][metric]['iid_std']:<14.6f} |"
            else:
                iid_line += f" {'N/A':<14} {'N/A':<14} |"
        print(iid_line)
        
        # OOD results
        ood_line = f"{'OOD':<12} |"
        for method in available_methods:
            if 'ood_mean' in results[method][metric]:
                ood_line += f" {results[method][metric]['ood_mean']:<14.6f} {results[method][metric]['ood_std']:<14.6f} |"
            else:
                ood_line += f" {'N/A':<14} {'N/A':<14} |"
        print(ood_line)
        
        print()
    
    # Also print a compact summary table
    print("=" * 120)
    print("COMPACT SUMMARY (Mean ± Std)")
    print("=" * 120)
    print()
    
    # Dynamic header based on available methods
    header = f"{'Metric':<12} | {'Split':<6} |"
    for method in available_methods:
        header += f" {method.upper():<24} |"
    print(header)
    print("-" * (22 + 27 * len(available_methods)))
    
    for metric in metrics:
        for split_type in ['IID', 'OOD']:
            key_mean = f'{split_type.lower()}_mean'
            key_std = f'{split_type.lower()}_std'
            
            metric_label = metric.upper() if split_type == 'IID' else ''
            line = f"{metric_label:<12} | {split_type:<6} |"
            
            for method in available_methods:
                if key_mean in results[method][metric]:
                    mean_val = results[method][metric][key_mean]
                    std_val = results[method][metric][key_std]
                    val_str = f"{mean_val:.4f} ± {std_val:.4f}"
                else:
                    val_str = "N/A"
                line += f" {val_str:<24} |"
            
            print(line)
        print("-" * (22 + 27 * len(available_methods)))


if __name__ == '__main__':
    main()
