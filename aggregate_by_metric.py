#!/usr/bin/env python3
"""
Aggregate evaluation results by metric across all generators.

For each metric (mmd_energy, mmd_rbf, swd), creates a table where:
- Rows: generators (SWD, Energy, FM)
- Columns: methods (Supervised=mfm, Semi-supervised=ours, Oracle)
- Each cell shows IID and OOD results

Usage:
    python aggregate_by_metric.py
    python aggregate_by_metric.py --logs_dir logs_eval_swd_and_energy_complete_01_28_2025
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_csv_data(csv_path: Path) -> dict:
    """
    Load CSV data and return as nested dict.
    Returns: data[method][metric][split] = model_mean
    """
    data = defaultdict(lambda: defaultdict(dict))
    
    if not csv_path.exists():
        return data
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row['method']
            metric = row['metric']
            split = row['split']
            model_mean = float(row['model_mean'])
            data[method][metric][split] = model_mean
    
    return data


def compute_aggregates(data: dict, iid_splits: list, ood_splits: list) -> dict:
    """
    Compute IID and OOD aggregates for each method and metric.
    Returns: results[method][metric] = {'iid_mean': ..., 'iid_std': ..., 'ood_mean': ..., 'ood_std': ...}
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    for method in data:
        for metric in data[method]:
            # IID aggregation
            iid_values = [data[method][metric][s] for s in iid_splits if s in data[method][metric]]
            if iid_values:
                results[method][metric]['iid_mean'] = np.mean(iid_values)
                results[method][metric]['iid_std'] = np.std(iid_values, ddof=0)
            
            # OOD aggregation
            ood_values = [data[method][metric][s] for s in ood_splits if s in data[method][metric]]
            if ood_values:
                results[method][metric]['ood_mean'] = np.mean(ood_values)
                results[method][metric]['ood_std'] = np.std(ood_values, ddof=0)
    
    return results


def format_value(mean: float, std: float) -> str:
    """Format mean±std."""
    return f"{mean:.4f}±{std:.4f}"


def main():
    parser = argparse.ArgumentParser(description='Aggregate results by metric across all generators.')
    parser.add_argument('--logs_dir', type=str, default='logs',
                        help='Directory containing log files (default: logs)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    logs_dir = base_dir / args.logs_dir
    
    # Generators and their display names
    generators = ['swd', 'energy', 'fm']
    generator_display = {'swd': 'SWD', 'energy': 'Energy', 'fm': 'FM'}
    
    # Methods and their display names (matching the image)
    methods = ['mfm', 'ours', 'oracle']
    method_display = {'mfm': 'Supervised', 'ours': 'Semi-supervised', 'oracle': 'Oracle'}
    
    # Metrics
    metrics = ['mmd_energy', 'mmd_rbf', 'swd']
    metric_display = {'mmd_energy': 'MMD Energy', 'mmd_rbf': 'MMD RBF', 'swd': 'SWD'}
    
    # Split groups
    iid_splits = ['replicas-1', 'replicas-2']
    ood_splits = ['pdo21', 'pdo27', 'pdo75']
    
    # Load data for all generators
    # all_results[generator][method][metric] = {'iid_mean': ..., 'iid_std': ..., 'ood_mean': ..., 'ood_std': ...}
    all_results = {}
    
    for gen in generators:
        csv_path = base_dir / f'evaluation_results_{gen}.csv'
        if csv_path.exists():
            data = load_csv_data(csv_path)
            all_results[gen] = compute_aggregates(data, iid_splits, ood_splits)
            print(f"Loaded data for {gen} generator from {csv_path}")
        else:
            print(f"Warning: {csv_path} not found, skipping {gen} generator")
            all_results[gen] = {}
    
    print()
    
    # Print tables for each metric
    for metric in metrics:
        print("=" * 100)
        print(f"METRIC: {metric_display[metric]}")
        print("=" * 100)
        print()
        
        # Calculate column widths
        col_width = 24
        gen_col_width = 10
        
        # Header row
        header = f"{'Generator':<{gen_col_width}} |"
        for method in methods:
            header += f" {method_display[method]:^{col_width}} |"
        print(header)
        print("-" * (gen_col_width + 2 + (col_width + 3) * len(methods)))
        
        # Data rows (one per generator, with IID and OOD sub-rows)
        for gen in generators:
            if gen not in all_results or not all_results[gen]:
                print(f"{generator_display[gen]:<{gen_col_width}} | (no data)")
                continue
            
            # IID row
            iid_line = f"{generator_display[gen]:<{gen_col_width}} |"
            for method in methods:
                if method in all_results[gen] and metric in all_results[gen][method]:
                    res = all_results[gen][method][metric]
                    if 'iid_mean' in res:
                        val_str = format_value(res['iid_mean'], res['iid_std'])
                        iid_line += f" IID: {val_str:<{col_width-5}} |"
                    else:
                        iid_line += f" IID: {'N/A':<{col_width-5}} |"
                else:
                    iid_line += f" IID: {'N/A':<{col_width-5}} |"
            print(iid_line)
            
            # OOD row
            ood_line = f"{'':<{gen_col_width}} |"
            for method in methods:
                if method in all_results[gen] and metric in all_results[gen][method]:
                    res = all_results[gen][method][metric]
                    if 'ood_mean' in res:
                        val_str = format_value(res['ood_mean'], res['ood_std'])
                        ood_line += f" OOD: {val_str:<{col_width-5}} |"
                    else:
                        ood_line += f" OOD: {'N/A':<{col_width-5}} |"
                else:
                    ood_line += f" OOD: {'N/A':<{col_width-5}} |"
            print(ood_line)
            print("-" * (gen_col_width + 2 + (col_width + 3) * len(methods)))
        
        print()
    
    # Also print a compact LaTeX-style table for each metric
    print("=" * 100)
    print("LATEX-STYLE COMPACT TABLES")
    print("=" * 100)
    
    for metric in metrics:
        print()
        print(f"% {metric_display[metric]}")
        print(f"{'Generator':<10} & {'Split':<5} & {' & '.join([method_display[m] for m in methods])} \\\\")
        print("\\hline")
        
        for gen in generators:
            if gen not in all_results or not all_results[gen]:
                continue
            
            for split_type in ['IID', 'OOD']:
                key_mean = f'{split_type.lower()}_mean'
                key_std = f'{split_type.lower()}_std'
                
                gen_label = generator_display[gen] if split_type == 'IID' else ''
                values = []
                
                for method in methods:
                    if method in all_results[gen] and metric in all_results[gen][method]:
                        res = all_results[gen][method][metric]
                        if key_mean in res:
                            val_str = format_value(res[key_mean], res[key_std])
                            values.append(val_str)
                        else:
                            values.append('N/A')
                    else:
                        values.append('N/A')
                
                print(f"{gen_label:<10} & {split_type:<5} & {' & '.join(values)} \\\\")
        print("\\hline")


if __name__ == '__main__':
    main()
