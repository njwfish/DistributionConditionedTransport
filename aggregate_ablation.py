#!/usr/bin/env python3
"""
Aggregate ablation evaluation results by consecutive_ratio and predictor_loss_weight.

For each combination of (consecutive_ratio, predictor_loss_weight), computes:
- IID score: mean across replicas-1, replicas-2 splits
- OOD score: mean across pdo21, pdo27, pdo75 splits

Usage:
    python aggregate_ablation.py
    python aggregate_ablation.py --input ablation_results.csv
    python aggregate_ablation.py --metric mmd_energy  # Filter by specific metric
"""

import csv
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_csv_data(csv_path: Path) -> list:
    """
    Load CSV data and return as list of dicts.
    """
    data = []
    
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return data
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'split': row['split'],
                'metric': row['metric'],
                'consecutive_ratio': float(row['consecutive_ratio']),
                'predictor_loss_weight': float(row['predictor_loss_weight']),
                'model_mean': float(row['model_mean']),
                'model_std': float(row['model_std']),
                'baseline_mean': float(row['baseline_mean']) if row['baseline_mean'] and row['baseline_mean'] != '' else None,
                'baseline_std': float(row['baseline_std']) if row['baseline_std'] and row['baseline_std'] != '' else None,
            })
    
    return data


def compute_aggregates(data: list, iid_splits: list, ood_splits: list, metric_filter: str = None) -> dict:
    """
    Compute IID and OOD aggregates for each (consecutive_ratio, predictor_loss_weight) pair.
    
    Returns: results[(cr, plw)] = {
        'iid_mean': ..., 'iid_std': ..., 
        'ood_mean': ..., 'ood_std': ...,
        'iid_baseline_mean': ..., 'iid_baseline_std': ...,
        'ood_baseline_mean': ..., 'ood_baseline_std': ...,
        'iid_count': ..., 'ood_count': ...
    }
    """
    # Group data by (consecutive_ratio, predictor_loss_weight)
    grouped = defaultdict(list)
    
    for row in data:
        if metric_filter and row['metric'] != metric_filter:
            continue
        key = (row['consecutive_ratio'], row['predictor_loss_weight'])
        grouped[key].append(row)
    
    results = {}
    
    for key, rows in grouped.items():
        # Separate IID and OOD
        iid_model_values = [r['model_mean'] for r in rows if r['split'] in iid_splits]
        ood_model_values = [r['model_mean'] for r in rows if r['split'] in ood_splits]
        
        iid_baseline_values = [r['baseline_mean'] for r in rows if r['split'] in iid_splits and r['baseline_mean'] is not None]
        ood_baseline_values = [r['baseline_mean'] for r in rows if r['split'] in ood_splits and r['baseline_mean'] is not None]
        
        result = {}
        
        if iid_model_values:
            result['iid_mean'] = np.mean(iid_model_values)
            result['iid_std'] = np.std(iid_model_values, ddof=0) if len(iid_model_values) > 1 else 0.0
            result['iid_count'] = len(iid_model_values)
        
        if ood_model_values:
            result['ood_mean'] = np.mean(ood_model_values)
            result['ood_std'] = np.std(ood_model_values, ddof=0) if len(ood_model_values) > 1 else 0.0
            result['ood_count'] = len(ood_model_values)
        
        if iid_baseline_values:
            result['iid_baseline_mean'] = np.mean(iid_baseline_values)
            result['iid_baseline_std'] = np.std(iid_baseline_values, ddof=0) if len(iid_baseline_values) > 1 else 0.0
        
        if ood_baseline_values:
            result['ood_baseline_mean'] = np.mean(ood_baseline_values)
            result['ood_baseline_std'] = np.std(ood_baseline_values, ddof=0) if len(ood_baseline_values) > 1 else 0.0
        
        results[key] = result
    
    return results


def format_value(mean: float, std: float) -> str:
    """Format mean±std."""
    return f"{mean:.4f}±{std:.4f}"


def main():
    parser = argparse.ArgumentParser(description='Aggregate ablation results by parameter combinations.')
    parser.add_argument('--input', type=str, default='ablation_results.csv',
                        help='Input CSV file (default: ablation_results.csv)')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['mmd_energy', 'mmd_rbf', 'swd'],
                        help='Filter by specific metric (default: aggregate all metrics)')
    args = parser.parse_args()
    
    base_dir = Path(__file__).parent
    csv_path = base_dir / args.input
    
    # Split groups
    iid_splits = ['replicas-1', 'replicas-2']
    ood_splits = ['pdo21', 'pdo27', 'pdo75']
    
    # All metrics
    metrics = ['mmd_energy', 'mmd_rbf', 'swd']
    
    # Load data
    data = load_csv_data(csv_path)
    if not data:
        print("No data loaded. Exiting.")
        return
    
    print(f"Loaded {len(data)} rows from {csv_path}")
    print()
    
    # If metric filter specified, only show that metric
    if args.metric:
        metrics_to_show = [args.metric]
    else:
        metrics_to_show = metrics
    
    # Parameter values
    consecutive_ratios = sorted(set(r['consecutive_ratio'] for r in data))
    predictor_loss_weights = sorted(set(r['predictor_loss_weight'] for r in data), reverse=True)
    
    print(f"Consecutive ratios: {consecutive_ratios}")
    print(f"Predictor loss weights: {predictor_loss_weights}")
    print()
    
    # Print table for each metric (or all combined)
    for metric in metrics_to_show:
        print("=" * 100)
        print(f"METRIC: {metric.upper()}" if metric else "ALL METRICS COMBINED")
        print("=" * 100)
        
        results = compute_aggregates(data, iid_splits, ood_splits, metric_filter=metric)
        
        # Table header
        col_width = 20
        print()
        print(f"{'Cons.Ratio':<12} {'Pred.Weight':<14} {'IID (mean±std)':<{col_width}} {'OOD (mean±std)':<{col_width}} {'IID Baseline':<{col_width}} {'OOD Baseline':<{col_width}}")
        print("-" * (12 + 14 + col_width * 4 + 4))
        
        for cr in consecutive_ratios:
            for plw in predictor_loss_weights:
                key = (cr, plw)
                if key in results:
                    r = results[key]
                    
                    iid_str = format_value(r['iid_mean'], r['iid_std']) if 'iid_mean' in r else 'N/A'
                    ood_str = format_value(r['ood_mean'], r['ood_std']) if 'ood_mean' in r else 'N/A'
                    iid_base_str = format_value(r['iid_baseline_mean'], r['iid_baseline_std']) if 'iid_baseline_mean' in r else 'N/A'
                    ood_base_str = format_value(r['ood_baseline_mean'], r['ood_baseline_std']) if 'ood_baseline_mean' in r else 'N/A'
                    
                    print(f"{cr:<12} {plw:<14} {iid_str:<{col_width}} {ood_str:<{col_width}} {iid_base_str:<{col_width}} {ood_base_str:<{col_width}}")
                else:
                    print(f"{cr:<12} {plw:<14} {'MISSING':<{col_width}} {'MISSING':<{col_width}} {'MISSING':<{col_width}} {'MISSING':<{col_width}}")
        
        print()
    
    # Print combined summary across all metrics
    print("=" * 100)
    print("COMBINED SUMMARY (averaged across all metrics)")
    print("=" * 100)
    
    results_all = compute_aggregates(data, iid_splits, ood_splits, metric_filter=None)
    
    col_width = 20
    print()
    print(f"{'Cons.Ratio':<12} {'Pred.Weight':<14} {'IID (mean±std)':<{col_width}} {'OOD (mean±std)':<{col_width}} {'n_IID':<8} {'n_OOD':<8}")
    print("-" * (12 + 14 + col_width * 2 + 16 + 4))
    
    for cr in consecutive_ratios:
        for plw in predictor_loss_weights:
            key = (cr, plw)
            if key in results_all:
                r = results_all[key]
                
                iid_str = format_value(r['iid_mean'], r['iid_std']) if 'iid_mean' in r else 'N/A'
                ood_str = format_value(r['ood_mean'], r['ood_std']) if 'ood_mean' in r else 'N/A'
                iid_count = r.get('iid_count', 0)
                ood_count = r.get('ood_count', 0)
                
                print(f"{cr:<12} {plw:<14} {iid_str:<{col_width}} {ood_str:<{col_width}} {iid_count:<8} {ood_count:<8}")
            else:
                print(f"{cr:<12} {plw:<14} {'MISSING':<{col_width}} {'MISSING':<{col_width}} {0:<8} {0:<8}")
    
    print()
    
    # Find best configuration
    print("=" * 100)
    print("BEST CONFIGURATIONS")
    print("=" * 100)
    print()
    
    # Best for IID
    best_iid_key = min(
        (k for k in results_all if 'iid_mean' in results_all[k]),
        key=lambda k: results_all[k]['iid_mean'],
        default=None
    )
    if best_iid_key:
        r = results_all[best_iid_key]
        print(f"Best IID:  consecutive_ratio={best_iid_key[0]}, predictor_loss_weight={best_iid_key[1]}")
        print(f"           IID = {format_value(r['iid_mean'], r['iid_std'])}")
    
    # Best for OOD
    best_ood_key = min(
        (k for k in results_all if 'ood_mean' in results_all[k]),
        key=lambda k: results_all[k]['ood_mean'],
        default=None
    )
    if best_ood_key:
        r = results_all[best_ood_key]
        print(f"Best OOD:  consecutive_ratio={best_ood_key[0]}, predictor_loss_weight={best_ood_key[1]}")
        print(f"           OOD = {format_value(r['ood_mean'], r['ood_std'])}")
    
    # Best combined (IID + OOD)
    best_combined_key = min(
        (k for k in results_all if 'iid_mean' in results_all[k] and 'ood_mean' in results_all[k]),
        key=lambda k: results_all[k]['iid_mean'] + results_all[k]['ood_mean'],
        default=None
    )
    if best_combined_key:
        r = results_all[best_combined_key]
        print(f"Best Combined (IID+OOD): consecutive_ratio={best_combined_key[0]}, predictor_loss_weight={best_combined_key[1]}")
        print(f"           IID = {format_value(r['iid_mean'], r['iid_std'])}, OOD = {format_value(r['ood_mean'], r['ood_std'])}")
    
    print()
    
    # LaTeX table
    print("=" * 100)
    print("LATEX TABLE")
    print("=" * 100)
    print()
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\begin{tabular}{cc|cc}")
    print("\\toprule")
    print("Cons. Ratio & Pred. Weight & IID & OOD \\\\")
    print("\\midrule")
    
    for cr in consecutive_ratios:
        for plw in predictor_loss_weights:
            key = (cr, plw)
            if key in results_all:
                r = results_all[key]
                iid_str = f"${r['iid_mean']:.4f} \\pm {r['iid_std']:.4f}$" if 'iid_mean' in r else 'N/A'
                ood_str = f"${r['ood_mean']:.4f} \\pm {r['ood_std']:.4f}$" if 'ood_mean' in r else 'N/A'
                print(f"{cr} & {plw} & {iid_str} & {ood_str} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Ablation study results for consecutive\\_ratio and predictor\\_loss\\_weight}")
    print("\\end{table}")


if __name__ == '__main__':
    main()
