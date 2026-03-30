#!/usr/bin/env python3
"""
Aggregate oval_ours evaluation results by variant (e.g. fm, energy, swd).

For each variant, computes:
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
    """Load CSV data and return as list of dicts."""
    data = []

    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return data

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'variant': row['variant'],
                'split': row['split'],
                'metric': row['metric'],
                'model_mean': float(row['model_mean']),
                'model_std': float(row['model_std']),
            })

    return data


def compute_aggregates(data: list, iid_splits: list, ood_splits: list, metric_filter: str = None) -> dict:
    """
    Compute IID and OOD aggregates for each variant.

    Returns: results[variant] = {
        'iid_mean': ..., 'iid_std': ..., 'iid_count': ...,
        'ood_mean': ..., 'ood_std': ..., 'ood_count': ...
    }
    """
    grouped = defaultdict(list)

    for row in data:
        if metric_filter and row['metric'] != metric_filter:
            continue
        grouped[row['variant']].append(row)

    results = {}

    for variant, rows in grouped.items():
        iid_values = [r['model_mean'] for r in rows if r['split'] in iid_splits]
        ood_values = [r['model_mean'] for r in rows if r['split'] in ood_splits]

        result = {}

        if iid_values:
            result['iid_mean'] = np.mean(iid_values)
            result['iid_std'] = np.std(iid_values, ddof=0) if len(iid_values) > 1 else 0.0
            result['iid_count'] = len(iid_values)

        if ood_values:
            result['ood_mean'] = np.mean(ood_values)
            result['ood_std'] = np.std(ood_values, ddof=0) if len(ood_values) > 1 else 0.0
            result['ood_count'] = len(ood_values)

        results[variant] = result

    return results


def format_value(mean: float, std: float) -> str:
    """Format mean±std."""
    return f"{mean:.4f}±{std:.4f}"


def main():
    parser = argparse.ArgumentParser(description='Aggregate oval_ours results by variant.')
    parser.add_argument('--input', type=str, default='ablation_results.csv',
                        help='Input CSV file (default: ablation_results.csv)')
    parser.add_argument('--metric', type=str, default=None,
                        choices=['mmd_energy', 'mmd_rbf', 'swd'],
                        help='Filter by specific metric (default: aggregate all metrics)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    csv_path = base_dir / args.input

    iid_splits = ['replicas-1', 'replicas-2']
    ood_splits = ['pdo21', 'pdo27', 'pdo75']
    metrics = ['mmd_energy', 'mmd_rbf', 'swd']

    data = load_csv_data(csv_path)
    if not data:
        print("No data loaded. Exiting.")
        return

    print(f"Loaded {len(data)} rows from {csv_path}")
    print()

    variants = sorted(set(r['variant'] for r in data))
    print(f"Variants found: {variants}")
    print()

    metrics_to_show = [args.metric] if args.metric else metrics

    col_width = 20
    sep = "=" * (12 + col_width * 2 + 20)

    for metric in metrics_to_show:
        results = compute_aggregates(data, iid_splits, ood_splits, metric_filter=metric)

        print(sep)
        print(f"METRIC: {metric.upper()}")
        print(sep)
        print(f"{'Variant':<12} {'IID (mean±std)':<{col_width}} {'OOD (mean±std)':<{col_width}} {'n_IID':<8} {'n_OOD':<8}")
        print("-" * (12 + col_width * 2 + 20))

        for variant in variants:
            if variant in results:
                r = results[variant]
                iid_str = format_value(r['iid_mean'], r['iid_std']) if 'iid_mean' in r else 'N/A'
                ood_str = format_value(r['ood_mean'], r['ood_std']) if 'ood_mean' in r else 'N/A'
                iid_count = r.get('iid_count', 0)
                ood_count = r.get('ood_count', 0)
                print(f"{variant:<12} {iid_str:<{col_width}} {ood_str:<{col_width}} {iid_count:<8} {ood_count:<8}")
            else:
                print(f"{variant:<12} {'MISSING':<{col_width}} {'MISSING':<{col_width}} {0:<8} {0:<8}")

        print()


if __name__ == '__main__':
    main()
