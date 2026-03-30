#!/usr/bin/env python3
"""
Parse oval_ours evaluation log files and extract final metrics into a CSV file.

This script reads log files from the logs/ directory matching the pattern
oval_ours* and extracts the final model metrics for each combination of:
- split: replicas-1, replicas-2, pdo21, pdo27, pdo75
- metric: mmd_energy, mmd_rbf, swd
- variant: extracted from the filename (e.g. fm, energy, swd)

Usage:
    python parse_ablation_logs.py
    python parse_ablation_logs.py --logs_dir /path/to/logs
    python parse_ablation_logs.py --output results.csv
"""

import os
import re
import csv
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Optional


def parse_log_file(filepath: str) -> Optional[Dict]:
    """
    Parse a single oval_ours log file and extract the relevant information.

    Returns a dict with:
        - split: the data split name
        - metric: the metric name (lowercase)
        - variant: model variant extracted from the filename
        - model_mean: the mean model score
        - model_std: the std of model score

    Returns None if the file cannot be parsed or is incomplete.
    """
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    result = {}

    # Extract variant from filename: oval_ours_{variant}_{job_num}
    filename = os.path.basename(filepath)
    variant_match = re.match(r'oval_ours_([^_]+)', filename)
    result['variant'] = variant_match.group(1) if variant_match else 'unknown'

    # Extract split and metric from the first line
    # Format: "Job X: Evaluating split=..., metric=..."
    job_match = re.search(
        r'Job \d+: Evaluating split=([^,\n]+), metric=(\w+)',
        content
    )
    if not job_match:
        print(f"Could not find job info in {filepath}")
        return None

    result['split'] = job_match.group(1).strip()
    result['metric'] = job_match.group(2).lower()

    # Find the FINAL RESULTS section and extract the metric line
    # Look for lines like:
    # MMD_ENERGY Model:    0.0923 +/- 0.0941
    # MMD_RBF Model:    0.0178 +/- 0.0134
    # SWD    Model:    0.2379 +/- 0.0863
    final_section = re.search(r'FINAL RESULTS.*?={20,}(.*)', content, re.DOTALL)
    if not final_section:
        print(f"Could not find FINAL RESULTS section in {filepath}")
        return None

    final_content = final_section.group(1)

    metric_pattern = r'(MMD_ENERGY|MMD_RBF|SWD)\s+Model:\s+([\d.]+)\s*\+/-\s*([\d.]+)'
    metric_match = re.search(metric_pattern, final_content)
    if not metric_match:
        print(f"Could not find metric line in FINAL RESULTS of {filepath}")
        return None

    result['metric_name'] = metric_match.group(1).lower()
    result['model_mean'] = float(metric_match.group(2))
    result['model_std'] = float(metric_match.group(3))

    return result


def main():
    parser = argparse.ArgumentParser(description='Parse oval_ours evaluation log files and extract metrics to CSV.')
    parser.add_argument('--logs_dir', type=str, default=None,
                        help='Directory containing log files (default: ./logs)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file path (default: ./ablation_results.csv)')
    parser.add_argument('--pattern', type=str, default='oval_ours*',
                        help='Glob pattern for log files (default: oval_ours*)')
    args = parser.parse_args()

    # Set default paths
    script_dir = Path(__file__).parent
    logs_dir = Path(args.logs_dir) if args.logs_dir else script_dir / 'logs'
    output_csv = Path(args.output) if args.output else script_dir / 'ablation_results.csv'

    # Find all matching log files
    log_pattern = str(logs_dir / args.pattern)
    log_files = glob.glob(log_pattern)

    print(f"Found {len(log_files)} log files matching pattern: {log_pattern}")

    if len(log_files) == 0:
        print("No log files found. Exiting.")
        return

    # Store all parsed results
    results = []

    for filepath in sorted(log_files):
        parsed = parse_log_file(filepath)
        if parsed is None:
            print(f"Skipping {filepath}: could not parse")
            continue

        results.append({
            'variant': parsed['variant'],
            'split': parsed['split'],
            'metric': parsed['metric'],
            'model_mean': parsed['model_mean'],
            'model_std': parsed['model_std'],
            'log_file': os.path.basename(filepath),
        })

    print(f"\nSuccessfully parsed {len(results)} log files")

    # Sort results for consistent output
    results.sort(key=lambda x: (x['variant'], x['split'], x['metric']))

    # Write to CSV
    fieldnames = ['variant', 'split', 'metric', 'model_mean', 'model_std', 'log_file']

    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")

    # Print a summary table
    print("\n" + "="*90)
    print("RESULTS SUMMARY")
    print("="*90)

    splits = sorted(set(r['split'] for r in results))
    metrics = ['mmd_energy', 'mmd_rbf', 'swd']
    variants = sorted(set(r['variant'] for r in results))

    # Create lookup dict
    lookup = {}
    for r in results:
        key = (r['variant'], r['split'], r['metric'])
        lookup[key] = r

    print(f"\n{'Variant':<10} {'Split':<14} {'Metric':<12} {'Model Mean':<14} {'Model Std':<12}")
    print("-"*65)

    for variant in variants:
        for split in splits:
            has_any = any((variant, split, metric) in lookup for metric in metrics)
            if not has_any:
                print(f"{variant:<10} {split:<14} (no data found)")
                continue

            for metric in metrics:
                key = (variant, split, metric)
                if key in lookup:
                    r = lookup[key]
                    print(f"{variant:<10} {split:<14} {metric:<12} {r['model_mean']:<14.6f} {r['model_std']:<12.6f}")
                else:
                    print(f"{variant:<10} {split:<14} {metric:<12} {'MISSING':<14} {'MISSING':<12}")
        print()

    # Print summary statistics aggregated by variant
    print("\n" + "="*90)
    print("AGGREGATED BY VARIANT (averaged across splits and metrics)")
    print("="*90)
    for variant in variants:
        v_results = [r for r in results if r['variant'] == variant]
        if v_results:
            avg_model = sum(r['model_mean'] for r in v_results) / len(v_results)
            print(f"  variant={variant}: avg model_mean = {avg_model:.6f} (n={len(v_results)})")

    # Count missing combinations
    total_expected = len(variants) * len(splits) * len(metrics)
    missing = total_expected - len(results)
    if missing > 0:
        print(f"\nWarning: {missing} combinations are missing out of {total_expected} expected")


if __name__ == '__main__':
    main()
