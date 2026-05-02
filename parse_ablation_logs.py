#!/usr/bin/env python3
"""
Parse Trellis evaluation log files and extract final metrics into a CSV file.

Supports logs from run_evaluate_trellis_w1_metrics_*.sh (e.g. oval_ctt_fm_*,
oval_km_energy_*) and legacy oval_uni_* patterns.

When evaluation used --aggregate_by_drug_class, FINAL RESULTS contain one line per
Trellis treatment code (CLASS O MMD_ENERGY Model: ...). The CSV then has one row
per (variant, split, metric, treatment); use aggregate_km.py --per_class for
IID/OOD tables per condition.

Usage:
    python parse_ablation_logs.py
    python parse_ablation_logs.py --logs_dir /path/to/logs --pattern 'oval_ctt_*'
    python parse_ablation_logs.py --output km_ablation_by_class.csv
"""

import csv
import glob
import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from datasets.trellis_drug_classes import class_description


# Per-class lines from evaluate_trellis_experimental.py --aggregate_by_drug_class
_CLASS_ROW_RE = re.compile(
    r"CLASS\s+(\S+)\s+(MMD_ENERGY|MMD_RBF|SWD|W1)\s+Model:\s+([\d.]+)\s*\+/-\s*([\d.]+)",
    re.IGNORECASE,
)
# Pooled (legacy) single summary line
_POOLED_ROW_RE = re.compile(
    r"(MMD_ENERGY|MMD_RBF|SWD|W1)\s+Model:\s+([\d.]+)\s*\+/-\s*([\d.]+)",
    re.IGNORECASE,
)


def extract_variant_from_filename(filename: str) -> Optional[str]:
    """e.g. oval_ctt_fm_0 -> fm, oval_km_swd_14 -> swd, oval_uni_energy_1 -> energy."""
    base = os.path.basename(filename)
    m = re.search(r"oval_(?:ctt|km|uni)_(energy|swd|fm)_", base, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def parse_log_file(filepath: str) -> Optional[List[Dict]]:
    """
    Parse one log file. Returns a list of row dicts (one row per treatment if
    per-class logs; otherwise one pooled row). None if the file is unusable.
    """
    try:
        with open(filepath, "r") as f:
            content = f.read()
    except OSError as e:
        print(f"Error reading {filepath}: {e}")
        return None

    variant = extract_variant_from_filename(filepath)
    if not variant:
        print(f"Could not infer variant from filename {filepath}")
        return None

    job_match = re.search(
        r"Job \d+: Evaluating split=([^,\n]+), metric=(\w+)", content
    )
    if not job_match:
        print(f"Could not find job info in {filepath}")
        return None

    split = job_match.group(1).strip()
    metric_job = job_match.group(2).lower()

    final_section = re.search(r"FINAL RESULTS.*?={20,}(.*)", content, re.DOTALL)
    if not final_section:
        print(f"Could not find FINAL RESULTS section in {filepath}")
        return None

    final_content = final_section.group(1)
    base_name = os.path.basename(filepath)

    class_matches = list(_CLASS_ROW_RE.finditer(final_content))
    rows: List[Dict] = []

    if class_matches:
        for m in class_matches:
            code = m.group(1)
            metric_tag = m.group(2).lower()
            if metric_tag == "mmd_energy":
                metric_norm = "mmd_energy"
            elif metric_tag == "mmd_rbf":
                metric_norm = "mmd_rbf"
            elif metric_tag == "swd":
                metric_norm = "swd"
            elif metric_tag == "w1":
                metric_norm = "w1"
            else:
                metric_norm = metric_tag
            if metric_norm != metric_job:
                print(
                    f"Warning: job metric={metric_job} vs CLASS line metric={metric_norm} in {base_name}"
                )
            rows.append(
                {
                    "variant": variant,
                    "split": split,
                    "metric": metric_norm,
                    "treatment": code,
                    "drug_class": class_description(code),
                    "model_mean": float(m.group(3)),
                    "model_std": float(m.group(4)),
                    "log_file": base_name,
                }
            )
        return rows

    pooled = _POOLED_ROW_RE.search(final_content)
    if not pooled:
        print(f"Could not find metric line in FINAL RESULTS of {filepath}")
        return None

    metric_tag = pooled.group(1).lower()
    if metric_tag == "mmd_energy":
        metric_norm = "mmd_energy"
    elif metric_tag == "mmd_rbf":
        metric_norm = "mmd_rbf"
    elif metric_tag == "swd":
        metric_norm = "swd"
    elif metric_tag == "w1":
        metric_norm = "w1"
    else:
        metric_norm = metric_tag

    return [
        {
            "variant": variant,
            "split": split,
            "metric": metric_norm,
            "treatment": "",
            "drug_class": "",
            "model_mean": float(pooled.group(2)),
            "model_std": float(pooled.group(3)),
            "log_file": base_name,
        }
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Parse Trellis evaluation logs and extract metrics to CSV."
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=None,
        help="Directory containing log files (default: ./logs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: ./ablation_results.csv)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="oval_ctt_*",
        help="Glob pattern for log files (default: oval_ctt_*)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    logs_dir = Path(args.logs_dir) if args.logs_dir else script_dir / "logs"
    output_csv = Path(args.output) if args.output else script_dir / "ablation_results.csv"

    log_pattern = str(logs_dir / args.pattern)
    log_files = glob.glob(log_pattern)

    print(f"Found {len(log_files)} log files matching pattern: {log_pattern}")

    if len(log_files) == 0:
        print("No log files found. Exiting.")
        return

    results: List[Dict] = []
    for filepath in sorted(log_files):
        parsed_list = parse_log_file(filepath)
        if not parsed_list:
            print(f"Skipping {filepath}: could not parse")
            continue
        results.extend(parsed_list)

    print(f"\nSuccessfully parsed {len(results)} metric rows from {len(log_files)} log files")

    results.sort(
        key=lambda x: (
            x["variant"],
            x["split"],
            x["metric"],
            x["treatment"] or "\x00",
        )
    )

    fieldnames = [
        "variant",
        "split",
        "metric",
        "treatment",
        "drug_class",
        "model_mean",
        "model_std",
        "log_file",
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_csv}")

    print("\n" + "=" * 90)
    print("RESULTS SUMMARY (first 40 rows)")
    print("=" * 90)
    for r in results[:40]:
        t = r["treatment"] or "(pooled)"
        print(
            f"  {r['variant']:<6} {r['split']:<12} {r['metric']:<10} {t:<6} "
            f"mean={r['model_mean']:.6f} std={r['model_std']:.6f}"
        )
    if len(results) > 40:
        print(f"  ... ({len(results) - 40} more rows)")


if __name__ == "__main__":
    main()
