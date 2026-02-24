#!/usr/bin/env python3
"""
Aggregate SCGEN evaluation logs across split groups and print a LaTeX-like table.

Expected log files:
  logs/o_eval_scgen_{array_id}

Array mapping (from run_trellis_scgen_evaluate.sh):
  split_idx = array_id // 3, metric_idx = array_id % 3
  splits  = [replicas-1, replicas-2, pdo21, pdo27, pdo75]
  metrics = [mmd_energy, mmd_rbf, swd]

The script reads only the final "Model: mean +/- std" line per job and uses:
  - IID splits: replicas-1, replicas-2
  - OOD splits: pdo21, pdo27, pdo75

Aggregation uses population std (ddof=0), matching aggregate_by_metric.py.
"""

import argparse
import re
from pathlib import Path

import numpy as np


SPLITS = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
METRICS = ["mmd_energy", "mmd_rbf", "swd"]
METRIC_DISPLAY = {"mmd_energy": "Energy", "mmd_rbf": "RBF", "swd": "SWD"}
IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]


def parse_final_model_mean(path: Path):
    if not path.exists():
        return None
    text = path.read_text()
    matches = re.findall(r"Model:\s*([0-9]*\.?[0-9]+)\s*\+/-\s*([0-9]*\.?[0-9]+)", text)
    if not matches:
        return None
    mean_str, _std_str = matches[-1]
    return float(mean_str)


def collect_split_means(logs_dir: Path):
    # values[metric][split] = mean
    values = {m: {} for m in METRICS}
    missing = []

    n_jobs = len(SPLITS) * len(METRICS)
    for job_id in range(n_jobs):
        split = SPLITS[job_id // len(METRICS)]
        metric = METRICS[job_id % len(METRICS)]
        log_path = logs_dir / f"o_eval_scgen_{job_id}"
        mean = parse_final_model_mean(log_path)
        if mean is None:
            missing.append((job_id, split, metric, str(log_path)))
            continue
        values[metric][split] = mean
    return values, missing


def agg(values_by_split: dict, splits: list):
    vals = [values_by_split[s] for s in splits if s in values_by_split]
    if not vals:
        return None, None, 0
    arr = np.array(vals, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0)), len(vals)


def main():
    parser = argparse.ArgumentParser(description="Aggregate SCGEN logs into IID/OOD LaTeX-like table")
    parser.add_argument("--logs_dir", type=str, default="logs")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    values, missing = collect_split_means(logs_dir)

    print(r"\begin{tabular}{llc}")
    print(r"\toprule")
    print(r"& & scGen \\")
    print(r"\midrule")
    print(r"\multirow{3}{*}{IID}")
    for metric in ["mmd_rbf", "mmd_energy", "swd"]:
        mean, std, n = agg(values[metric], IID_SPLITS)
        label = METRIC_DISPLAY[metric]
        if n == 0:
            print(f"& {label} & N/A \\\\")
        else:
            print(f"& {label} & {mean:.4f}{{\\scriptsize$\\pm${std:.4f}}} \\\\")
    print(r"\midrule")
    print(r"\multirow{3}{*}{OOD}")
    for metric in ["mmd_rbf", "mmd_energy", "swd"]:
        mean, std, n = agg(values[metric], OOD_SPLITS)
        label = METRIC_DISPLAY[metric]
        if n == 0:
            print(f"& {label} & N/A \\\\")
        else:
            print(f"& {label} & {mean:.4f}{{\\scriptsize$\\pm${std:.4f}}} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    if missing:
        print("\nMissing/unfinished logs:")
        for job_id, split, metric, path in missing:
            print(f"- job {job_id:02d} ({split}, {metric}): {path}")


if __name__ == "__main__":
    main()
