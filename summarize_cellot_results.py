#!/usr/bin/env python3
"""
Read evaluation logs from run_cellot_evaluate.sh and print a ready-to-paste
LaTeX tabular showing IID / OOD distributional metrics for CellOT.

Log files expected at: logs/o_eval_cellot_{0..14}

Usage:
    python summarize_cellot_results.py
    python summarize_cellot_results.py --logs_dir logs
"""

import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# ── Parameters must match run_cellot_evaluate.sh ──────────────────────────────
SPLIT_NAMES = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
METRICS     = ["mmd_energy", "mmd_rbf", "swd"]   # bash array order

IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

METRIC_DISPLAY_ORDER = ["mmd_energy", "swd", "mmd_rbf"]
METRIC_LATEX_LABELS  = {"mmd_energy": "Energy", "swd": "SWD", "mmd_rbf": "RBF"}

N_METRICS = len(METRICS)


def task_id_to_params(task_id: int):
    split_idx  = task_id // N_METRICS
    metric_idx = task_id  % N_METRICS
    return SPLIT_NAMES[split_idx], METRICS[metric_idx]


_RESULT_RE = re.compile(
    r'(MMD_ENERGY|MMD_RBF|SWD)\s+Model:\s+([\d.]+)\s+\+/-\s+([\d.]+)',
    re.IGNORECASE,
)


def parse_model_mean(log_path: Path, metric: str) -> Optional[float]:
    try:
        text = log_path.read_text()
    except FileNotFoundError:
        return None

    marker = "FINAL RESULTS"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    label = metric.upper()
    for m in _RESULT_RE.finditer(text[idx:]):
        if m.group(1).upper() == label:
            return float(m.group(2))
    return None


def fmt_cell(mean: float, std: float) -> str:
    return f"{mean:.4f}{{\\scriptsize$\\pm${std:.4f}}}"


def main():
    parser = argparse.ArgumentParser(
        description="Summarise CellOT evaluation logs as a LaTeX tabular."
    )
    parser.add_argument("--logs_dir", default="logs", help="Directory containing log files")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    # data[split][metric] = float or None
    data = {split: {metric: None for metric in METRICS} for split in SPLIT_NAMES}

    missing = []
    for task_id in range(15):
        split, metric = task_id_to_params(task_id)
        log_path = logs_dir / f"o_eval_cellot_{task_id}"
        val = parse_model_mean(log_path, metric)
        if val is None:
            missing.append((task_id, log_path.name, split, metric))
        data[split][metric] = val

    if missing:
        print(f"% WARNING: {len(missing)} log file(s) missing or unparseable:")
        for task_id, fname, split, metric in missing:
            print(f"%   job {task_id:>3d}  {fname:<25s}  split={split}, metric={metric}")
        print()

    def get_cell(split_group, metric):
        vals = [data[s][metric] for s in split_group if data[s].get(metric) is not None]
        if not vals:
            return "N/A"
        return fmt_cell(float(np.mean(vals)), float(np.std(vals, ddof=0)))

    metric_headers = " & ".join(METRIC_LATEX_LABELS[m] for m in METRIC_DISPLAY_ORDER)

    L = []
    L.append(r"\begin{tabular}{lccc}")
    L.append(r"\toprule")
    L.append(f"& {metric_headers} \\\\")
    L.append(r"\midrule")

    for row_label, split_group in [("IID", IID_SPLITS), ("OOD", OOD_SPLITS)]:
        cells = " & ".join(get_cell(split_group, m) for m in METRIC_DISPLAY_ORDER)
        L.append(f"{row_label} & {cells} \\\\")

    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")

    print("\n".join(L))


if __name__ == "__main__":
    main()
