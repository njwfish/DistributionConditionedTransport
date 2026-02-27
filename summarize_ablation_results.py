#!/usr/bin/env python3
"""
Read evaluation logs for the stratified ablation (60 jobs) and print 4 summary tables,
one per (consecutive_ratio, predictor_loss_weight) combination.

Each table has:
  - Columns: mmd_energy | swd | mmd_rbf  (fixed order)
  - Rows:    IID (mean±std over replicas-1, replicas-2)
             OOD (mean±std over pdo21, pdo27, pdo75)

Usage:
    python summarize_ablation_results.py
    python summarize_ablation_results.py --logs_dir logs --log_prefix o_ours_fm_
"""

import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# ── Parameters must match run_evaluate_trellis_all_metrics.sh ─────────────────
SPLIT_NAMES           = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
CONSECUTIVE_RATIOS    = [0.5, 0.2]
PREDICTOR_LOSS_WEIGHTS = [0.1, 0.001]
METRICS               = ["mmd_energy", "mmd_rbf", "swd"]   # internal order (same as bash array)

METRIC_DISPLAY_ORDER  = ["mmd_energy", "swd", "mmd_rbf"]   # column order in tables
METRIC_LABELS         = {"mmd_energy": "MMD_ENERGY", "swd": "SWD", "mmd_rbf": "MMD_RBF"}

IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

N_METRICS  = len(METRICS)
N_PLW      = len(PREDICTOR_LOSS_WEIGHTS)
N_CR       = len(CONSECUTIVE_RATIOS)


def task_id_to_params(task_id: int):
    """Return (split, consecutive_ratio, predictor_loss_weight, metric) for a given job index."""
    split_idx   = task_id // (N_CR * N_PLW * N_METRICS)
    remaining   = task_id  % (N_CR * N_PLW * N_METRICS)
    cr_idx      = remaining // (N_PLW * N_METRICS)
    remaining   = remaining  % (N_PLW * N_METRICS)
    plw_idx     = remaining // N_METRICS
    metric_idx  = remaining  % N_METRICS
    return (
        SPLIT_NAMES[split_idx],
        CONSECUTIVE_RATIOS[cr_idx],
        PREDICTOR_LOSS_WEIGHTS[plw_idx],
        METRICS[metric_idx],
    )


# e.g. "MMD_ENERGY Model:    0.1947 +/- 0.2263  Baseline: ..."
_RESULT_RE = re.compile(
    r'(MMD_ENERGY|MMD_RBF|SWD)\s+Model:\s+([\d.]+)\s+\+/-\s+([\d.]+)',
    re.IGNORECASE,
)


def parse_model_mean(log_path: Path, metric: str) -> Optional[float]:
    """Extract the model mean from the FINAL RESULTS block of a log file."""
    try:
        text = log_path.read_text()
    except FileNotFoundError:
        return None

    # Only look in the section after the last "FINAL RESULTS" header
    marker = "FINAL RESULTS"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    final_section = text[idx:]

    label = metric.upper()
    for m in _RESULT_RE.finditer(final_section):
        if m.group(1).upper() == label:
            return float(m.group(2))
    return None


def fmt_cell(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def print_table(cr: float, plw: float, data: dict):
    """data[split][metric] = float or None"""
    col_w = 20
    row_label_w = 5

    title = f"consecutive_ratio={cr}  |  predictor_loss_weight={plw}"
    total_w = row_label_w + 2 + (col_w + 3) * len(METRIC_DISPLAY_ORDER)

    print("=" * total_w)
    print(title)
    print("=" * total_w)

    # Header
    header = f"{'':>{row_label_w}}  "
    header += "  ".join(f"{METRIC_LABELS[m]:^{col_w}}" for m in METRIC_DISPLAY_ORDER)
    print(header)
    print("-" * total_w)

    for row_label, split_group in [("IID", IID_SPLITS), ("OOD", OOD_SPLITS)]:
        cells = []
        for m in METRIC_DISPLAY_ORDER:
            vals = [data[s][m] for s in split_group if data[s].get(m) is not None]
            if vals:
                cells.append(fmt_cell(np.mean(vals), np.std(vals, ddof=0)))
            else:
                cells.append("N/A")
        row = f"{row_label:>{row_label_w}}  " + "  ".join(f"{c:^{col_w}}" for c in cells)
        print(row)

    print()


def main():
    parser = argparse.ArgumentParser(description="Summarise stratified ablation evaluation logs.")
    parser.add_argument("--logs_dir",   default="logs",        help="Directory containing log files")
    parser.add_argument("--log_prefix", default="o_eval_strat_",  help="Log file name prefix (before the task ID)")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    # results[cr][plw][split][metric] = float or None
    results: dict = {
        cr: {
            plw: {split: {metric: None for metric in METRICS} for split in SPLIT_NAMES}
            for plw in PREDICTOR_LOSS_WEIGHTS
        }
        for cr in CONSECUTIVE_RATIOS
    }

    missing = []
    for task_id in range(60):
        split, cr, plw, metric = task_id_to_params(task_id)
        log_path = logs_dir / f"{args.log_prefix}{task_id}"
        val = parse_model_mean(log_path, metric)
        if val is None:
            missing.append((task_id, log_path.name, split, cr, plw, metric))
        results[cr][plw][split][metric] = val

    if missing:
        print(f"WARNING: {len(missing)} log file(s) missing or unparseable:")
        for task_id, fname, split, cr, plw, metric in missing:
            print(f"  job {task_id:>3d}  {fname:<20s}  split={split}, cr={cr}, plw={plw}, metric={metric}")
        print()

    for cr in CONSECUTIVE_RATIOS:
        for plw in PREDICTOR_LOSS_WEIGHTS:
            print_table(cr, plw, results[cr][plw])


if __name__ == "__main__":
    main()
