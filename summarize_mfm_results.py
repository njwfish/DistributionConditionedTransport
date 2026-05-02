#!/usr/bin/env python3
"""
Read evaluation logs from the three MFM bash scripts and print a single
ready-to-paste LaTeX table with IID / OOD rows for each generator.

Log files expected at:
  logs/o_eval_mfm_fm_{0..14}
  logs/o_eval_mfm_energy_{0..14}
  logs/o_eval_mfm_swd_{0..14}

Usage:
    python summarize_mfm_results.py
    python summarize_mfm_results.py --logs_dir logs
"""

import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# ── Parameters must match the bash scripts ────────────────────────────────────
SPLIT_NAMES = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
METRICS     = ["mmd_energy", "mmd_rbf", "swd"]   # bash array order

IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

METRIC_DISPLAY_ORDER = ["mmd_energy", "swd", "mmd_rbf"]
METRIC_LATEX_LABELS  = {"mmd_energy": "Energy", "swd": "SWD", "mmd_rbf": "MMD-RBF"}

GENERATORS = [
    ("swd",    "SWD"),
    ("energy", "Energy"),
    ("fm",     "FM"),
]

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


def load_generator_data(logs_dir: Path, gen_key: str):
    """Returns (data, missing) where data[split][metric] = float or None."""
    data = {split: {metric: None for metric in METRICS} for split in SPLIT_NAMES}
    missing = []
    for task_id in range(15):
        split, metric = task_id_to_params(task_id)
        log_path = logs_dir / f"o_eval_mfm_{gen_key}_{task_id}"
        val = parse_model_mean(log_path, metric)
        if val is None:
            missing.append((task_id, log_path.name, split, metric))
        data[split][metric] = val
    return data, missing


def fmt_cell(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def get_cell(data: dict, split_group: list, metric: str) -> str:
    vals = [data[s][metric] for s in split_group if data[s].get(metric) is not None]
    if not vals:
        return "N/A"
    return fmt_cell(float(np.mean(vals)), float(np.std(vals, ddof=0)))


def main():
    parser = argparse.ArgumentParser(
        description="Summarise MFM evaluation logs as a LaTeX table."
    )
    parser.add_argument("--logs_dir", default="logs", help="Directory containing log files")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)

    all_data    = {}
    all_missing = {}
    for gen_key, _ in GENERATORS:
        data, missing = load_generator_data(logs_dir, gen_key)
        all_data[gen_key]    = data
        all_missing[gen_key] = missing

    total_missing = sum(len(v) for v in all_missing.values())
    if total_missing:
        print(f"% WARNING: {total_missing} log file(s) missing or unparseable:")
        for gen_key, missing in all_missing.items():
            for task_id, fname, split, metric in missing:
                print(f"%   [{gen_key}] job {task_id:>3d}  {fname:<30s}  split={split}, metric={metric}")
        print()

    metric_headers = " & ".join(
        f"\\textbf{{{METRIC_LATEX_LABELS[m]}}}" for m in METRIC_DISPLAY_ORDER
    )

    L = []
    L.append(r"\begin{table}[t]")
    L.append(r"    \centering")
    L.append(r"    \caption{Distributional metrics ($\downarrow$). "
             r"Mean $\pm$ standard deviation reported for IID (replicate holdout) "
             r"and OOD (patient holdout) settings.}")
    L.append(r"    \label{tab:trellis_mfm}")
    L.append(r"    \begin{tabular}{llccc}")
    L.append(r"    \toprule")
    L.append(f"    \\textbf{{Generator}} & & {metric_headers} \\\\")

    sections = [
        ("IID (Replicate Holdout)", IID_SPLITS),
        ("OOD (Patient Holdout)",   OOD_SPLITS),
    ]

    for section_label, split_group in sections:
        L.append(r"    \midrule")
        L.append(f"    \\multicolumn{{5}}{{c}}{{\\textit{{{section_label}}}}} \\\\")
        L.append(r"    \midrule")

        for gen_key, gen_display in GENERATORS:
            data = all_data[gen_key]
            cells = " & ".join(get_cell(data, split_group, m) for m in METRIC_DISPLAY_ORDER)
            L.append(f"    {gen_display} & & {cells} \\\\")

    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")

    print("\n".join(L))


if __name__ == "__main__":
    main()
