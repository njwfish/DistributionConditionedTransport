#!/usr/bin/env python3
"""
Read evaluation logs from the predictor ablation runs and print a ready-to-paste
LaTeX table showing distributional metrics for each generator × predictor
combination, split into IID (replicate holdout) and OOD (patient holdout) sections.

Log files are expected at:
  logs/o_eval_predictor_ablation_{fm,energy,swd}_{0..44}

Usage:
    python summarize_predictor_ablation_results.py
    python summarize_predictor_ablation_results.py --logs_dir logs
"""

import re
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# ── Parameters must match the bash scripts ────────────────────────────────────
SPLIT_NAMES     = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
METRICS         = ["mmd_energy", "mmd_rbf", "swd"]   # internal order (same as bash array)
PREDICTOR_TYPES = ["ridge", "random_forest", "mlp"]

IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

METRIC_DISPLAY_ORDER = ["mmd_energy", "swd", "mmd_rbf"]
METRIC_LATEX_LABELS  = {"mmd_energy": "Energy", "swd": "SWD", "mmd_rbf": "MMD-RBF"}

GENERATORS = [
    ("swd",    "SWD"),
    ("energy", "Energy"),
    ("fm",     "FM"),
]

PREDICTOR_LATEX_LABELS = {
    "ridge":         "Ridge",
    "random_forest": "Random Forest",
    "mlp":           "MLP",
}

N_METRICS    = len(METRICS)
N_PREDICTORS = len(PREDICTOR_TYPES)


def task_id_to_params(task_id: int):
    """Return (split, metric, predictor_type) for a given job index."""
    split_idx     = task_id // (N_METRICS * N_PREDICTORS)
    remaining     = task_id  % (N_METRICS * N_PREDICTORS)
    metric_idx    = remaining // N_PREDICTORS
    predictor_idx = remaining  % N_PREDICTORS
    return (
        SPLIT_NAMES[split_idx],
        METRICS[metric_idx],
        PREDICTOR_TYPES[predictor_idx],
    )


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


def load_generator_data(logs_dir: Path, gen_key: str):
    """Load all results for one generator.

    Returns (data, missing) where
      data[split][metric][predictor] = float or None
    """
    data = {
        split: {metric: {pred: None for pred in PREDICTOR_TYPES} for metric in METRICS}
        for split in SPLIT_NAMES
    }
    missing = []
    for task_id in range(45):
        split, metric, predictor = task_id_to_params(task_id)
        log_path = logs_dir / f"o_eval_predictor_ablation_{gen_key}_{task_id}"
        val = parse_model_mean(log_path, metric)
        if val is None:
            missing.append((task_id, log_path.name, split, metric, predictor))
        data[split][metric][predictor] = val
    return data, missing


def fmt_cell(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def get_cell(data: dict, split_group: list, metric: str, predictor: str) -> str:
    vals = [
        data[s][metric][predictor]
        for s in split_group
        if data[s][metric].get(predictor) is not None
    ]
    if not vals:
        return "N/A"
    return fmt_cell(float(np.mean(vals)), float(np.std(vals, ddof=0)))


def print_latex_table(all_data: dict, all_missing: dict) -> None:
    total_missing = sum(len(v) for v in all_missing.values())
    if total_missing:
        print(f"% WARNING: {total_missing} log file(s) missing or unparseable:")
        for gen_key, missing in all_missing.items():
            for task_id, fname, split, metric, predictor in missing:
                print(f"%   [{gen_key}] job {task_id:>3d}  {fname:<40s}  "
                      f"split={split}, metric={metric}, predictor={predictor}")
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
    L.append(r"    \label{tab:trellis_predictor_ablation}")
    L.append(r"    \begin{tabular}{llccc}")
    L.append(r"    \toprule")
    L.append(f"    \\textbf{{Generator}} & \\textbf{{Predictor}} & {metric_headers} \\\\")

    sections = [
        ("IID (Replicate Holdout)", IID_SPLITS),
        ("OOD (Patient Holdout)",   OOD_SPLITS),
    ]

    for section_label, split_group in sections:
        L.append(r"    \midrule")
        L.append(f"    \\multicolumn{{5}}{{c}}{{\\textit{{{section_label}}}}} \\\\")

        for gen_key, gen_display in GENERATORS:
            L.append(r"    \midrule")
            data = all_data[gen_key]

            pred_rows = [
                (
                    PREDICTOR_LATEX_LABELS[pred],
                    " & ".join(
                        get_cell(data, split_group, m, pred) for m in METRIC_DISPLAY_ORDER
                    ),
                )
                for pred in PREDICTOR_TYPES
            ]

            n = len(pred_rows)
            first_label, first_cells = pred_rows[0]
            L.append(f"    \\multirow[t]{{{n}}}{{*}}{{{gen_display}}}")
            L.append(f"        & {first_label} & {first_cells} \\\\")
            for label, cells in pred_rows[1:]:
                L.append(f"        & {label} & {cells} \\\\")

    L.append(r"    \bottomrule")
    L.append(r"    \end{tabular}")
    L.append(r"\end{table}")

    print("\n".join(L))


def main():
    parser = argparse.ArgumentParser(
        description="Summarise predictor ablation evaluation logs as a LaTeX table."
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

    print_latex_table(all_data, all_missing)


if __name__ == "__main__":
    main()
