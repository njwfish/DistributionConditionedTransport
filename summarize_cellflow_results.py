#!/usr/bin/env python3
"""
Summarize CellFlow evaluation logs into an IID/OOD LaTeX table.

Expected log format: output from run_evaluate_trellis_cellflow.sh, e.g.
    logs/oval03_cellflow_0 ... logs/oval03_cellflow_4

The aggregation matches aggregate_ablation.py / aggregate_by_metric.py:
- IID uses splits: replicas-1, replicas-2
- OOD uses splits: pdo21, pdo27, pdo75
- For each metric, aggregate over split-level model means using ddof=0 std.
"""

import argparse
import re
from pathlib import Path

import numpy as np

SPLIT_NAMES = ["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"]
IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

METRIC_DISPLAY_ORDER = ["mmd_energy", "swd", "mmd_rbf"]
METRIC_LABELS = {"mmd_energy": "Energy", "swd": "SWD", "mmd_rbf": "RBF"}

_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_SPLIT_RE = re.compile(r"Split:\s*([A-Za-z0-9\-]+)")
_RESULT_RE = re.compile(
    rf"(MMD_ENERGY|MMD_RBF|SWD)\s+Model:\s+({_NUM})\s+\+/-\s+({_NUM})",
    re.IGNORECASE,
)
_METRIC_FROM_LABEL = {
    "MMD_ENERGY": "mmd_energy",
    "MMD_RBF": "mmd_rbf",
    "SWD": "swd",
}


def parse_final_results(log_path: Path) -> tuple[str | None, dict[str, float]]:
    """Return (split_name, metric_means) parsed from FINAL RESULTS block."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None, {}

    idx = text.rfind("FINAL RESULTS")
    if idx == -1:
        return None, {}
    tail = text[idx:]

    split_match = _SPLIT_RE.search(tail)
    split_name = split_match.group(1) if split_match else None

    metric_means: dict[str, float] = {}
    for m in _RESULT_RE.finditer(tail):
        metric = _METRIC_FROM_LABEL[m.group(1).upper()]
        metric_means[metric] = float(m.group(2))
    return split_name, metric_means


def fmt_cell(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    return f"{mean:.4f}{{\\scriptsize$\\pm${std:.4f}}}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build IID/OOD LaTeX table from CellFlow evaluation logs."
    )
    parser.add_argument(
        "--logs_dir",
        default="logs",
        help="Directory containing CellFlow output logs (default: logs)",
    )
    parser.add_argument(
        "--log_glob",
        default="oval03_cellflow_*",
        help="Glob pattern for CellFlow output logs inside logs_dir.",
    )
    parser.add_argument(
        "--caption_model_name",
        default="CellFlow",
        help="Model name to include in table caption/label (default: CellFlow).",
    )
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    log_paths = sorted(logs_dir.glob(args.log_glob))
    if not log_paths:
        raise FileNotFoundError(
            f"No logs matched {args.log_glob!r} in {logs_dir}. "
            "Try passing --log_glob with your current run prefix."
        )

    # metrics_by_split[split][metric] = model_mean
    metrics_by_split: dict[str, dict[str, float]] = {s: {} for s in SPLIT_NAMES}
    parse_warnings: list[str] = []

    for log_path in log_paths:
        split_name, metric_means = parse_final_results(log_path)
        if split_name is None or not metric_means:
            parse_warnings.append(f"{log_path.name}: missing FINAL RESULTS block")
            continue
        if split_name not in metrics_by_split:
            parse_warnings.append(f"{log_path.name}: unknown split {split_name!r}, skipped")
            continue
        metrics_by_split[split_name] = metric_means

    if parse_warnings:
        for w in parse_warnings:
            print(f"% WARNING: {w}")
        print()

    missing_splits = [s for s in SPLIT_NAMES if not metrics_by_split[s]]
    if missing_splits:
        print(f"% WARNING: missing split results for: {', '.join(missing_splits)}")
        print()

    def aggregate(split_group: list[str], metric: str) -> str:
        vals = [
            metrics_by_split[s][metric]
            for s in split_group
            if metric in metrics_by_split[s]
        ]
        return fmt_cell(vals)

    header = " & ".join(METRIC_LABELS[m] for m in METRIC_DISPLAY_ORDER)
    iid_cells = " & ".join(aggregate(IID_SPLITS, m) for m in METRIC_DISPLAY_ORDER)
    ood_cells = " & ".join(aggregate(OOD_SPLITS, m) for m in METRIC_DISPLAY_ORDER)

    model_name = args.caption_model_name
    print(r"\begin{table}[]")
    print(r"    \centering")
    print(
        "    "
        + rf"\caption{{Distributional metrics for {model_name} ($\downarrow$). "
        + r"Mean $\pm$ standard deviation reported for IID (replicate holdout) "
        + r"and OOD (patient holdout) settings.}"
    )
    print("    " + rf"\label{{tab:{model_name}}}")
    print(r"    \begin{tabular}{lccc}")
    print(r"    \toprule")
    print(f"    & {header} \\\\")
    print(r"    \midrule")
    print(f"    IID & {iid_cells} \\\\")
    print(f"    OOD & {ood_cells} \\\\")
    print(r"    \bottomrule")
    print(r"    \end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
