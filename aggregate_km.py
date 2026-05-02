#!/usr/bin/env python3
"""
Aggregate KM / Trellis W1 evaluation results and print LaTeX table(s).

For each variant × metric, computes:
  - IID score: mean ± std across replicas-1, replicas-2 splits
  - OOD score: mean ± std across pdo21, pdo27, pdo75 splits

When the input CSV contains a non-empty ``treatment`` column (from
``parse_ablation_logs.py`` on per-class logs), prints one table per Trellis
treatment code by default. Use --single_table to pool (ignores treatment column).

Usage:
    python aggregate_km.py
    python aggregate_km.py --input km_ablation.csv
    python aggregate_km.py --input km_by_class.csv --label tab:km_cs
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from datasets.trellis_drug_classes import TRELLIS_TREATMENT_CODES


IID_SPLITS = ["replicas-1", "replicas-2"]
OOD_SPLITS = ["pdo21", "pdo27", "pdo75"]

METRICS = ["mmd_energy", "swd", "mmd_rbf", "w1"]
METRIC_LABELS = {
    "mmd_energy": "Energy",
    "swd": "SWD",
    "mmd_rbf": "MMD-RBF",
    "w1": "W1",
}

VARIANT_ORDER = ["swd", "energy", "fm"]
VARIANT_LABELS = {
    "swd": "SWD",
    "energy": "Energy",
    "fm": "FM",
}


def load_csv(csv_path: Path) -> list:
    data = []
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "variant": row["variant"],
                    "split": row["split"],
                    "metric": row["metric"],
                    "model_mean": float(row["model_mean"]),
                    "model_std": float(row["model_std"]),
                    "treatment": (row.get("treatment") or "").strip(),
                    "drug_class": (row.get("drug_class") or "").strip(),
                }
            )
    return data


def compute_aggregates(data: list) -> dict:
    """
    Returns aggregates[(variant, metric, setting)] = (mean, std)
    where setting is 'iid' or 'ood'.
    """
    grouped = defaultdict(list)
    for row in data:
        grouped[(row["variant"], row["metric"])].append(row)

    aggregates = {}
    for (variant, metric), rows in grouped.items():
        iid_vals = [r["model_mean"] for r in rows if r["split"] in IID_SPLITS]
        ood_vals = [r["model_mean"] for r in rows if r["split"] in OOD_SPLITS]

        if iid_vals:
            aggregates[(variant, metric, "iid")] = (
                np.mean(iid_vals),
                np.std(iid_vals, ddof=0),
            )
        if ood_vals:
            aggregates[(variant, metric, "ood")] = (
                np.mean(ood_vals),
                np.std(ood_vals, ddof=0),
            )

    return aggregates


def fmt(mean: float, std: float) -> str:
    return f"${mean:.4f} \\pm {std:.4f}$"


def _metric_sort_key(m: str) -> int:
    try:
        return METRICS.index(m)
    except ValueError:
        return len(METRICS)


def print_latex_table(
    aggregates: dict,
    variants: list,
    metrics: List[str],
    caption: str,
    label: str,
):
    metric_header = " & ".join(f"\\textbf{{{METRIC_LABELS.get(m, m)}}}" for m in metrics)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{l" + "c" * len(metrics) + "}",
        r"\toprule",
        f"\\textbf{{Generator}} & {metric_header} \\\\",
        r"\midrule",
    ]

    for setting, label_text in [("iid", "IID"), ("ood", "OOD")]:
        lines.append(
            f"\\multicolumn{{{1 + len(metrics)}}}{{c}}{{\\textit{{{label_text}}}}} \\\\"
        )
        lines.append(r"\midrule")

        for variant in variants:
            cells = []
            for metric in metrics:
                key = (variant, metric, setting)
                if key in aggregates:
                    mean, std = aggregates[key]
                    cells.append(fmt(mean, std))
                else:
                    cells.append("N/A")
            row = f"{VARIANT_LABELS.get(variant, variant):<8} & " + " & ".join(cells) + r" \\"
            lines.append(row)

        if setting == "iid":
            lines.append(r"\midrule")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n".join(lines))
    print()


def _treatment_sort_key(code: str) -> Tuple[int, str]:
    if code in TRELLIS_TREATMENT_CODES:
        return (TRELLIS_TREATMENT_CODES.index(code), code)
    return (len(TRELLIS_TREATMENT_CODES), code)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate KM results and print LaTeX table(s)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="km_ablation.csv",
        help="Input CSV file (default: km_ablation.csv)",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default=(
            "Distributional metrics ($\\downarrow$) — KM. "
            "Mean $\\pm$ standard deviation for IID and OOD settings."
        ),
        help="LaTeX table caption (used when not in per-class mode)",
    )
    parser.add_argument(
        "--label", type=str, default="tab:km_metrics", help="LaTeX table label"
    )
    parser.add_argument(
        "--single_table",
        action="store_true",
        help="Ignore treatment column and aggregate all rows together.",
    )
    args = parser.parse_args()

    csv_path = Path(__file__).parent / args.input
    data = load_csv(csv_path)
    print(f"Loaded {len(data)} rows from {csv_path}\n")

    variants_found = sorted(set(r["variant"] for r in data))
    variants = [v for v in VARIANT_ORDER if v in variants_found]
    variants += [v for v in variants_found if v not in VARIANT_ORDER]
    print(f"Variants: {variants}\n")

    has_class = any(r["treatment"] for r in data) and not args.single_table

    if has_class:
        treatments = sorted(
            {r["treatment"] for r in data if r["treatment"]},
            key=_treatment_sort_key,
        )
        class_desc: Dict[str, str] = {}
        for r in data:
            if r["treatment"] and r["treatment"] not in class_desc and r["drug_class"]:
                class_desc[r["treatment"]] = r["drug_class"]

        for tcode in treatments:
            sub = [r for r in data if r["treatment"] == tcode]
            metrics_use = sorted(
                {r["metric"] for r in sub},
                key=_metric_sort_key,
            )
            agg = compute_aggregates(sub)
            desc = class_desc.get(tcode, "")
            safe = (
                desc.replace("\\", "")
                .replace("&", "and")
                .replace("%", "")
                .replace("_", " ")
            )
            cap = (
                f"Distributional metrics ($\\downarrow$) — treatment \\texttt{{{tcode}}}"
                + (f". {safe}" if safe else "")
            )
            lbl = f"{args.label}_{tcode.lower()}"
            print(f"%% --- Treatment {tcode} ---")
            print_latex_table(agg, variants, metrics_use, caption=cap, label=lbl)
    else:
        metrics_use = sorted({r["metric"] for r in data}, key=_metric_sort_key)
        aggregates = compute_aggregates(data)
        print_latex_table(
            aggregates, variants, metrics_use, args.caption, args.label
        )


if __name__ == "__main__":
    main()
