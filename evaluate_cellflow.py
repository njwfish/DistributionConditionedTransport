"""
Evaluate a trained CellFlow model on Trellis test samples.

For each Trellis test sample, this script:
1. Loads the trained CellFlow model.
2. Predicts treated cells from the sample's control cells using the sample's treatment.
3. Computes a distributional metric against the true treated cells.
4. Optionally computes the x0 -> x1 baseline.

The script assumes `cellflow` is already installed in the environment.

Usage:
    python evaluate_cellflow.py --split_name pdo21 --metric mmd_energy
    python evaluate_cellflow.py --split_name replicas-1 --compute_baseline
"""

import argparse
import gc
from pathlib import Path
from typing import Literal

import anndata
import numpy as np
import pandas as pd
import torch

from cellflow.model import CellFlow

from datasets.trellis import trellis_dataset
from generator.losses import mmd, sliced_wasserstein_distance, wasserstein

TREATMENTS = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]


def decode_treatment(cond_treat: np.ndarray) -> str:
    treat_idx = int(np.argmax(cond_treat[0]))
    return TREATMENTS[treat_idx]


def build_control_adata(x0: np.ndarray) -> anndata.AnnData:
    n = x0.shape[0]
    return anndata.AnnData(
        X=x0.astype(np.float32),
        obs={
            "control": [True] * n,
            "treatment": ["control"] * n,
        },
    )


def build_covariate_data(treatment_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "treatment": [treatment_name],
            "condition_id": [treatment_name],
        }
    )


def compute_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    metric: Literal["w1", "mmd_energy", "mmd_rbf", "swd"] = "w1",
    swd_subsample_rounds: int = 100,
) -> float:
    if metric == "w1":
        return wasserstein(pred, target, p=1)
    if metric == "mmd_energy":
        return mmd(pred, target, kernel="energy").item()
    if metric == "mmd_rbf":
        return mmd(pred, target, kernel="rbf").item()
    if metric == "swd":
        n_pred, n_target = pred.shape[0], target.shape[0]
        if n_pred == n_target:
            return sliced_wasserstein_distance(pred, target).item()

        min_size = min(n_pred, n_target)
        vals = []
        for _ in range(swd_subsample_rounds):
            if n_pred > n_target:
                idx = torch.randperm(n_pred, device=pred.device)[:min_size]
                pred_sub, target_sub = pred[idx], target
            else:
                idx = torch.randperm(n_target, device=target.device)[:min_size]
                pred_sub, target_sub = pred, target[idx]
            vals.append(sliced_wasserstein_distance(pred_sub, target_sub).item())
        return float(np.mean(vals))
    raise ValueError(f"Unknown metric: {metric}")


def predict_sample(model: CellFlow, x0: np.ndarray, treatment_name: str) -> np.ndarray:
    x0_adata = build_control_adata(x0)
    covariate_data = build_covariate_data(treatment_name)
    pred_dict = model.predict(
        adata=x0_adata,
        covariate_data=covariate_data,
        sample_rep="X",
        condition_id_key="condition_id",
    )
    if pred_dict is None:
        raise RuntimeError("CellFlow.predict returned None unexpectedly.")
    return np.asarray(pred_dict[treatment_name])


def main():
    parser = argparse.ArgumentParser(description="Evaluate CellFlow on Trellis test data")
    parser.add_argument(
        "--split_name",
        type=str,
        required=True,
        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Directory with the trained CellFlow model. Defaults to cellflow_model_{split_name}",
    )
    parser.add_argument("--set_size", type=int, default=32)
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mmd_energy", "mmd_rbf", "swd"],
        default="mmd_energy",
    )
    parser.add_argument("--compute_baseline", action="store_true")
    args = parser.parse_args()

    model_dir = Path(args.model_dir or f"cellflow_model_{args.split_name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Trellis dataset split: {args.split_name}")
    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)
    samples_test = dataset.samples_test
    print(f"Number of test samples: {len(samples_test)}")

    print(f"Loading CellFlow model from: {model_dir}")
    model = CellFlow.load(str(model_dir))

    metric_name = args.metric.upper()
    all_model_metrics = []
    all_baseline_metrics = [] if args.compute_baseline else None

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("Mode: CellFlow joint treatment-conditioned transport")
    print(f"Metric: {metric_name}")
    print("=" * 80)

    for i, sample in enumerate(samples_test):
        culture, x0, x1, _c0, _c1, cond_treat, patient = sample
        treatment_name = decode_treatment(cond_treat)

        x1_pred = predict_sample(model, x0, treatment_name)
        x1_pred_tensor = torch.tensor(x1_pred, dtype=torch.float32)
        x1_tensor = torch.tensor(x1, dtype=torch.float32)

        model_metric = compute_metric(x1_pred_tensor, x1_tensor, metric=args.metric)
        all_model_metrics.append(model_metric)

        baseline_metric = None
        if args.compute_baseline:
            x0_tensor = torch.tensor(x0, dtype=torch.float32)
            baseline_metric = compute_metric(x0_tensor, x1_tensor, metric=args.metric)
            all_baseline_metrics.append(baseline_metric)

        del x1_pred_tensor, x1_tensor
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        running_model = float(np.mean(all_model_metrics))
        print(f"\nSample {i + 1}/{len(samples_test)}:")
        print(f"  Culture: {culture}, Patient: {patient}, Treatment: {treatment_name}")
        print(f"  x0: {x0.shape}, x1: {x1.shape}")
        if args.compute_baseline:
            running_baseline = float(np.mean(all_baseline_metrics))
            print(
                f"  {metric_name:<10} Model: {model_metric:>12.6f}"
                f"  Baseline: {baseline_metric:>12.6f}"
            )
            print(
                f"  {'Running':<10} Model: {running_model:>12.6f}"
                f"  Baseline: {running_baseline:>12.6f}"
            )
        else:
            print(f"  {metric_name:<10} Model: {model_metric:>12.6f}")
            print(f"  {'Running':<10} Model: {running_model:>12.6f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print(f"Split: {args.split_name}")
    print(f"Samples evaluated: {len(all_model_metrics)}/{len(samples_test)}")
    print("=" * 80)

    if len(all_model_metrics) == 0:
        print("No samples evaluated.")
        return

    model_mean = float(np.mean(all_model_metrics))
    model_std = float(np.std(all_model_metrics))
    if args.compute_baseline:
        baseline_mean = float(np.mean(all_baseline_metrics))
        baseline_std = float(np.std(all_baseline_metrics))
        print(
            f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}"
            f"  Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}"
        )
    else:
        print(f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}")


if __name__ == "__main__":
    main()
