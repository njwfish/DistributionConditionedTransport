"""
Evaluate a trained CellFlow model on Trellis test samples.

For each Trellis test sample, this script:
1. Loads the trained CellFlow model.
2. Predicts treated cells from the sample's control cells using the sample's treatment.
3. Computes a distributional metric against the true treated cells.
4. Optionally computes the x0 -> x1 baseline.

The script assumes `cellflow` is already installed in the environment.

Usage:
    python evaluate_cellflow.py --split_name pdo21
    python evaluate_cellflow.py --split_name replicas-1 --compute_baseline
"""

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, Literal

import anndata
import numpy as np
import pandas as pd
import torch

# Reduce JAX GPU allocator pressure and fragmentation in long-running jobs.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax

from cellflow.model import CellFlow

from datasets.trellis import trellis_dataset
from generator.losses import mmd, sliced_wasserstein_distance, wasserstein

TREATMENTS = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]
EVAL_METRICS = ("mmd_energy", "mmd_rbf", "swd")
CACHE_SCHEMA_VERSION = 1


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
            # CellFlow expects all registered covariate columns from training, including control_key.
            "control": [True],
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
    pred = np.asarray(pred_dict[treatment_name], dtype=np.float32)
    del pred_dict, x0_adata, covariate_data
    return pred


def _normalize_metric_dict(metric_dict: dict[str, Any]) -> dict[str, float]:
    return {metric: float(metric_dict[metric]) for metric in EVAL_METRICS}


def load_cached_metrics(
    cache_path: Path,
    compute_baseline: bool,
) -> dict[int, dict[str, dict[str, float] | None]]:
    if not cache_path.exists():
        return {}

    cached: dict[int, dict[str, dict[str, float] | None]] = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: skipping malformed cache line {line_no} in {cache_path}")
                continue

            if row.get("schema_version") != CACHE_SCHEMA_VERSION:
                continue

            sample_idx = row.get("sample_idx")
            model_metrics_raw = row.get("model_metrics")
            if not isinstance(sample_idx, int) or not isinstance(model_metrics_raw, dict):
                continue
            if any(metric not in model_metrics_raw for metric in EVAL_METRICS):
                continue

            baseline_metrics: dict[str, float] | None = None
            baseline_metrics_raw = row.get("baseline_metrics")
            if compute_baseline:
                if not isinstance(baseline_metrics_raw, dict):
                    continue
                if any(metric not in baseline_metrics_raw for metric in EVAL_METRICS):
                    continue
                baseline_metrics = _normalize_metric_dict(baseline_metrics_raw)
            elif isinstance(baseline_metrics_raw, dict) and all(
                metric in baseline_metrics_raw for metric in EVAL_METRICS
            ):
                baseline_metrics = _normalize_metric_dict(baseline_metrics_raw)

            cached[sample_idx] = {
                "model_metrics": _normalize_metric_dict(model_metrics_raw),
                "baseline_metrics": baseline_metrics,
            }
    return cached


def append_cached_metrics(
    cache_path: Path,
    sample_idx: int,
    culture: Any,
    patient: Any,
    treatment_name: str,
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
) -> None:
    row = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "sample_idx": sample_idx,
        "culture": str(culture),
        "patient": str(patient),
        "treatment": treatment_name,
        "model_metrics": _normalize_metric_dict(model_metrics),
        "baseline_metrics": (
            _normalize_metric_dict(baseline_metrics) if baseline_metrics is not None else None
        ),
    }
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


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
    # Backward compatibility: older launch scripts may still pass --metric.
    # The argument is ignored because all metrics in EVAL_METRICS are always evaluated.
    parser.add_argument(
        "--metric",
        type=str,
        choices=["mmd_energy", "mmd_rbf", "swd"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--compute_baseline", action="store_true")
    parser.add_argument(
        "--cache_path",
        type=str,
        default=None,
        help="JSONL file used to cache per-sample metrics for resume.",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="If set, delete any existing per-sample cache before evaluation.",
    )
    parser.add_argument(
        "--clear_jax_cache_every",
        type=int,
        default=5,
        help="Clear JAX compilation caches every N samples to reduce memory growth. Set <= 0 to disable.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir or f"cellflow_model_{args.split_name}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading Trellis dataset split: {args.split_name}")
    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)
    samples_test = dataset.samples_test
    print(f"Number of test samples: {len(samples_test)}")

    cache_path = Path(args.cache_path or f"logs/cellflow_eval_{args.split_name}.jsonl")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_cache and cache_path.exists():
        print(f"Overwriting existing cache: {cache_path}")
        cache_path.unlink()
    cached_metrics_by_idx = load_cached_metrics(
        cache_path=cache_path,
        compute_baseline=args.compute_baseline,
    )
    if cached_metrics_by_idx:
        print(f"Loaded cached metrics for {len(cached_metrics_by_idx)} samples from {cache_path}")
    else:
        print(f"Caching per-sample metrics to {cache_path}")

    print(f"Loading CellFlow model from: {model_dir}")
    model = CellFlow.load(str(model_dir))

    if args.metric is not None:
        print(
            f"Ignoring --metric={args.metric}; evaluating only metric: "
            + ", ".join(EVAL_METRICS)
        )

    all_model_metrics = {metric: [] for metric in EVAL_METRICS}
    all_baseline_metrics = (
        {metric: [] for metric in EVAL_METRICS} if args.compute_baseline else None
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("Mode: CellFlow joint treatment-conditioned transport")
    print("Metrics: " + ", ".join(metric.upper() for metric in EVAL_METRICS))
    print("=" * 80)

    for i, sample in enumerate(samples_test):
        culture, x0, x1, _c0, _c1, cond_treat, patient = sample
        treatment_name = decode_treatment(cond_treat)

        baseline_metrics = None
        source = "computed"
        cached_metrics = cached_metrics_by_idx.get(i)
        if cached_metrics is not None:
            model_metrics = cached_metrics["model_metrics"]
            if model_metrics is None:
                raise RuntimeError(f"Corrupt cache for sample index {i}: missing model metrics.")
            for metric in EVAL_METRICS:
                all_model_metrics[metric].append(model_metrics[metric])

            if args.compute_baseline:
                baseline_metrics = cached_metrics["baseline_metrics"]
                if baseline_metrics is None:
                    raise RuntimeError(
                        f"Corrupt cache for sample index {i}: missing baseline metrics."
                    )
                for metric in EVAL_METRICS:
                    all_baseline_metrics[metric].append(baseline_metrics[metric])
            source = "cache"
        else:
            x1_pred = predict_sample(model, x0, treatment_name)
            x1_pred_tensor = torch.tensor(x1_pred, dtype=torch.float32)
            x1_tensor = torch.tensor(x1, dtype=torch.float32)

            model_metrics = {}
            for metric in EVAL_METRICS:
                metric_val = compute_metric(x1_pred_tensor, x1_tensor, metric=metric)
                model_metrics[metric] = metric_val
                all_model_metrics[metric].append(metric_val)

            if args.compute_baseline:
                x0_tensor = torch.tensor(x0, dtype=torch.float32)
                baseline_metrics = {}
                for metric in EVAL_METRICS:
                    metric_val = compute_metric(x0_tensor, x1_tensor, metric=metric)
                    baseline_metrics[metric] = metric_val
                    all_baseline_metrics[metric].append(metric_val)
                del x0_tensor

            append_cached_metrics(
                cache_path=cache_path,
                sample_idx=i,
                culture=culture,
                patient=patient,
                treatment_name=treatment_name,
                model_metrics=model_metrics,
                baseline_metrics=baseline_metrics,
            )

            del x1_pred, x1_pred_tensor, x1_tensor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if args.clear_jax_cache_every > 0 and (i + 1) % args.clear_jax_cache_every == 0:
            jax.clear_caches()

        print(f"\nSample {i + 1}/{len(samples_test)}:")
        print(f"  Culture: {culture}, Patient: {patient}, Treatment: {treatment_name}")
        print(f"  Source: {source}")
        print(f"  x0: {x0.shape}, x1: {x1.shape}")
        if args.compute_baseline and baseline_metrics is not None:
            for metric in EVAL_METRICS:
                running_model = float(np.mean(all_model_metrics[metric]))
                running_baseline = float(np.mean(all_baseline_metrics[metric]))
                metric_name = metric.upper()
                print(
                    f"  {metric_name:<10} Model: {model_metrics[metric]:>12.6f}"
                    f"  Baseline: {baseline_metrics[metric]:>12.6f}"
                )
                print(
                    f"  {'Running':<10} Model: {running_model:>12.6f}"
                    f"  Baseline: {running_baseline:>12.6f}"
                )
        else:
            for metric in EVAL_METRICS:
                running_model = float(np.mean(all_model_metrics[metric]))
                metric_name = metric.upper()
                print(f"  {metric_name:<10} Model: {model_metrics[metric]:>12.6f}")
                print(f"  {'Running':<10} Model: {running_model:>12.6f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print(f"Split: {args.split_name}")
    print(f"Samples evaluated: {len(next(iter(all_model_metrics.values())))}/{len(samples_test)}")
    print("=" * 80)

    if len(next(iter(all_model_metrics.values()))) == 0:
        print("No samples evaluated.")
        return

    for metric in EVAL_METRICS:
        metric_name = metric.upper()
        model_mean = float(np.mean(all_model_metrics[metric]))
        model_std = float(np.std(all_model_metrics[metric]))
        if args.compute_baseline:
            baseline_mean = float(np.mean(all_baseline_metrics[metric]))
            baseline_std = float(np.std(all_baseline_metrics[metric]))
            print(
                f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}"
                f"  Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}"
            )
        else:
            print(f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}")


if __name__ == "__main__":
    main()
