"""
Evaluate trained CellOT models on trellis test samples.

For each test sample, loads the CellOT model trained for the corresponding
treatment, transports control cells (x0) to predicted treated cells (x1_pred),
and computes distributional metrics against the true treated cells (x1).

Usage:
    python evaluate_cellot.py --split_name pdo21 --metric mmd_energy
    python evaluate_cellot.py --split_name replicas-1 --metric swd --compute_baseline
"""

import argparse
import gc
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cellot"))
from cellot.models.cellot import load_cellot_model
from cellot.utils.helpers import load_config

from datasets.trellis import trellis_dataset
from generator.losses import wasserstein, mmd, sliced_wasserstein_distance

TREATMENTS = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]


def compute_metric(pred, target, metric="w1", swd_subsample_rounds=100):
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


def load_config_from_dir(model_dir):
    """Load the CellOT model config (same yaml used for training)."""
    return load_config(os.path.join("cellot", "configs", "models", "cellot.yaml"))


def load_cellot_for_treatment(config, model_dir, treatment_name, input_dim, device):
    """Load a trained CellOT model for a specific treatment."""
    treatment_dir = os.path.join(model_dir, treatment_name)
    model_path = os.path.join(treatment_dir, "cache", "model.pt")
    if not os.path.exists(model_path):
        last_path = os.path.join(treatment_dir, "cache", "last.pt")
        if os.path.exists(last_path):
            model_path = last_path
        else:
            return None
    (f, g), _opts = load_cellot_model(config, restore=model_path, input_dim=input_dim)
    f.to(device).eval()
    g.to(device).eval()
    return f, g


def transport_cellot(g, x0, device, batch_size=2048):
    """Transport control cells through the learned OT map g."""
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    transported_parts = []
    for start in range(0, len(x0_tensor), batch_size):
        batch = x0_tensor[start:start + batch_size].requires_grad_(True)
        out = g.transport(batch)
        transported_parts.append(out.detach())
    return torch.cat(transported_parts, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate CellOT on trellis test data")
    parser.add_argument(
        "--split_name", type=str, required=True,
        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
    )
    parser.add_argument(
        "--model_dir", type=str, default=None,
        help="Base directory with trained models. Default: cellot_models/{split_name}",
    )
    parser.add_argument(
        "--metric", type=str,
        choices=["w1", "mmd_energy", "mmd_rbf", "swd"],
        default="mmd_energy",
    )
    parser.add_argument("--compute_baseline", action="store_true")
    parser.add_argument("--set_size", type=int, default=32)
    parser.add_argument(
        "--transport_batch_size", type=int, default=2048,
        help="Batch size for transporting cells through the model",
    )
    args = parser.parse_args()

    model_dir = args.model_dir or f"cellot_models/{args.split_name}"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    print(f"Loading dataset split: {args.split_name}")
    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)
    samples_test = dataset.samples_test
    print(f"Number of test samples: {len(samples_test)}")

    config = load_config_from_dir(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    input_dim = samples_test[0][1].shape[1]

    model_cache = {}
    missing_treatments = set()
    metric_name = args.metric.upper()
    all_model_metrics = []
    all_baseline_metrics = [] if args.compute_baseline else None

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("Mode: CellOT per-treatment transport (separate model per treatment)")
    print(f"Metric: {metric_name}")
    print("=" * 80)

    for i, sample in enumerate(samples_test):
        culture, x0, x1, _c0, _c1, cond_treat, patient = sample
        t_idx = int(np.argmax(cond_treat[0]))
        t_name = TREATMENTS[t_idx]

        if t_name in missing_treatments:
            continue

        if t_name not in model_cache:
            result = load_cellot_for_treatment(
                config, model_dir, t_name, input_dim, device,
            )
            if result is None:
                print(f"\nWARNING: No model for treatment '{t_name}', skipping.")
                missing_treatments.add(t_name)
                continue
            model_cache[t_name] = result

        _f, g = model_cache[t_name]
        x1_pred = transport_cellot(
            g, x0, device, batch_size=args.transport_batch_size,
        ).cpu()
        x1_tensor = torch.tensor(x1, dtype=torch.float32)
        model_metric = compute_metric(x1_pred, x1_tensor, metric=args.metric)
        all_model_metrics.append(model_metric)

        baseline_metric = None
        if args.compute_baseline:
            x0_tensor = torch.tensor(x0, dtype=torch.float32)
            baseline_metric = compute_metric(x0_tensor, x1_tensor, metric=args.metric)
            all_baseline_metrics.append(baseline_metric)

        del x1_pred, x1_tensor
        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        running_model = float(np.mean(all_model_metrics))
        print(f"\nSample {i + 1}/{len(samples_test)}:")
        print(f"  Culture: {culture}, Patient: {patient}, Treatment: {t_name}")
        print(f"  x0: {x0.shape}, x1: {x1.shape}")
        if args.compute_baseline:
            running_baseline = float(np.mean(all_baseline_metrics))
            print(f"  {metric_name:<10} Model: {model_metric:>12.6f}"
                  f"  Baseline: {baseline_metric:>12.6f}")
            print(f"  {'Running':<10} Model: {running_model:>12.6f}"
                  f"  Baseline: {running_baseline:>12.6f}")
        else:
            print(f"  {metric_name:<10} Model: {model_metric:>12.6f}")
            print(f"  {'Running':<10} Model: {running_model:>12.6f}")

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print(f"Split: {args.split_name}")
    print(f"Samples evaluated: {len(all_model_metrics)}/{len(samples_test)}")
    if missing_treatments:
        print(f"Treatments without models: {sorted(missing_treatments)}")
    print("=" * 80)

    if len(all_model_metrics) == 0:
        print("No samples evaluated.")
        return

    model_mean = float(np.mean(all_model_metrics))
    model_std = float(np.std(all_model_metrics))
    if args.compute_baseline:
        bl_mean = float(np.mean(all_baseline_metrics))
        bl_std = float(np.std(all_baseline_metrics))
        print(f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}"
              f"  Baseline: {bl_mean:.4f} +/- {bl_std:.4f}")
    else:
        print(f"{metric_name:<10} Model: {model_mean:.4f} +/- {model_std:.4f}")


if __name__ == "__main__":
    main()
