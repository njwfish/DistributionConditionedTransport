"""
Evaluate Trellis test samples using SCGEN latent-space treatment shifts.

This script:
1. Loads the Trellis split and trained SCGEN model.
2. Computes one latent shift vector per treatment from train samples:
   delta_t = mean_latent(x1_t) - mean_latent(x0_t)
3. For each test sample, encodes all x0 cells, applies delta_t, decodes to x1_pred.
4. Computes per-sample metric (and optional baseline), then reports mean +/- std.

Unlike the Trellis encoder-generator evaluation, this ignores patient/culture/cell-type
for shift estimation and uses only treatment identity.
"""

import argparse
import gc
import os
import sys
from typing import Dict, Literal

import anndata
import numpy as np
import torch

from datasets.trellis import trellis_dataset
from generator.losses import wasserstein, mmd, sliced_wasserstein_distance

# Add local scgen package to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scgen"))
from scgen import SCGEN


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


def build_train_adata(samples_train, treatments):
    all_cells = []
    condition_labels = []
    treatment_labels = []
    culture_labels = []
    patient_labels = []

    for culture, x0, x1, _c0, _c1, cond_treat, patient in samples_train:
        treat_idx = int(np.argmax(cond_treat[0]))
        treat_name = treatments[treat_idx]

        all_cells.append(x0)
        condition_labels.extend(["control"] * x0.shape[0])
        treatment_labels.extend([treat_name] * x0.shape[0])
        culture_labels.extend([culture] * x0.shape[0])
        patient_labels.extend([patient] * x0.shape[0])

        all_cells.append(x1)
        condition_labels.extend(["treated"] * x1.shape[0])
        treatment_labels.extend([treat_name] * x1.shape[0])
        culture_labels.extend([culture] * x1.shape[0])
        patient_labels.extend([patient] * x1.shape[0])

    X = np.concatenate(all_cells, axis=0)
    return anndata.AnnData(
        X=X,
        obs={
            "condition": condition_labels,
            "treatment": treatment_labels,
            "culture": culture_labels,
            "patient": patient_labels,
        },
    )


def build_cells_adata(
    x: np.ndarray, treatment_name: str, condition: str, model: SCGEN
) -> anndata.AnnData:
    """Build AnnData and transfer scvi-tools registry from the trained model."""
    n = x.shape[0]
    adata = anndata.AnnData(
        X=x,
        obs={
            "condition": [condition] * n,
            "treatment": [treatment_name] * n,
            "culture": ["NA"] * n,
            "patient": ["NA"] * n,
        },
    )
    # Validate to transfer the model's registry (suppresses INFO messages)
    return model._validate_anndata(adata)


def load_scgen_model(model_dir: str, adata_for_registry: anndata.AnnData) -> SCGEN:
    # PyTorch 2.6 changed weights_only default to True, but scvi checkpoints contain
    # numpy objects. Allowlist the required numpy globals before loading.
    # Checkpoint is from our own training run, so weights_only=False is safe.
    _original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: _original_torch_load(
        *args, **{**kwargs, "weights_only": False}
    )
    try:
        try:
            return SCGEN.load(model_dir, adata=adata_for_registry)
        except TypeError:
            return SCGEN.load(model_dir, adata_for_registry)
    finally:
        torch.load = _original_torch_load


def compute_treatment_deltas(
    model: SCGEN,
    samples_train,
    treatments,
) -> Dict[int, np.ndarray]:
    x0_by_treat: Dict[int, list] = {}
    x1_by_treat: Dict[int, list] = {}

    for _culture, x0, x1, _c0, _c1, cond_treat, _patient in samples_train:
        t = int(np.argmax(cond_treat[0]))
        x0_by_treat.setdefault(t, []).append(x0)
        x1_by_treat.setdefault(t, []).append(x1)

    deltas: Dict[int, np.ndarray] = {}
    for t_idx, x0_chunks in x0_by_treat.items():
        x1_chunks = x1_by_treat[t_idx]
        x0_all = np.concatenate(x0_chunks, axis=0)
        x1_all = np.concatenate(x1_chunks, axis=0)

        t_name = treatments[t_idx]
        x0_adata = build_cells_adata(x0_all, treatment_name=t_name, condition="control", model=model)
        x1_adata = build_cells_adata(x1_all, treatment_name=t_name, condition="treated", model=model)

        z0 = model.get_latent_representation(x0_adata)
        z1 = model.get_latent_representation(x1_adata)
        deltas[t_idx] = np.mean(z1, axis=0) - np.mean(z0, axis=0)

    return deltas


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trellis with SCGEN treatment shifts")
    parser.add_argument(
        "--split_name",
        type=str,
        required=True,
        choices=["replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"],
    )
    parser.add_argument("--set_size", type=int, default=32)
    parser.add_argument(
        "--scgen_model_dir",
        type=str,
        default=None,
        help="Path to trained SCGEN model dir. Defaults to scgen_model_{split_name}",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=["w1", "mmd_energy", "mmd_rbf", "swd"],
        default="w1",
    )
    parser.add_argument("--compute_baseline", action="store_true")
    args = parser.parse_args()

    model_dir = args.scgen_model_dir or f"scgen_model_{args.split_name}"
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"SCGEN model dir not found: {model_dir}")

    print(f"Loading dataset split: {args.split_name}")
    dataset = trellis_dataset(split_name=args.split_name, set_size=args.set_size)
    samples_train = dataset.samples_train
    samples_test = dataset.samples_test
    treatments = dataset.treatment

    print(f"Building train AnnData for model registry transfer: {len(samples_train)} samples")
    adata_train = build_train_adata(samples_train, treatments)
    SCGEN.setup_anndata(adata_train, batch_key="condition", labels_key="treatment")

    print(f"Loading SCGEN model from: {model_dir}")
    model = load_scgen_model(model_dir, adata_train)
    model.module.eval()
    decode_device = next(model.module.parameters()).device

    print("Computing per-treatment latent shifts from train x0 -> x1")
    deltas = compute_treatment_deltas(model, samples_train, treatments)
    print(f"Computed shifts for {len(deltas)}/{len(treatments)} treatments")

    metric_name = args.metric.upper()
    all_model_metrics = []
    all_baseline_metrics = [] if args.compute_baseline else None

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("Mode: SCGEN per-treatment latent shift (train x0->x1), sample-level test evaluation")
    print(f"Metric: {metric_name}")
    print("=" * 80)

    for i, sample in enumerate(samples_test):
        culture, x0, x1, _c0, _c1, cond_treat, patient = sample
        t_idx = int(np.argmax(cond_treat[0]))
        if t_idx not in deltas:
            raise ValueError(
                f"No train-time shift available for treatment index {t_idx} ({treatments[t_idx]})."
            )

        t_name = treatments[t_idx]
        delta = deltas[t_idx]

        x0_adata = build_cells_adata(x0, treatment_name=t_name, condition="control", model=model)
        z0 = model.get_latent_representation(x0_adata)
        del x0_adata
        z1_pred = z0 + delta[None, :]
        del z0

        with torch.no_grad():
            z1_pred_tensor = torch.tensor(z1_pred, dtype=torch.float32, device=decode_device)
            x1_pred = model.module.generative(z1_pred_tensor)["px"].cpu()
            del z1_pred_tensor
        del z1_pred

        x1_tensor = torch.tensor(x1, dtype=torch.float32)
        model_metric = compute_metric(x1_pred, x1_tensor, metric=args.metric)
        del x1_pred
        all_model_metrics.append(model_metric)

        if args.compute_baseline:
            x0_tensor = torch.tensor(x0, dtype=torch.float32)
            baseline_metric = compute_metric(x0_tensor, x1_tensor, metric=args.metric)
            del x0_tensor
            all_baseline_metrics.append(baseline_metric)

        del x1_tensor

        if i % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        running_model = float(np.mean(all_model_metrics))
        print(f"\nSample {i + 1}/{len(samples_test)}:")
        print(f"  Culture: {culture}, Patient: {patient}, Treatment: {t_name}")
        if args.compute_baseline:
            running_baseline = float(np.mean(all_baseline_metrics))
            print(f"  {metric_name:<6} Model: {model_metric:>12.6f}  Baseline: {baseline_metric:>12.6f}")
            print(f"  {'Avg':<6} Model: {running_model:>12.6f}  Baseline: {running_baseline:>12.6f}")
        else:
            print(f"  {metric_name:<6} Model: {model_metric:>12.6f}")
            print(f"  {'Avg':<6} Model: {running_model:>12.6f}")

    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print("=" * 80)
    model_mean = float(np.mean(all_model_metrics))
    model_std = float(np.std(all_model_metrics))
    if args.compute_baseline:
        baseline_mean = float(np.mean(all_baseline_metrics))
        baseline_std = float(np.std(all_baseline_metrics))
        print(
            f"{metric_name:<6} Model: {model_mean:.4f} +/- {model_std:.4f}  "
            f"Baseline: {baseline_mean:.4f} +/- {baseline_std:.4f}"
        )
    else:
        print(f"{metric_name:<6} Model: {model_mean:.4f} +/- {model_std:.4f}")


if __name__ == "__main__":
    main()
