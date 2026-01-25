import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from typing import Optional
import ot as pot
from functools import partial

from utils.latents import normalize_latent, normalize_latent_np
from utils.predictor import LinearPredictor
from utils.experiment_utils import (
    load_config,
    load_experiment,
    is_source_only_model,
    find_experiment_dir,
)


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=1e7)
    if power == 2:
        ret = math.sqrt(ret)
    return ret

def compute_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    max_samples_w1: Optional[int] = None,
):
    """Compute W1 distance between two distributions."""
    if max_samples_w1 is not None:
        if pred.shape[0] > max_samples_w1:
            pred = pred[torch.randperm(pred.shape[0])[:max_samples_w1]]
        if target.shape[0] > max_samples_w1:
            target = target[torch.randperm(target.shape[0])[:max_samples_w1]]
    
    return {'W1': wasserstein(pred, target, power=1)}

# ============================================================================
# Latent Caching and Predictor Training
# ============================================================================

def get_latent_cache_path(experiment_dir: str, split: str) -> str:
    """Get the path for caching latents for a given split."""
    return os.path.join(experiment_dir, f"{split}_latents_cache_unified.pt")


def compute_and_cache_latents(
    encoder: torch.nn.Module,
    samples: list,
    device: torch.device,
    cache_path: str,
    split_name: str = "dataset",
) -> tuple:
    """Compute E(x0) and E(x1) for all samples. Returns (source_latents, target_latents, treat_conds)."""
    print(f"Computing {split_name} latents for {len(samples)} samples...")
    
    source_latents = []
    target_latents = []
    treat_conds = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
            
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)
            
            source_latents.append(encoder(x0_tensor).cpu().numpy())
            target_latents.append(encoder(x1_tensor).cpu().numpy())
            treat_conds.append(treat_cond[0:1])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")
    
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    treat_conds = np.vstack(treat_conds)
    
    print(f"Saving {split_name} latents to {cache_path}")
    torch.save({
        'source_latents': source_latents,
        'target_latents': target_latents,
        'treat_conds': treat_conds,
    }, cache_path)
    
    return source_latents, target_latents, treat_conds


def train_linear_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    device: torch.device,
    ridge_alpha: float = 1e-3,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    verbose: bool = True,
) -> LinearPredictor:
    """Train a linear predictor to map source latents to target latents."""
    print(f"Training linear predictor...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    print(f"  ridge_alpha={ridge_alpha}, lr={lr}, epochs={num_epochs}")
    
    input_dim = source_latents.shape[1]
    output_dim = target_latents.shape[1]
    
    predictor = LinearPredictor(input_dim, output_dim)
    predictor.fit(
        source_latents,
        target_latents,
        loss_type="mse",
        ridge_alpha=ridge_alpha,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        verbose=verbose,
    )
    
    return predictor


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor = None,
    compute_baseline: bool = False,
    max_samples_w1: Optional[int] = None,
    normalize_predicted_latent: bool = True,
    is_source_only: bool = False,
):
    """Evaluate the model on a single sample using precomputed latents."""
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    baseline_metrics = None
    if compute_baseline:
        baseline_metrics = compute_all_metrics(x0_tensor, x1_tensor, max_samples_w1=max_samples_w1)
    
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    if is_source_only:
        target_latent_tensor = None
    elif predictor is not None:
        # predictor is LinearPredictor with .predict() method
        predicted_target_latent = predictor.predict(source_latent)
        if normalize_predicted_latent:
            predicted_target_latent = normalize_latent_np(predicted_target_latent)
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        target_latent_tensor = torch.tensor(target_latent, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Generate x1_pred from x0 using source latent (and target latent if not source-only)
        x1_pred = generator.sample(
            x0_tensor,              # [N, dim] - source samples
            source_latent_tensor,   # [1, latent_dim]
            target_latent_tensor,   # [1, latent_dim] or None for source-only
        )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
    
    # Compute model metrics
    model_metrics = compute_all_metrics(x1_pred, x1_tensor, max_samples_w1=max_samples_w1)
    
    result = {
        'x1_pred': x1_pred.cpu().numpy(),
        'model': model_metrics,
    }
    
    if compute_baseline:
        result['baseline'] = baseline_metrics
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for all Trellis model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory. If not provided, searches based on --match criteria."
    )
    parser.add_argument(
        "--match",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter criterion in 'key=value' format. Keys use dot notation for nested access "
             "(e.g., 'experiment.split_name=replicas-1', 'seed=42'). "
             "Use 'null' or 'none' to match None values. Can be specified multiple times."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs"
    )
    parser.add_argument(
        "--compute_baseline",
        action="store_true",
        help="Compute baseline metrics (x0 vs x1)"
    )
    parser.add_argument(
        "--max_samples_w1",
        type=int,
        default=30000,
        help="Max samples for W1 computation. Set to 0 to disable subsampling."
    )
    parser.add_argument(
        "--use_predictor",
        action="store_true",
        help="Use a predictor P to predict target latent from source latent"
    )

    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        if not args.match:
            parser.error("Must provide either --experiment_dir or at least one --match criterion")
        
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            match_criteria=dict(item.split('=', 1) for item in args.match),
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experiment
    encoder, generator, dataset, cfg = load_experiment(args.experiment_dir, device)
    
    # Check model type
    is_source_only = is_source_only_model(cfg)
    if is_source_only:
        print("Detected SOURCE-ONLY model (no target latent used)")
    
    if is_source_only and args.use_predictor:
        raise ValueError("Source-only models do not use target latent, so predictors are not applicable.")
    
    test_samples = dataset.samples_test
    train_samples = dataset.samples_train
    
    # Compute and cache latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING LATENTS")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents, test_target_latents, test_treat_conds = compute_and_cache_latents(
        encoder, test_samples, device, test_cache_path, split_name="test"
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print("TRAINING LINEAR PREDICTOR")
        print("=" * 80)
        
        train_cache_path = get_latent_cache_path(args.experiment_dir, "train")
        train_source_latents, train_target_latents, _ = compute_and_cache_latents(
            encoder, train_samples, device, train_cache_path, split_name="train"
        )
        
        predictor = train_linear_predictor(train_source_latents, train_target_latents, device=device)
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if is_source_only:
        print("Mode: SOURCE-ONLY (no target latent used)")
    elif args.use_predictor:
        print("Mode: Using LINEAR predictor as target latent")
    else:
        print("Mode: Using oracle E(x1) as target latent")

    max_samples_w1 = args.max_samples_w1 if args.max_samples_w1 > 0 else None
    if max_samples_w1 is not None:
        print(f"W1 subsampling: max {max_samples_w1} samples from source/target")
    print("=" * 80)
    
    metric_names = ['W1']
    
    all_model_metrics = {name: [] for name in metric_names}
    all_baseline_metrics = {name: [] for name in metric_names} if args.compute_baseline else None
    
    for i, sample in enumerate(test_samples):
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
        
        source_latent = test_source_latents[i:i+1]
        target_latent = test_target_latents[i:i+1]
        sample_treat_cond = test_treat_conds[i:i+1]
        
        print(f"\nSample {i + 1}/{len(test_samples)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, device, 
            predictor=predictor,
            compute_baseline=args.compute_baseline,
            max_samples_w1=max_samples_w1,
            normalize_predicted_latent=True,
            is_source_only=is_source_only,
        )
        
        model = results['model']
        
        if args.compute_baseline:
            baseline = results['baseline']
            print(f"  {'Metric':<6} {'Model (pred vs true)':>20} {'Baseline (x0 vs true)':>22}")
            print(f"  {'-'*6} {'-'*20} {'-'*22}")
            for metric_name in metric_names:
                print(f"  {metric_name:<6} {model[metric_name]:>20.6f} {baseline[metric_name]:>22.6f}")
        else:
            print(f"  {'Metric':<6} {'Model (pred vs true)':>20}")
            print(f"  {'-'*6} {'-'*20}")
            for metric_name in metric_names:
                print(f"  {metric_name:<6} {model[metric_name]:>20.6f}")
        
        for key in metric_names:
            all_model_metrics[key].append(model[key])
            if args.compute_baseline:
                all_baseline_metrics[key].append(results['baseline'][key])
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print("=" * 80)
    
    if args.compute_baseline:
        print(f"{'Metric':<6} {'Model (pred vs true)':>25} {'Baseline (x0 vs true)':>25}")
        print(f"{'-'*6} {'-'*25} {'-'*25}")
        
        for metric_name in metric_names:
            model_mean = np.mean(all_model_metrics[metric_name])
            model_std = np.std(all_model_metrics[metric_name])
            baseline_mean = np.mean(all_baseline_metrics[metric_name])
            baseline_std = np.std(all_baseline_metrics[metric_name])
            
            model_str = f"{model_mean:.4f} +/- {model_std:.4f}"
            baseline_str = f"{baseline_mean:.4f} +/- {baseline_std:.4f}"
            print(f"{metric_name:<6} {model_str:>25} {baseline_str:>25}")
    else:
        print(f"{'Metric':<6} {'Model (pred vs true)':>25}")
        print(f"{'-'*6} {'-'*25}")
        
        for metric_name in metric_names:
            model_mean = np.mean(all_model_metrics[metric_name])
            model_std = np.std(all_model_metrics[metric_name])
            
            model_str = f"{model_mean:.4f} +/- {model_std:.4f}"
            print(f"{metric_name:<6} {model_str:>25}")


if __name__ == "__main__":
    main()
