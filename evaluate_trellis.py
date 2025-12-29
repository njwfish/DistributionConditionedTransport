"""
Evaluation script for the trained Trellis model.

This script:
1. Loads the trained model from the best checkpoint
2. Loads the config and instantiates the dataset with split_mode="test"
3. For each sample, encodes x0 and x1 from the same sample
4. Generates x1_pred using the generator with x0 as source and the latent of x1
5. Computes and prints W1, W2, MMD, and r2 metrics
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from typing import Optional
import ot as pot
from functools import partial
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

# ============================================================================
# Metric Functions (from the provided script)
# ============================================================================

def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
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


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))


def compute_pairwise_corrs(df):
    corr = df.corr().rename_axis(index="lhs", columns="rhs")
    return (
        corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
        .set_index(["lhs", "rhs"])
        .squeeze()
    )


def cellot_corr(pred, ground_truth):
    pwct = compute_pairwise_corrs(pd.DataFrame(pred))
    pwci = compute_pairwise_corrs(pd.DataFrame(ground_truth))
    r2_pairwise_feat_corrs = pd.Series(pwct).corr(pd.Series(pwci))
    return r2_pairwise_feat_corrs


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor):
    """
    Compute all metrics between two distributions.
    
    Args:
        pred: Predicted/source samples [N, dim] (torch tensor)
        target: Target samples [M, dim] (torch tensor)
        
    Returns:
        Dictionary with W1, W2, MMD, and r2
    """
    w1 = wasserstein(pred, target, power=1)
    w2 = wasserstein(pred, target, power=2)
    mmd = compute_scalar_mmd(target.cpu().numpy(), pred.cpu().numpy())
    r2 = cellot_corr(pred.cpu().numpy(), target.cpu().numpy())
    
    return {
        'W1': w1,
        'W2': w2,
        'MMD': mmd,
        'r2': r2,
    }


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def load_experiment(experiment_dir: str, device: torch.device):
    """
    Load the trained model, config, and instantiate the components.
    
    Args:
        experiment_dir: Path to the experiment directory containing best_model.pt and config.yaml
        device: Device to load the model onto
        
    Returns:
        encoder, generator, dataset, config
    """
    # Load config
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Resolve config references for instantiation
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    
    # Modify dataset config to use test split
    resolved_cfg.dataset.split_mode = "test"
    
    # Instantiate dataset
    dataset = hydra.utils.instantiate(resolved_cfg.dataset)
    print(f"Instantiated dataset with split_mode='test', {len(dataset.samples)} samples")
    
    # Instantiate encoder
    encoder = hydra.utils.instantiate(resolved_cfg.encoder)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    encoder.to(device)
    encoder.eval()
    print("Loaded encoder")
    
    # Instantiate generator
    generator = hydra.utils.instantiate(resolved_cfg.generator)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.to(device)
    generator.eval()
    print("Loaded generator")
    
    return encoder, generator, dataset, cfg


def evaluate_sample(
    encoder: torch.nn.Module,
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    device: torch.device,
):
    """
    Evaluate the model on a single sample.
    
    Args:
        encoder: Trained encoder
        generator: Trained generator
        x0: Source samples (numpy array of shape [N, dim])
        x1: Target samples (numpy array of shape [M, dim])
        device: Device to run on
        
    Returns:
        Dictionary with generated samples, model metrics, and baseline metrics
    """
    # Convert to tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    # Compute baseline metrics: x0 vs x1_true (before any transport)
    baseline_metrics = compute_all_metrics(x0_tensor, x1_tensor)
    
    with torch.no_grad():
        # Add batch dimension for encoding: [N, dim] -> [1, N, dim]
        x0_batched = x0_tensor.unsqueeze(0)
        x1_batched = x1_tensor.unsqueeze(0)
        
        # Encode x0 and x1 (full distributions)
        source_latent = encoder(x0_batched)  # [1, latent_dim]
        target_latent = encoder(x1_batched)  # [1, latent_dim]
        
        # Generate x1_pred from x0 using target latent
        # The generator expects source_samples of shape [batch_size, ...]
        # and latents of shape [num_sets, latent_dim]
        x1_pred = generator.sample(
            x0_tensor,           # [N, dim] - source samples
            source_latent,       # [1, latent_dim]
            target_latent,       # [1, latent_dim]
        )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
    
    # Compute model metrics: x1_pred vs x1_true
    model_metrics = compute_all_metrics(x1_pred, x1_tensor)
    
    return {
        'x1_pred': x1_pred.cpu().numpy(),
        'model': model_metrics,
        'baseline': baseline_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Trellis model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to the experiment directory (e.g., outputs/trellis_a2a_replicas-2_xxx)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load experiment
    encoder, generator, dataset, cfg = load_experiment(args.experiment_dir, device)
    
    # Evaluate each sample
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    
    # Track both model and baseline metrics
    all_model_metrics = {'W1': [], 'W2': [], 'MMD': [], 'r2': []}
    all_baseline_metrics = {'W1': [], 'W2': [], 'MMD': [], 'r2': []}
    
    for i, sample in enumerate(dataset.samples):
        culture, x0, x1, cell_cond, treat_cond, patient = sample
        
        print(f"\nSample {i + 1}/{len(dataset.samples)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(encoder, generator, x0, x1, device)
        
        model = results['model']
        baseline = results['baseline']
        
        print(f"  {'Metric':<6} {'Model (pred vs true)':>20} {'Baseline (x0 vs true)':>22}")
        print(f"  {'-'*6} {'-'*20} {'-'*22}")
        print(f"  {'W1':<6} {model['W1']:>20.6f} {baseline['W1']:>22.6f}")
        print(f"  {'W2':<6} {model['W2']:>20.6f} {baseline['W2']:>22.6f}")
        print(f"  {'MMD':<6} {model['MMD']:>20.6f} {baseline['MMD']:>22.6f}")
        print(f"  {'r2':<6} {model['r2']:>20.6f} {baseline['r2']:>22.6f}")
        
        # Collect metrics
        for key in all_model_metrics:
            all_model_metrics[key].append(model[key])
            all_baseline_metrics[key].append(baseline[key])
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for metric_name in ['W1', 'W2', 'MMD', 'r2']:
        model_values = np.array(all_model_metrics[metric_name])
        baseline_values = np.array(all_baseline_metrics[metric_name])
        
        print(f"\n{metric_name}:")
        print(f"  {'':15} {'Model':>15} {'Baseline':>15}")
        print(f"  {'Mean':15} {np.mean(model_values):>15.6f} {np.mean(baseline_values):>15.6f}")
        print(f"  {'Std':15} {np.std(model_values):>15.6f} {np.std(baseline_values):>15.6f}")
        print(f"  {'Median':15} {np.median(model_values):>15.6f} {np.median(baseline_values):>15.6f}")
        print(f"  {'Min':15} {np.min(model_values):>15.6f} {np.min(baseline_values):>15.6f}")
        print(f"  {'Max':15} {np.max(model_values):>15.6f} {np.max(baseline_values):>15.6f}")
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean ± std)")
    print("=" * 80)
    print(f"{'Metric':<6} {'Model (pred vs true)':>25} {'Baseline (x0 vs true)':>25}")
    print(f"{'-'*6} {'-'*25} {'-'*25}")
    
    for metric_name in ['W1', 'W2', 'MMD', 'r2']:
        model_mean = np.mean(all_model_metrics[metric_name])
        model_std = np.std(all_model_metrics[metric_name])
        baseline_mean = np.mean(all_baseline_metrics[metric_name])
        baseline_std = np.std(all_baseline_metrics[metric_name])
        
        model_str = f"{model_mean:.4f} ± {model_std:.4f}"
        baseline_str = f"{baseline_mean:.4f} ± {baseline_std:.4f}"
        print(f"{metric_name:<6} {model_str:>25} {baseline_str:>25}")


if __name__ == "__main__":
    main()

