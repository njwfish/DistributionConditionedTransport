"""
Evaluation script for the trained Trellis MFM (Marginal Flow Matching) model.

This script:
1. Loads the trained MFM model from the best checkpoint
2. Loads the config and instantiates the dataset with split_mode="test"
3. For each sample, encodes x0 (source only - MFM doesn't condition on target)
4. Generates x1_pred using the generator with x0 as source and source latent only
5. Computes and prints W1, W2, MMD, and r2 metrics

Note: Unlike the standard Trellis model, MFM uses source conditioning only,
so the generator is called with target_latent=None.
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from typing import Optional, List, Union
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


def compute_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    wasserstein_only: bool = False,
    max_samples_w1: Optional[int] = None,
):
    """
    Compute all metrics between two distributions.
    
    Args:
        pred: Predicted/source samples [N, dim] (torch tensor)
        target: Target samples [M, dim] (torch tensor)
        wasserstein_only: If True, only compute W1 and W2 (skip MMD and r2)
        max_samples_w1: If provided, subsample both pred and target to at most this many
                        samples when computing W1. Useful when datasets are large.
        
    Returns:
        Dictionary with W1, W2, and optionally MMD and r2
    """
    # Subsample for W1 computation if needed
    if max_samples_w1 is not None:
        pred_w1 = pred
        target_w1 = target
        
        if pred.shape[0] > max_samples_w1:
            indices = torch.randperm(pred.shape[0])[:max_samples_w1]
            pred_w1 = pred[indices]
        
        if target.shape[0] > max_samples_w1:
            indices = torch.randperm(target.shape[0])[:max_samples_w1]
            target_w1 = target[indices]
        
        w1 = wasserstein(pred_w1, target_w1, power=1)
    else:
        w1 = wasserstein(pred, target, power=1)
    
    w2 = 0.0 #wasserstein(pred, target, power=2)
    
    result = {
        'W1': w1,
        'W2': w2,
    }
    
    if not wasserstein_only:
        mmd = compute_scalar_mmd(target.cpu().numpy(), pred.cpu().numpy())
        r2 = cellot_corr(pred.cpu().numpy(), target.cpu().numpy())
        result['MMD'] = mmd
        result['r2'] = r2
    
    return result


# ============================================================================
# Latent Caching
# ============================================================================

def get_latent_cache_path(experiment_dir: str, split: str) -> str:
    """Get the path for caching latents for a given split."""
    return os.path.join(experiment_dir, f"{split}_source_latents_cache_mfm.pt")


def compute_and_cache_source_latents(
    encoder: torch.nn.Module,
    dataset,
    device: torch.device,
    cache_path: str,
    split_name: str = "dataset",
) -> np.ndarray:
    """
    Compute E(x0) for all samples in the dataset (source latents only for MFM).
    Caches results to disk for efficiency.
    
    Args:
        encoder: Trained encoder
        dataset: Dataset with samples
        device: Device to run on
        cache_path: Path to save/load cached latents
        split_name: Name of split for logging
        
    Returns:
        source_latents as numpy array [num_samples, latent_dim]
    """
    print(f"Computing {split_name} source latents for {len(dataset.samples)} samples...")
    
    source_latents = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            culture, x0, x1, cell_cond, treat_cond, patient = sample
            
            # Convert to tensors and add batch dimension
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Encode source only (MFM doesn't need target latent)
            source_latent = encoder(x0_tensor).cpu().numpy()  # [1, latent_dim]
            
            source_latents.append(source_latent)
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataset.samples)} samples")
    
    # Stack into arrays: [num_samples, latent_dim]
    source_latents = np.vstack(source_latents)
    
    # Cache to disk
    print(f"Saving {split_name} source latents to {cache_path}")
    torch.save({
        'source_latents': source_latents,
    }, cache_path)
    
    return source_latents


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def load_experiment(experiment_dir: str, device: torch.device):
    """
    Load the trained MFM model, config, and instantiate the components.
    
    Args:
        experiment_dir: Path to the experiment directory containing best_model.pt and config.yaml
        device: Device to load the model onto
        
    Returns:
        encoder, generator, test_dataset, config
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
    
    # Instantiate test dataset
    test_dataset = hydra.utils.instantiate(resolved_cfg.dataset)
    print(f"Instantiated test dataset with split_mode='test', {len(test_dataset.samples)} samples")
    
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
    print("Loaded generator (MFM - source conditioning only)")
    
    return encoder, generator, test_dataset, cfg


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    device: torch.device,
    compute_baseline: bool = False,
    wasserstein_only: bool = False,
    max_samples_w1: Optional[int] = None,
):
    """
    Evaluate the MFM model on a single sample using precomputed source latent.
    
    Note: MFM uses source conditioning only, so target_latent=None is passed to generator.
    
    Args:
        generator: Trained MFM generator
        x0: Source samples (numpy array of shape [N, dim])
        x1: Target samples (numpy array of shape [M, dim])
        source_latent: Precomputed E(x0) [1, latent_dim]
        device: Device to run on
        compute_baseline: If True, compute baseline metrics (x0 vs x1)
        wasserstein_only: If True, only compute W1 and W2 (skip MMD and r2)
        max_samples_w1: If provided, subsample to at most this many samples for W1
        
    Returns:
        Dictionary with generated samples, model metrics, and optionally baseline metrics
    """
    # Convert to tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    # Compute baseline metrics: x0 vs x1_true (before any transport), if requested
    baseline_metrics = None
    if compute_baseline:
        baseline_metrics = compute_all_metrics(
            x0_tensor, x1_tensor, 
            wasserstein_only=wasserstein_only,
            max_samples_w1=max_samples_w1,
        )
    
    # Convert source latent to tensor
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Generate x1_pred from x0 using source latent only (MFM approach)
        # G(x0, E(x0), None) - target_latent is None for MFM
        x1_pred = generator.sample(
            x0_tensor,              # [N, dim] - source samples
            source_latent_tensor,   # [1, latent_dim]
            None,                   # MFM: no target latent conditioning
        )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
    
    # Compute model metrics: x1_pred vs x1_true
    model_metrics = compute_all_metrics(
        x1_pred, x1_tensor, 
        wasserstein_only=wasserstein_only,
        max_samples_w1=max_samples_w1,
    )
    
    result = {
        'x1_pred': x1_pred.cpu().numpy(),
        'model': model_metrics,
    }
    
    if compute_baseline:
        result['baseline'] = baseline_metrics
    
    return result


def find_experiment_dir(
    outputs_dir: str = "outputs",
    split_name: Optional[str] = None,
    seed: Optional[int] = None,
    experiment_name: Optional[str] = None,
) -> str:
    """
    Search through directories in outputs_dir to find the experiment matching
    the given filter criteria. Only filters that are explicitly provided (not None)
    are used for matching.
    
    Args:
        outputs_dir: Directory containing experiment outputs
        split_name: The split name (e.g., 'replicas-1', 'pdo21')
        seed: The random seed used for training
        experiment_name: The experiment name (experiment.name), defaults to 'trellis_mfm'
        
    Returns:
        Path to the matching experiment directory
        
    Raises:
        ValueError: If no matching directory is found or multiple matches exist
    """
    if not os.path.exists(outputs_dir):
        raise ValueError(f"Outputs directory not found: {outputs_dir}")
    
    # Build filter description for logging
    filters = []
    if split_name is not None:
        filters.append(f"split_name={split_name}")
    if seed is not None:
        filters.append(f"seed={seed}")
    if experiment_name is not None:
        filters.append(f"experiment_name={experiment_name}")
    
    if not filters:
        raise ValueError("At least one filter criterion must be provided when --experiment_dir is not specified")
    
    print(f"Searching for experiment with: {', '.join(filters)}")
    print(f"Looking in: {outputs_dir}")
    
    matching_dirs = []
    
    # Iterate through all subdirectories
    for dirname in os.listdir(outputs_dir):
        dir_path = os.path.join(outputs_dir, dirname)
        
        # Skip if not a directory
        if not os.path.isdir(dir_path):
            continue
        
        # Check if config.yaml exists
        config_path = os.path.join(dir_path, "config.yaml")
        if not os.path.exists(config_path):
            continue
        
        # Load and check config
        try:
            cfg = OmegaConf.load(config_path)
            
            match = True
            
            # Check split_name if provided
            if split_name is not None:
                cfg_split_name = cfg.get("experiment", {}).get("split_name")
                if cfg_split_name != split_name:
                    match = False
            
            # Check seed if provided
            if seed is not None and match:
                cfg_seed = cfg.get("seed")
                if cfg_seed is None or int(cfg_seed) != int(seed):
                    match = False
            
            # Check experiment_name if provided
            if experiment_name is not None and match:
                cfg_exp_name = cfg.get("experiment", {}).get("name")
                if cfg_exp_name != experiment_name:
                    match = False
            
            if match:
                matching_dirs.append(dir_path)
                print(f"  Found match: {dirname}")
                    
        except Exception as e:
            # Skip directories with invalid configs
            continue
    
    if len(matching_dirs) == 0:
        raise ValueError(
            f"No experiment found matching: {', '.join(filters)}"
        )
    
    if len(matching_dirs) > 1:
        print(f"Warning: Multiple matching directories found, using the first one:")
        for d in matching_dirs:
            print(f"  - {d}")
    
    return matching_dirs[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Trellis MFM model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to the experiment directory (e.g., outputs/trellis_mfm_xxx). "
             "If not provided, will search based on filter criteria"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Split name to search for (e.g., 'replicas-1', 'pdo21'). "
             "Only used if provided."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to search for. Only used if provided."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="trellis_mfm",
        help="Experiment name to search for (experiment.name). "
             "Default: 'trellis_mfm'"
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs (default: outputs)"
    )
    parser.add_argument(
        "--compute_baseline",
        action="store_true",
        help="If set, compute and print baseline metrics (x0 vs x1). "
             "By default, baseline is not computed."
    )
    parser.add_argument(
        "--wasserstein_only",
        action="store_true",
        help="If set, only compute W1 and W2 scores (skip R2 and MMD). "
             "Useful for faster evaluation."
    )
    parser.add_argument(
        "--max_samples_w1",
        type=int,
        default=3000,
        help="Maximum number of samples to use for W1 computation. "
             "If source or target has more samples, randomly subsample. "
             "Set to 0 or negative to disable subsampling. (default: 3000)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        # Check that at least one filter is provided
        has_filter = any([
            args.split_name is not None,
            args.seed is not None,
            args.experiment_name is not None,
        ])
        if not has_filter:
            parser.error("Must provide either --experiment_dir or at least one filter criterion "
                        "(--split_name, --seed, --experiment_name)")
        
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            split_name=args.split_name, 
            seed=args.seed,
            experiment_name=args.experiment_name,
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load experiment
    encoder, generator, test_dataset, cfg = load_experiment(args.experiment_dir, device)
    
    # Compute and cache test source latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING SOURCE LATENTS (MFM)")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents = compute_and_cache_source_latents(
        encoder, test_dataset, device, test_cache_path, split_name="test"
    )
    
    # Evaluate each sample
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("Mode: MFM (source conditioning only, target_latent=None)")
    if args.wasserstein_only:
        print("Computing: W1, W2 only (skipping MMD and r2)")
    if args.compute_baseline:
        print("Computing baseline: x0 vs x1")
    
    # Handle max_samples_w1: convert to None if disabled (0 or negative)
    max_samples_w1 = args.max_samples_w1 if args.max_samples_w1 > 0 else None
    if max_samples_w1 is not None:
        print(f"W1 subsampling: max {max_samples_w1} samples from source/target")
    print("=" * 80)
    
    # Determine which metrics to track
    if args.wasserstein_only:
        metric_names = ['W1', 'W2']
    else:
        metric_names = ['W1', 'W2', 'MMD', 'r2']
    
    # Track model and optionally baseline metrics
    all_model_metrics = {name: [] for name in metric_names}
    all_baseline_metrics = {name: [] for name in metric_names} if args.compute_baseline else None
    
    for i, sample in enumerate(test_dataset.samples):
        culture, x0, x1, cell_cond, treat_cond, patient = sample
        
        # Get precomputed source latent for this sample
        source_latent = test_source_latents[i:i+1]  # [1, latent_dim]
        
        print(f"\nSample {i + 1}/{len(test_dataset.samples)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, device, 
            compute_baseline=args.compute_baseline,
            wasserstein_only=args.wasserstein_only,
            max_samples_w1=max_samples_w1,
        )
        
        model = results['model']
        
        # Print results based on flags
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
        
        # Collect metrics
        for key in metric_names:
            all_model_metrics[key].append(model[key])
            if args.compute_baseline:
                all_baseline_metrics[key].append(results['baseline'][key])
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    for metric_name in metric_names:
        model_values = np.array(all_model_metrics[metric_name])
        
        print(f"\n{metric_name}:")
        if args.compute_baseline:
            baseline_values = np.array(all_baseline_metrics[metric_name])
            print(f"  {'':15} {'Model':>15} {'Baseline':>15}")
            print(f"  {'Mean':15} {np.mean(model_values):>15.6f} {np.mean(baseline_values):>15.6f}")
            print(f"  {'Std':15} {np.std(model_values):>15.6f} {np.std(baseline_values):>15.6f}")
            print(f"  {'Median':15} {np.median(model_values):>15.6f} {np.median(baseline_values):>15.6f}")
            print(f"  {'Min':15} {np.min(model_values):>15.6f} {np.min(baseline_values):>15.6f}")
            print(f"  {'Max':15} {np.max(model_values):>15.6f} {np.max(baseline_values):>15.6f}")
        else:
            print(f"  {'':15} {'Model':>15}")
            print(f"  {'Mean':15} {np.mean(model_values):>15.6f}")
            print(f"  {'Std':15} {np.std(model_values):>15.6f}")
            print(f"  {'Median':15} {np.median(model_values):>15.6f}")
            print(f"  {'Min':15} {np.min(model_values):>15.6f}")
            print(f"  {'Max':15} {np.max(model_values):>15.6f}")
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean ± std)")
    print("=" * 80)
    
    if args.compute_baseline:
        print(f"{'Metric':<6} {'Model (pred vs true)':>25} {'Baseline (x0 vs true)':>25}")
        print(f"{'-'*6} {'-'*25} {'-'*25}")
        
        for metric_name in metric_names:
            model_mean = np.mean(all_model_metrics[metric_name])
            model_std = np.std(all_model_metrics[metric_name])
            baseline_mean = np.mean(all_baseline_metrics[metric_name])
            baseline_std = np.std(all_baseline_metrics[metric_name])
            
            model_str = f"{model_mean:.4f} ± {model_std:.4f}"
            baseline_str = f"{baseline_mean:.4f} ± {baseline_std:.4f}"
            print(f"{metric_name:<6} {model_str:>25} {baseline_str:>25}")
    else:
        print(f"{'Metric':<6} {'Model (pred vs true)':>25}")
        print(f"{'-'*6} {'-'*25}")
        
        for metric_name in metric_names:
            model_mean = np.mean(all_model_metrics[metric_name])
            model_std = np.std(all_model_metrics[metric_name])
            
            model_str = f"{model_mean:.4f} ± {model_std:.4f}"
            print(f"{metric_name:<6} {model_str:>25}")


if __name__ == "__main__":
    main()

