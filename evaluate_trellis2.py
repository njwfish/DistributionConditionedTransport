"""
Evaluation script for the trained Trellis model.

This script:
1. Loads the trained model from the best checkpoint
2. Loads the config and instantiates the dataset with split_mode="test"
3. For each sample, encodes x0 and x1 from the same sample
4. Generates x1_pred using the generator with x0 as source and the latent of x1
5. Computes and prints W1, W2, MMD, and r2 metrics

Optionally, with --use_predictor:
- Trains a predictor P on training data to map E(x0) -> E(x1)
- Predictor type can be "ridge" (default) or "mlp"
- Uses G(x0, E(x0), P(E(x0))) instead of G(x0, E(x0), E(x1))

Optimization flags:
- --num_ode_steps: Number of ODE integration steps for flow matching (default: 50)
- --compile: Use torch.compile() for PyTorch 2.0+ (experimental)
"""

import os
import sys
import argparse
import math
import time
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import OmegaConf
from typing import Optional, Union
import ot as pot
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import Ridge


# ============================================================================
# Timing Utilities
# ============================================================================

def print_elapsed_time(start_time: float, step_name: str):
    """Print the elapsed time since start_time."""
    elapsed = time.time() - start_time
    print(f"[TIME] {step_name}: {elapsed:.2f}s ({elapsed/60:.2f}m)")
    return time.time()


# ============================================================================
# MLP Predictor
# ============================================================================

class MLPPredictor(nn.Module):
    """MLP predictor to map source latents to target latents."""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        layers = []
        in_dim = latent_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Sklearn-compatible predict method for numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            output = self.forward(x_tensor)
            return output.cpu().numpy()

# ============================================================================
# Metric Functions (from the provided script)
# ============================================================================

# Global settings for fast metric computation (set via command line args)
_METRIC_SETTINGS = {
    'method': 'exact',  # 'exact', 'sinkhorn', or 'sliced'
    'sinkhorn_reg': 0.05,
    'sliced_n_projections': 100,
    'subsample_size': None,  # If set, subsample before computing metrics
}

def set_metric_settings(method='exact', sinkhorn_reg=0.05, sliced_n_projections=100, subsample_size=None):
    """Configure global metric computation settings."""
    _METRIC_SETTINGS['method'] = method
    _METRIC_SETTINGS['sinkhorn_reg'] = sinkhorn_reg
    _METRIC_SETTINGS['sliced_n_projections'] = sliced_n_projections
    _METRIC_SETTINGS['subsample_size'] = subsample_size


def subsample_if_needed(x0: torch.Tensor, x1: torch.Tensor, max_samples: Optional[int] = None):
    """Subsample tensors if they exceed max_samples."""
    if max_samples is None:
        max_samples = _METRIC_SETTINGS['subsample_size']
    
    if max_samples is None:
        return x0, x1
    
    n0, n1 = x0.shape[0], x1.shape[0]
    
    if n0 > max_samples:
        idx = torch.randperm(n0)[:max_samples]
        x0 = x0[idx]
    
    if n1 > max_samples:
        idx = torch.randperm(n1)[:max_samples]
        x1 = x1[idx]
    
    return x0, x1


def sliced_wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    n_projections: int = 100,
    power: int = 2,
    device: Optional[torch.device] = None,
) -> float:
    """
    Compute Sliced Wasserstein distance (much faster than exact OT).
    
    Projects distributions onto random 1D lines and averages the 1D Wasserstein distances.
    This is O(n log n) per projection instead of O(n³) for exact OT.
    
    Args:
        x0: Source samples [N, dim]
        x1: Target samples [M, dim]  
        n_projections: Number of random projections
        power: 1 for W1, 2 for W2
        device: Device for computation
        
    Returns:
        Sliced Wasserstein distance
    """
    if device is None:
        device = x0.device
    
    dim = x0.shape[1]
    n0, n1 = x0.shape[0], x1.shape[0]
    
    # Generate random projection directions (unit vectors)
    projections = torch.randn(n_projections, dim, device=device)
    projections = projections / projections.norm(dim=1, keepdim=True)
    
    # Project both distributions: [n_projections, N] and [n_projections, M]
    x0_proj = projections @ x0.T  # [n_projections, N]
    x1_proj = projections @ x1.T  # [n_projections, M]
    
    # Sort the projections
    x0_sorted, _ = torch.sort(x0_proj, dim=1)  # [n_projections, N]
    x1_sorted, _ = torch.sort(x1_proj, dim=1)  # [n_projections, M]
    
    # Interpolate to same size if different
    if n0 != n1:
        # Use linear interpolation to make sizes equal
        target_size = max(n0, n1)
        
        if n0 < target_size:
            # Interpolate x0_sorted
            idx = torch.linspace(0, n0 - 1, target_size, device=device)
            idx_floor = idx.floor().long().clamp(0, n0 - 1)
            idx_ceil = idx.ceil().long().clamp(0, n0 - 1)
            weight = (idx - idx_floor.float()).unsqueeze(0)
            x0_sorted = x0_sorted[:, idx_floor] * (1 - weight) + x0_sorted[:, idx_ceil] * weight
        
        if n1 < target_size:
            idx = torch.linspace(0, n1 - 1, target_size, device=device)
            idx_floor = idx.floor().long().clamp(0, n1 - 1)
            idx_ceil = idx.ceil().long().clamp(0, n1 - 1)
            weight = (idx - idx_floor.float()).unsqueeze(0)
            x1_sorted = x1_sorted[:, idx_floor] * (1 - weight) + x1_sorted[:, idx_ceil] * weight
    
    # Compute 1D Wasserstein distances
    if power == 1:
        distances = torch.abs(x0_sorted - x1_sorted).mean(dim=1)
    else:  # power == 2
        distances = ((x0_sorted - x1_sorted) ** 2).mean(dim=1)
    
    # Average over projections
    mean_dist = distances.mean().item()
    
    if power == 2:
        mean_dist = math.sqrt(mean_dist)
    
    return mean_dist


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    assert power == 1 or power == 2
    
    # Use global method setting if not specified
    if method is None:
        method = _METRIC_SETTINGS['method']
        reg = _METRIC_SETTINGS['sinkhorn_reg']
    
    # Use sliced Wasserstein if configured
    if method == "sliced":
        return sliced_wasserstein(
            x0, x1, 
            n_projections=_METRIC_SETTINGS['sliced_n_projections'],
            power=power
        )
    
    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    M_np = M.detach().cpu().numpy()
    
    if method == "exact":
        ret = pot.emd2(a, b, M_np, numItermax=int(1e7))
    elif method == "sinkhorn":
        # Normalize cost matrix to avoid numerical issues
        # Scale reg relative to median cost
        M_median = np.median(M_np[M_np > 0]) if np.any(M_np > 0) else 1.0
        reg_scaled = reg * M_median
        ret = pot.sinkhorn2(a, b, M_np, reg=reg_scaled, numItermax=int(1e4), warn=False)
        # sinkhorn2 can return array-like, ensure we get a scalar
        if hasattr(ret, '__len__'):
            ret = float(ret[0]) if len(ret) > 0 else 0.0
        else:
            ret = float(ret)
    else:
        raise ValueError(f"Unknown method: {method}")

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
    # Apply subsampling if configured
    pred_sub, target_sub = subsample_if_needed(pred, target)
    
    w1 = wasserstein(pred_sub, target_sub, power=1)
    w2 = wasserstein(pred_sub, target_sub, power=2)
    mmd = compute_scalar_mmd(target_sub.cpu().numpy(), pred_sub.cpu().numpy())
    r2 = cellot_corr(pred.cpu().numpy(), target.cpu().numpy())  # Use full data for correlation
    
    return {
        'W1': w1,
        'W2': w2,
        'MMD': mmd,
        'r2': r2,
    }


# ============================================================================
# Latent Caching and Predictor Training
# ============================================================================

def get_latent_cache_path(experiment_dir: str, split: str) -> str:
    """Get the path for caching latents for a given split."""
    return os.path.join(experiment_dir, f"{split}_latents_cache.pt")


def compute_and_cache_latents(
    encoder: torch.nn.Module,
    dataset,
    device: torch.device,
    cache_path: str,
    split_name: str = "dataset",
) -> tuple:
    """
    Compute E(x0) and E(x1) for all samples in the dataset.
    Processes samples sequentially since they have variable sizes.
    Caches results to disk for efficiency.
    
    Args:
        encoder: Trained encoder
        dataset: Dataset with samples
        device: Device to run on
        cache_path: Path to save/load cached latents
        split_name: Name of split for logging
        
    Returns:
        Tuple of (source_latents, target_latents) as numpy arrays [num_samples, latent_dim]
    """
    step_start = time.time()
    
    # Check if cache exists
    if os.path.exists(cache_path):
        print(f"Loading cached {split_name} latents from {cache_path}")
        cache = torch.load(cache_path, map_location='cpu', weights_only=False)
        print_elapsed_time(step_start, f"Loading {split_name} latents from cache")
        return cache['source_latents'], cache['target_latents']
    
    print(f"Computing {split_name} latents for {len(dataset.samples)} samples...")
    compute_start = time.time()
    
    source_latents = []
    target_latents = []
    
    encoder.eval()
    num_samples = len(dataset.samples)
    
    with torch.no_grad():
        for i, sample in enumerate(dataset.samples):
            culture, x0, x1, cell_cond, treat_cond, patient = sample
            
            # Convert to tensors and add batch dimension
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Encode
            source_latent = encoder(x0_tensor).cpu().numpy()  # [1, latent_dim]
            target_latent = encoder(x1_tensor).cpu().numpy()  # [1, latent_dim]
            
            source_latents.append(source_latent)
            target_latents.append(target_latent)
            
            if (i + 1) % 50 == 0 or (i + 1) == num_samples:
                print(f"  Encoded {i + 1}/{num_samples} samples")
    
    print_elapsed_time(compute_start, f"Computing {split_name} encodings")
    
    # Stack into arrays: [num_samples, latent_dim]
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    
    # Cache to disk
    print(f"Saving {split_name} latents to {cache_path}")
    save_start = time.time()
    torch.save({
        'source_latents': source_latents,
        'target_latents': target_latents,
    }, cache_path)
    print_elapsed_time(save_start, f"Saving {split_name} latents to cache")
    
    print_elapsed_time(step_start, f"Total time for {split_name} latents")
    
    return source_latents, target_latents


def train_ridge_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """
    Train a ridge regression predictor to map source latents to target latents.
    
    Args:
        source_latents: [num_samples, latent_dim] source latent encodings E(x0)
        target_latents: [num_samples, latent_dim] target latent encodings E(x1)
        alpha: Ridge regularization strength
        
    Returns:
        Trained Ridge regression model
    """
    train_start = time.time()
    print(f"Training ridge regression predictor (alpha={alpha})...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    
    predictor = Ridge(alpha=alpha)
    predictor.fit(source_latents, target_latents)
    
    # Compute training R^2
    train_r2 = predictor.score(source_latents, target_latents)
    print(f"  Training R^2: {train_r2:.4f}")
    
    print_elapsed_time(train_start, "Training ridge predictor")
    
    return predictor


def train_mlp_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    device: torch.device,
    hidden_dim: int = 128,
    num_layers: int = 2,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    batch_size: int = 32,
) -> MLPPredictor:
    """
    Train an MLP predictor to map source latents to target latents.
    
    Args:
        source_latents: [num_samples, latent_dim] source latent encodings E(x0)
        target_latents: [num_samples, latent_dim] target latent encodings E(x1)
        device: Device to train on
        hidden_dim: Hidden dimension of MLP
        num_layers: Number of hidden layers
        lr: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Trained MLPPredictor model
    """
    train_start = time.time()
    latent_dim = source_latents.shape[1]
    num_samples = source_latents.shape[0]
    
    print(f"Training MLP predictor...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    print(f"  Architecture: {latent_dim} -> {num_layers}x{hidden_dim} -> {latent_dim}")
    print(f"  lr={lr}, epochs={num_epochs}, batch_size={batch_size}")
    
    # Create model
    predictor = MLPPredictor(latent_dim, hidden_dim, num_layers).to(device)
    
    # Convert data to tensors
    source_tensor = torch.tensor(source_latents, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_latents, dtype=torch.float32, device=device)
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    predictor.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Shuffle data
        perm = torch.randperm(num_samples)
        source_shuffled = source_tensor[perm]
        target_shuffled = target_tensor[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_source = source_shuffled[i:i+batch_size]
            batch_target = target_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = predictor(batch_source)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    # Compute final training metrics
    predictor.eval()
    with torch.no_grad():
        pred = predictor(source_tensor)
        final_loss = criterion(pred, target_tensor).item()
        
        # Compute R^2-like metric
        ss_res = ((pred - target_tensor) ** 2).sum().item()
        ss_tot = ((target_tensor - target_tensor.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"  Final MSE: {final_loss:.6f}, R^2: {r2:.4f}")
    
    print_elapsed_time(train_start, "Training MLP predictor")
    
    return predictor


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def load_experiment(experiment_dir: str, device: torch.device, load_train_dataset: bool = False):
    """
    Load the trained model, config, and instantiate the components.
    
    Args:
        experiment_dir: Path to the experiment directory containing best_model.pt and config.yaml
        device: Device to load the model onto
        load_train_dataset: If True, also instantiate training dataset for predictor training
        
    Returns:
        encoder, generator, test_dataset, config, train_dataset (or None if not requested)
    """
    load_start = time.time()
    
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
    
    # Instantiate training dataset if needed (keep split_mode="train")
    train_dataset = None
    if load_train_dataset:
        train_cfg = OmegaConf.create(OmegaConf.to_container(resolved_cfg, resolve=True))
        train_cfg.dataset.split_mode = "train"
        train_dataset = hydra.utils.instantiate(train_cfg.dataset)
        print(f"Instantiated training dataset with split_mode='train', {len(train_dataset.samples)} samples")
    
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
    print("Loaded generator")
    
    print_elapsed_time(load_start, "Loading experiment (model + datasets)")
    
    return encoder, generator, test_dataset, cfg, train_dataset


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor: Optional[Union[Ridge, MLPPredictor]] = None,
    verbose: bool = False,
    num_ode_steps: int = 100,
    compute_baseline: bool = True,
):
    """
    Evaluate the model on a single sample using precomputed latents.
    
    Args:
        generator: Trained generator
        x0: Source samples (numpy array of shape [N, dim])
        x1: Target samples (numpy array of shape [M, dim])
        source_latent: Precomputed E(x0) [1, latent_dim]
        target_latent: Precomputed E(x1) [1, latent_dim]
        device: Device to run on
        predictor: Optional predictor (Ridge or MLPPredictor). If provided, uses P(E(x0)) 
                   as target latent instead of E(x1)
        verbose: If True, print timing information
        num_ode_steps: Number of ODE integration steps (default: 100, try 50 for faster inference)
        compute_baseline: If True, compute baseline metrics (x0 vs x1_true)
        
    Returns:
        Dictionary with generated samples, model metrics, baseline metrics (or None), and timings
    """
    timings = {}
    
    # Convert to tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    # Compute baseline metrics: x0 vs x1_true (before any transport)
    if compute_baseline:
        baseline_start = time.time()
        baseline_metrics = compute_all_metrics(x0_tensor, x1_tensor)
        timings['baseline_metrics'] = time.time() - baseline_start
    else:
        baseline_metrics = None
        timings['baseline_metrics'] = 0.0
    
    # Convert latents to tensors
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    # Get target latent: either from predictor or use precomputed
    if predictor is not None:
        # Use predictor: P(E(x0)) - both Ridge and MLPPredictor have .predict() method
        pred_start = time.time()
        predicted_target_latent = predictor.predict(source_latent)
        timings['predictor'] = time.time() - pred_start
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        # Use precomputed E(x1)
        target_latent_tensor = torch.tensor(target_latent, dtype=torch.float32, device=device)
        timings['predictor'] = 0.0
    
    with torch.no_grad():
        # Generate x1_pred from x0 using target latent
        # G(x0, E(x0), target_latent) where target_latent is either E(x1) or P(E(x0))
        gen_start = time.time()
        
        x1_pred = generator.sample(
            x0_tensor,              # [N, dim] - source samples
            source_latent_tensor,   # [1, latent_dim]
            target_latent_tensor,   # [1, latent_dim]
            num_steps=num_ode_steps,
        )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
        timings['generation'] = time.time() - gen_start
    
    # Compute model metrics: x1_pred vs x1_true
    metrics_start = time.time()
    model_metrics = compute_all_metrics(x1_pred, x1_tensor)
    timings['model_metrics'] = time.time() - metrics_start
    
    if verbose:
        if compute_baseline:
            print(f"    [TIME] Baseline metrics: {timings['baseline_metrics']:.3f}s")
        if predictor is not None:
            print(f"    [TIME] Predictor: {timings['predictor']:.3f}s")
        print(f"    [TIME] Generation: {timings['generation']:.3f}s")
        print(f"    [TIME] Model metrics: {timings['model_metrics']:.3f}s")
    
    return {
        'x1_pred': x1_pred.cpu().numpy(),
        'model': model_metrics,
        'baseline': baseline_metrics,
        'timings': timings,
    }


def find_experiment_dir(
    split_name: str, 
    outputs_dir: str = "outputs",
    predictor_loss_weight: Optional[float] = None,
    seed: Optional[int] = None,
    experiment_name: Optional[str] = None,
    selective_pairing_mode: Optional[str] = None,
) -> str:
    """
    Search through directories in outputs_dir to find the experiment matching
    the given criteria. Only filters by optional parameters if they are provided.
    
    Args:
        split_name: The split name (e.g., 'replicas-1', 'pdo21') - required
        outputs_dir: Directory containing experiment outputs
        predictor_loss_weight: The predictor loss weight (e.g., 1, 0.1, 0.01, 0.0) - optional
        seed: The random seed used in the experiment - optional
        experiment_name: The experiment name (e.g., 'trellis', 'trellis_mfm') - optional
        selective_pairing_mode: The selective pairing mode (e.g., 'single_step', 'null') - optional
        
    Returns:
        Path to the matching experiment directory
        
    Raises:
        ValueError: If no matching directory is found or multiple matches exist
    """
    if not os.path.exists(outputs_dir):
        raise ValueError(f"Outputs directory not found: {outputs_dir}")
    
    # Build search criteria string for logging
    criteria = [f"split_name={split_name}"]
    if predictor_loss_weight is not None:
        criteria.append(f"predictor_loss_weight={predictor_loss_weight}")
    if seed is not None:
        criteria.append(f"seed={seed}")
    if experiment_name is not None:
        criteria.append(f"experiment.name={experiment_name}")
    if selective_pairing_mode is not None:
        criteria.append(f"selective_pairing_mode={selective_pairing_mode}")
    
    print(f"Searching for experiment with {', '.join(criteria)}")
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
            
            # Check if split_name matches (required)
            cfg_split_name = cfg.get("experiment", {}).get("split_name")
            if cfg_split_name != split_name:
                continue
            
            # Check predictor_loss_weight if specified
            if predictor_loss_weight is not None:
                cfg_weight = cfg.get("experiment", {}).get("predictor_loss_weight")
                if cfg_weight is None:
                    continue
                # Compare weights with tolerance for floating point
                if abs(float(cfg_weight) - float(predictor_loss_weight)) >= 1e-6:
                    continue
            
            # Check seed if specified
            if seed is not None:
                cfg_seed = cfg.get("seed")
                if cfg_seed is None or int(cfg_seed) != seed:
                    continue
            
            # Check experiment.name if specified
            if experiment_name is not None:
                cfg_exp_name = cfg.get("experiment", {}).get("name")
                if cfg_exp_name != experiment_name:
                    continue
            
            # Check selective_pairing_mode if specified
            if selective_pairing_mode is not None:
                cfg_spm = cfg.get("experiment", {}).get("selective_pairing_mode")
                # Handle both string and None/null values in config
                cfg_spm_str = str(cfg_spm) if cfg_spm is not None else "null"
                if cfg_spm_str != selective_pairing_mode and cfg_spm != selective_pairing_mode:
                    continue
            
            # All criteria matched
            matching_dirs.append(dir_path)
            print(f"  Found match: {dirname}")
            
        except Exception as e:
            # Skip directories with invalid configs
            continue
    
    if len(matching_dirs) == 0:
        raise ValueError(
            f"No experiment found with {', '.join(criteria)}"
        )
    
    if len(matching_dirs) > 1:
        print(f"Warning: Multiple matching directories found, using the first one:")
        for d in matching_dirs:
            print(f"  - {d}")
    
    return matching_dirs[0]


def main():
    script_start = time.time()
    
    parser = argparse.ArgumentParser(description="Evaluate trained Trellis model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to the experiment directory (e.g., outputs/trellis_a2a_replicas-2_xxx). "
             "If not provided, will search based on --split_name and optionally "
             "--predictor_loss_weight, --seed, --experiment_name, --selective_pairing_mode"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Split name to search for (e.g., 'replicas-1', 'pdo21'). "
             "Required if --experiment_dir is not provided."
    )
    parser.add_argument(
        "--predictor_loss_weight",
        type=float,
        default=None,
        help="Predictor loss weight to search for (e.g., 1, 0.1, 0.01, 0.0). "
             "Optional filter when searching for experiments."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to search for in experiment config. "
             "Optional filter when searching for experiments."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name to search for (e.g., 'trellis', 'trellis_mfm'). "
             "Matches experiment.name in config. Optional filter when searching."
    )
    parser.add_argument(
        "--selective_pairing_mode",
        type=str,
        default=None,
        help="Selective pairing mode to search for (e.g., 'single_step', 'null'). "
             "Matches experiment.selective_pairing_mode in config. Optional filter."
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        default="outputs",
        help="Directory containing experiment outputs (default: outputs)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--use_predictor",
        action="store_true",
        help="Use a predictor P to predict target latent from source latent. "
             "Uses G(x0, E(x0), P(E(x0))) instead of G(x0, E(x0), E(x1))"
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["ridge", "mlp"],
        default="ridge",
        help="Type of predictor to use: 'ridge' for ridge regression, 'mlp' for neural network (default: ridge)"
    )
    # Ridge-specific arguments
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularization strength (default: 1.0)"
    )
    # MLP-specific arguments
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for MLP predictor (default: 128)"
    )
    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=2,
        help="Number of hidden layers for MLP predictor (default: 2)"
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="Learning rate for MLP predictor (default: 1e-3)"
    )
    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=1000,
        help="Number of training epochs for MLP predictor (default: 1000)"
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=32,
        help="Batch size for MLP predictor training (default: 32)"
    )
    
    # === PERFORMANCE OPTIMIZATION ARGUMENTS ===
    parser.add_argument(
        "--num_ode_steps",
        type=int,
        default=50,
        help="Number of ODE integration steps for flow matching (default: 50). "
             "Original default is 100. Lower values are faster but may reduce quality."
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for encoder/generator (PyTorch 2.0+, experimental)."
    )
    parser.add_argument(
        "--skip_baseline",
        action="store_true",
        help="Skip computing baseline metrics (x0 vs x1_true). Saves time if only model metrics are needed."
    )
    
    # === FAST METRIC COMPUTATION ARGUMENTS ===
    parser.add_argument(
        "--metric_method",
        type=str,
        choices=["exact", "sinkhorn", "sliced"],
        default="exact",
        help="Method for Wasserstein distance: 'exact' (slow, accurate), "
             "'sinkhorn' (fast, entropic regularization), 'sliced' (very fast, projection-based). "
             "Recommended: 'sliced' for ~50-100x speedup with minimal accuracy loss."
    )
    parser.add_argument(
        "--sinkhorn_reg",
        type=float,
        default=0.05,
        help="Regularization parameter for Sinkhorn algorithm (default: 0.05). "
             "Smaller values = more accurate but slower."
    )
    parser.add_argument(
        "--sliced_projections",
        type=int,
        default=100,
        help="Number of random projections for sliced Wasserstein (default: 100). "
             "More projections = more accurate but slower."
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Subsample to this many cells before computing metrics (default: None = no subsampling). "
             "Recommended: 1000-2000 for fast approximate metrics. "
             "Example: --subsample 1000 reduces 8634 cells to 1000 for metric computation."
    )
    
    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        if args.split_name is None:
            parser.error("Must provide either --experiment_dir or --split_name")
        
        args.experiment_dir = find_experiment_dir(
            args.split_name,
            outputs_dir=args.outputs_dir,
            predictor_loss_weight=args.predictor_loss_weight,
            seed=args.seed,
            experiment_name=args.experiment_name,
            selective_pairing_mode=args.selective_pairing_mode,
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    print("\n" + "=" * 80)
    print("LOADING EXPERIMENT")
    print("=" * 80)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Optimization settings: num_ode_steps={args.num_ode_steps}, compile={args.compile}")
    
    # Configure fast metric computation
    set_metric_settings(
        method=args.metric_method,
        sinkhorn_reg=args.sinkhorn_reg,
        sliced_n_projections=args.sliced_projections,
        subsample_size=args.subsample,
    )
    metric_info = f"Metric settings: method={args.metric_method}"
    if args.metric_method == "sinkhorn":
        metric_info += f", reg={args.sinkhorn_reg}"
    elif args.metric_method == "sliced":
        metric_info += f", projections={args.sliced_projections}"
    if args.subsample:
        metric_info += f", subsample={args.subsample}"
    print(metric_info)
    
    # Load experiment (also load training dataset if using predictor)
    encoder, generator, test_dataset, cfg, train_dataset = load_experiment(
        args.experiment_dir, device, load_train_dataset=args.use_predictor
    )
    
    # Optional: compile models for PyTorch 2.0+
    if args.compile:
        print("Compiling encoder and generator with torch.compile()...")
        try:
            encoder = torch.compile(encoder)
            generator = torch.compile(generator)
            print("  Compilation successful!")
        except Exception as e:
            print(f"  Warning: torch.compile() failed: {e}")
            print("  Continuing without compilation.")
    
    # Compute and cache test latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING LATENTS")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents, test_target_latents = compute_and_cache_latents(
        encoder, test_dataset, device, test_cache_path, split_name="test"
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print(f"TRAINING PREDICTOR ({args.predictor_type.upper()})")
        print("=" * 80)
        
        # Compute and cache training latents
        train_cache_path = get_latent_cache_path(args.experiment_dir, "train")
        train_source_latents, train_target_latents = compute_and_cache_latents(
            encoder, train_dataset, device, train_cache_path, split_name="train"
        )
        
        # Train predictor based on type
        if args.predictor_type == "ridge":
            predictor = train_ridge_predictor(
                train_source_latents, 
                train_target_latents, 
                alpha=args.ridge_alpha
            )
        elif args.predictor_type == "mlp":
            predictor = train_mlp_predictor(
                train_source_latents,
                train_target_latents,
                device=device,
                hidden_dim=args.mlp_hidden_dim,
                num_layers=args.mlp_num_layers,
                lr=args.mlp_lr,
                num_epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
            )
        
        print(f"\nUsing {args.predictor_type} predictor: G(x0, E(x0), P(E(x0)))")
    else:
        print(f"\nUsing oracle target: G(x0, E(x0), E(x1))")
    
    # Evaluate each sample
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if args.use_predictor:
        print(f"Mode: Using {args.predictor_type.upper()} predictor P(E(x0)) as target latent")
    else:
        print("Mode: Using oracle E(x1) as target latent")
    if args.skip_baseline:
        print("Baseline comparison: SKIPPED")
    print("=" * 80)
    
    compute_baseline = not args.skip_baseline
    
    # Track both model and baseline metrics
    all_model_metrics = {'W1': [], 'W2': [], 'MMD': [], 'r2': []}
    all_baseline_metrics = {'W1': [], 'W2': [], 'MMD': [], 'r2': []} if compute_baseline else None
    
    # Track timings across samples
    all_timings = {'baseline_metrics': [], 'predictor': [], 'generation': [], 'model_metrics': []}
    
    eval_start = time.time()
    
    num_test_samples = len(test_dataset.samples)
    
    for i, sample in enumerate(test_dataset.samples):
        sample_start = time.time()
        culture, x0, x1, cell_cond, treat_cond, patient = sample
        
        # Get precomputed latents for this sample
        source_latent = test_source_latents[i:i+1]  # [1, latent_dim]
        target_latent = test_target_latents[i:i+1]  # [1, latent_dim]
        
        print(f"\nSample {i + 1}/{num_test_samples}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, device, 
            predictor=predictor, verbose=True,
            num_ode_steps=args.num_ode_steps,
            compute_baseline=compute_baseline
        )
        
        model = results['model']
        baseline = results['baseline']
        timings = results['timings']
        
        if compute_baseline:
            print(f"  {'Metric':<6} {'Model (pred vs true)':>20} {'Baseline (x0 vs true)':>22}")
            print(f"  {'-'*6} {'-'*20} {'-'*22}")
            print(f"  {'W1':<6} {model['W1']:>20.6f} {baseline['W1']:>22.6f}")
            print(f"  {'W2':<6} {model['W2']:>20.6f} {baseline['W2']:>22.6f}")
            print(f"  {'MMD':<6} {model['MMD']:>20.6f} {baseline['MMD']:>22.6f}")
            print(f"  {'r2':<6} {model['r2']:>20.6f} {baseline['r2']:>22.6f}")
        else:
            print(f"  {'Metric':<6} {'Model (pred vs true)':>20}")
            print(f"  {'-'*6} {'-'*20}")
            print(f"  {'W1':<6} {model['W1']:>20.6f}")
            print(f"  {'W2':<6} {model['W2']:>20.6f}")
            print(f"  {'MMD':<6} {model['MMD']:>20.6f}")
            print(f"  {'r2':<6} {model['r2']:>20.6f}")
        
        sample_elapsed = time.time() - sample_start
        print(f"  [TIME] Total sample time: {sample_elapsed:.3f}s")
        
        # Collect metrics
        for key in all_model_metrics:
            all_model_metrics[key].append(model[key])
            if compute_baseline:
                all_baseline_metrics[key].append(baseline[key])
        
        # Collect timings
        for key in all_timings:
            all_timings[key].append(timings[key])
    
    print_elapsed_time(eval_start, "Total evaluation time (all samples)")
    
    # Print timing summary
    print("\n" + "=" * 80)
    print("TIMING SUMMARY (per sample)")
    print("=" * 80)
    print(f"{'Step':<25} {'Mean':>12} {'Std':>12} {'Total':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    
    for step_name, times in all_timings.items():
        times_array = np.array(times)
        mean_time = np.mean(times_array)
        std_time = np.std(times_array)
        total_time = np.sum(times_array)
        
        step_label = step_name.replace('_', ' ').title()
        print(f"{step_label:<25} {mean_time:>11.3f}s {std_time:>11.3f}s {total_time:>11.3f}s")
    
    # Print summary
    print("\n" + "=" * 80)
    print("METRIC SUMMARY STATISTICS")
    print("=" * 80)
    
    for metric_name in ['W1', 'W2', 'MMD', 'r2']:
        model_values = np.array(all_model_metrics[metric_name])
        
        print(f"\n{metric_name}:")
        if compute_baseline:
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
    
    if compute_baseline:
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
    else:
        print(f"{'Metric':<6} {'Model (pred vs true)':>25}")
        print(f"{'-'*6} {'-'*25}")
        
        for metric_name in ['W1', 'W2', 'MMD', 'r2']:
            model_mean = np.mean(all_model_metrics[metric_name])
            model_std = np.std(all_model_metrics[metric_name])
            
            model_str = f"{model_mean:.4f} ± {model_std:.4f}"
            print(f"{metric_name:<6} {model_str:>25}")
    
    # Print overall script timing
    print("\n" + "=" * 80)
    print_elapsed_time(script_start, "TOTAL SCRIPT EXECUTION TIME")
    print("=" * 80)


if __name__ == "__main__":
    main()

