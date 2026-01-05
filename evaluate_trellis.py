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
from sklearn.linear_model import Ridge


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


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, wasserstein_only: bool = False):
    """
    Compute all metrics between two distributions.
    
    Args:
        pred: Predicted/source samples [N, dim] (torch tensor)
        target: Target samples [M, dim] (torch tensor)
        wasserstein_only: If True, only compute W1 and W2 (skip MMD and r2)
        
    Returns:
        Dictionary with W1, W2, and optionally MMD and r2
    """
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
    ## Check if cache exists
    #if os.path.exists(cache_path):
    #    print(f"Loading cached {split_name} latents from {cache_path}")
    #    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    #    return cache['source_latents'], cache['target_latents']
    
    print(f"Computing {split_name} latents for {len(dataset.samples)} samples...")
    
    source_latents = []
    target_latents = []
    
    encoder.eval()
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
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(dataset.samples)} samples")
    
    # Stack into arrays: [num_samples, latent_dim]
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    
    # Cache to disk
    print(f"Saving {split_name} latents to {cache_path}")
    torch.save({
        'source_latents': source_latents,
        'target_latents': target_latents,
    }, cache_path)
    
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
    print(f"Training ridge regression predictor (alpha={alpha})...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    
    predictor = Ridge(alpha=alpha)
    predictor.fit(source_latents, target_latents)
    
    # Compute training R^2
    train_r2 = predictor.score(source_latents, target_latents)
    print(f"  Training R^2: {train_r2:.4f}")
    
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
    
    return encoder, generator, test_dataset, cfg, train_dataset


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor: Optional[Union[Ridge, MLPPredictor]] = None,
    compute_baseline: bool = False,
    wasserstein_only: bool = False,
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
        compute_baseline: If True, compute baseline metrics (x0 vs x1)
        wasserstein_only: If True, only compute W1 and W2 (skip MMD and r2)
        
    Returns:
        Dictionary with generated samples, model metrics, and optionally baseline metrics
    """
    # Convert to tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    # Compute baseline metrics: x0 vs x1_true (before any transport), if requested
    baseline_metrics = None
    if compute_baseline:
        baseline_metrics = compute_all_metrics(x0_tensor, x1_tensor, wasserstein_only=wasserstein_only)
    
    # Convert latents to tensors
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    # Get target latent: either from predictor or use precomputed
    if predictor is not None:
        # Use predictor: P(E(x0)) - both Ridge and MLPPredictor have .predict() method
        predicted_target_latent = predictor.predict(source_latent)
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        # Use precomputed E(x1)
        target_latent_tensor = torch.tensor(target_latent, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Generate x1_pred from x0 using target latent
        # G(x0, E(x0), target_latent) where target_latent is either E(x1) or P(E(x0))
        x1_pred = generator.sample(
            x0_tensor,              # [N, dim] - source samples
            source_latent_tensor,   # [1, latent_dim]
            target_latent_tensor,   # [1, latent_dim]
        )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
    
    # Compute model metrics: x1_pred vs x1_true
    model_metrics = compute_all_metrics(x1_pred, x1_tensor, wasserstein_only=wasserstein_only)
    
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
    predictor_loss_weight: Optional[float] = None,
    seed: Optional[int] = None,
    selective_pairing_mode: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> str:
    """
    Search through directories in outputs_dir to find the experiment matching
    the given filter criteria. Only filters that are explicitly provided (not None)
    are used for matching.
    
    Args:
        outputs_dir: Directory containing experiment outputs
        split_name: The split name (e.g., 'replicas-1', 'pdo21')
        predictor_loss_weight: The predictor loss weight (e.g., 1, 0.1, 0.01, 0.0)
        seed: The random seed used for training
        selective_pairing_mode: The selective pairing mode (experiment.selective_pairing_mode)
        experiment_name: The experiment name (experiment.name)
        
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
    if predictor_loss_weight is not None:
        filters.append(f"predictor_loss_weight={predictor_loss_weight}")
    if seed is not None:
        filters.append(f"seed={seed}")
    if selective_pairing_mode is not None:
        filters.append(f"selective_pairing_mode={selective_pairing_mode}")
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
            
            # Check predictor_loss_weight if provided
            if predictor_loss_weight is not None and match:
                cfg_weight = cfg.get("experiment", {}).get("predictor_loss_weight")
                if cfg_weight is None or abs(float(cfg_weight) - float(predictor_loss_weight)) >= 1e-6:
                    match = False
            
            # Check seed if provided
            if seed is not None and match:
                cfg_seed = cfg.get("seed")
                if cfg_seed is None or int(cfg_seed) != int(seed):
                    match = False
            
            # Check selective_pairing_mode if provided
            if selective_pairing_mode is not None and match:
                cfg_pairing_mode = cfg.get("experiment", {}).get("selective_pairing_mode")
                if cfg_pairing_mode != selective_pairing_mode:
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
    parser = argparse.ArgumentParser(description="Evaluate trained Trellis model")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to the experiment directory (e.g., outputs/trellis_a2a_replicas-2_xxx). "
             "If not provided, will search based on --split_name and --predictor_loss_weight"
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Split name to search for (e.g., 'replicas-1', 'pdo21'). "
             "Only used if provided."
    )
    parser.add_argument(
        "--predictor_loss_weight",
        type=float,
        default=None,
        help="Predictor loss weight to search for (e.g., 1, 0.1, 0.01, 0.0). "
             "Only used if provided."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to search for. Only used if provided."
    )
    parser.add_argument(
        "--selective_pairing_mode",
        type=str,
        default=None,
        help="Selective pairing mode to search for (experiment.selective_pairing_mode). "
             "Only used if provided."
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name to search for (experiment.name). "
             "Only used if provided."
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
    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        # Check that at least one filter is provided
        has_filter = any([
            args.split_name is not None,
            args.predictor_loss_weight is not None,
            args.seed is not None,
            args.selective_pairing_mode is not None,
            args.experiment_name is not None,
        ])
        if not has_filter:
            parser.error("Must provide either --experiment_dir or at least one filter criterion "
                        "(--split_name, --predictor_loss_weight, --seed, --selective_pairing_mode, --experiment_name)")
        
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            split_name=args.split_name, 
            predictor_loss_weight=args.predictor_loss_weight,
            seed=args.seed,
            selective_pairing_mode=args.selective_pairing_mode,
            experiment_name=args.experiment_name,
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load experiment (also load training dataset if using predictor)
    encoder, generator, test_dataset, cfg, train_dataset = load_experiment(
        args.experiment_dir, device, load_train_dataset=args.use_predictor
    )
    
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
    if args.wasserstein_only:
        print("Computing: W1, W2 only (skipping MMD and r2)")
    if args.compute_baseline:
        print("Computing baseline: x0 vs x1")
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
        
        # Get precomputed latents for this sample
        source_latent = test_source_latents[i:i+1]  # [1, latent_dim]
        target_latent = test_target_latents[i:i+1]  # [1, latent_dim]
        
        print(f"\nSample {i + 1}/{len(test_dataset.samples)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, device, 
            predictor=predictor,
            compute_baseline=args.compute_baseline,
            wasserstein_only=args.wasserstein_only,
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

