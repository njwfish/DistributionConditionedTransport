"""
Evaluation script for the trained Trellis model WITH CONDITIONING.

This script is a modified version of evaluate_trellis_a2a.py that handles:
1. Concatenating cell_cond with samples before feeding to encoder (43 -> 45 dim)
2. Conditioning the predictor on treat_cond (latent_dim -> latent_dim + 11 dim)

This script:
1. Loads the trained model from the best checkpoint
2. Loads the config and instantiates the dataset with split_mode="test"
3. For each sample, encodes x0 and x1 from the same sample (with cell_cond)
4. Generates x1_pred using the generator with x0 as source and the latent of x1
5. Computes and prints W1, W2, MMD, and r2 metrics

Optionally, with --use_predictor:
- Trains a predictor P on training data to map [E(x0), treat_cond] -> E(x1)
- Predictor type can be "ridge" (default) or "mlp"
- Uses G(x0, E(x0), P([E(x0), treat_cond])) instead of G(x0, E(x0), E(x1))
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
# MLP Predictor (Conditioned on treat_cond)
# ============================================================================

class MLPPredictorConditioned(nn.Module):
    """MLP predictor to map [source latents, treat_cond] to target latents."""
    
    def __init__(self, latent_dim: int, treat_dim: int = 11, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.latent_dim = latent_dim
        self.treat_dim = treat_dim
        
        layers = []
        in_dim = latent_dim + treat_dim  # Conditioned input
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, treat_cond: torch.Tensor) -> torch.Tensor:
        # Concatenate latent with treat_cond
        combined = torch.cat([x, treat_cond], dim=-1)
        return self.network(combined)
    
    def predict(self, x: np.ndarray, treat_cond: np.ndarray) -> np.ndarray:
        """Sklearn-compatible predict method for numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            treat_tensor = torch.tensor(treat_cond, dtype=torch.float32, device=next(self.parameters()).device)
            output = self.forward(x_tensor, treat_tensor)
            return output.cpu().numpy()


class RidgePredictorConditioned:
    """Ridge predictor that takes [source latents, treat_cond] as input."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = None
    
    def fit(self, source_latents: np.ndarray, treat_conds: np.ndarray, target_latents: np.ndarray):
        """Fit the ridge model on [source_latents, treat_conds] -> target_latents."""
        # Concatenate source latents with treat_cond
        X = np.concatenate([source_latents, treat_conds], axis=-1)
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, target_latents)
        return self.model.score(X, target_latents)
    
    def predict(self, source_latent: np.ndarray, treat_cond: np.ndarray) -> np.ndarray:
        """Predict target latent from [source_latent, treat_cond]."""
        X = np.concatenate([source_latent, treat_cond], axis=-1)
        return self.model.predict(X)


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
    """Compute all metrics between two distributions."""
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
    
    w2 = 0.0
    
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
# Latent Caching and Predictor Training (Conditioned)
# ============================================================================

def get_latent_cache_path(experiment_dir: str, split: str) -> str:
    """Get the path for caching latents for a given split."""
    return os.path.join(experiment_dir, f"{split}_latents_cache_conditioned.pt")


def compute_and_cache_latents_conditioned(
    encoder: torch.nn.Module,
    samples: list,
    device: torch.device,
    cache_path: str,
    split_name: str = "dataset",
    is_conditioned: bool = True,
) -> tuple:
    """
    Compute E(x0) and E(x1) for all samples in the dataset WITH conditioning.
    
    For conditioned models:
    - Concatenates cell_cond with samples before encoding (43 -> 45 dim)
    - Also returns treat_cond for predictor training
    
    Args:
        encoder: Trained encoder
        samples: List of samples from dataset
        device: Device to run on
        cache_path: Path to save/load cached latents
        split_name: Name of split for logging
        is_conditioned: Whether the model is conditioned
        
    Returns:
        Tuple of (source_latents, target_latents, treat_conds) as numpy arrays
    """
    print(f"Computing {split_name} latents for {len(samples)} samples...")
    
    source_latents = []
    target_latents = []
    treat_conds = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            # Handle both conditioned (7 elements) and non-conditioned (6 elements) samples
            if len(sample) == 7:
                # Conditioned: culture, x0, x1, cell_cond_x0, cell_cond_x1, treat_cond, patient
                culture, x0, x1, cell_cond_x0, cell_cond_x1, treat_cond, patient = sample
            else:
                # Non-conditioned: culture, x0, x1, cell_cond, treat_cond, patient
                culture, x0, x1, cell_cond, treat_cond, patient = sample
                cell_cond_x0 = cell_cond
                # For x1, we need to create cell_cond with correct size
                cell_cond_x1 = np.tile(cell_cond[0:1], (x1.shape[0], 1))
            
            # Convert to tensors
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
            
            if is_conditioned:
                # Concatenate cell_cond with samples (43 -> 45 dim)
                cell_cond_x0_tensor = torch.tensor(cell_cond_x0, dtype=torch.float32, device=device)
                cell_cond_x1_tensor = torch.tensor(cell_cond_x1, dtype=torch.float32, device=device)
                
                x0_input = torch.cat([x0_tensor, cell_cond_x0_tensor], dim=-1)
                x1_input = torch.cat([x1_tensor, cell_cond_x1_tensor], dim=-1)
            else:
                x0_input = x0_tensor
                x1_input = x1_tensor
            
            # Add batch dimension and encode
            source_latent = encoder(x0_input.unsqueeze(0)).cpu().numpy()  # [1, latent_dim]
            target_latent = encoder(x1_input.unsqueeze(0)).cpu().numpy()  # [1, latent_dim]
            
            source_latents.append(source_latent)
            target_latents.append(target_latent)
            
            # Store treat_cond (take first row since all rows are the same)
            treat_conds.append(treat_cond[0:1])  # [1, 11]
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")
    
    # Stack into arrays
    source_latents = np.vstack(source_latents)  # [num_samples, latent_dim]
    target_latents = np.vstack(target_latents)  # [num_samples, latent_dim]
    treat_conds = np.vstack(treat_conds)  # [num_samples, 11]
    
    # Cache to disk
    print(f"Saving {split_name} latents to {cache_path}")
    torch.save({
        'source_latents': source_latents,
        'target_latents': target_latents,
        'treat_conds': treat_conds,
    }, cache_path)
    
    return source_latents, target_latents, treat_conds


def train_ridge_predictor_conditioned(
    source_latents: np.ndarray,
    treat_conds: np.ndarray,
    target_latents: np.ndarray,
    alpha: float = 1.0,
) -> RidgePredictorConditioned:
    """
    Train a ridge regression predictor to map [source latents, treat_cond] to target latents.
    """
    print(f"Training conditioned ridge regression predictor (alpha={alpha})...")
    print(f"  Training data shape: [{source_latents.shape}, {treat_conds.shape}] -> {target_latents.shape}")
    
    predictor = RidgePredictorConditioned(alpha=alpha)
    train_r2 = predictor.fit(source_latents, treat_conds, target_latents)
    
    print(f"  Training R^2: {train_r2:.4f}")
    
    return predictor


def train_mlp_predictor_conditioned(
    source_latents: np.ndarray,
    treat_conds: np.ndarray,
    target_latents: np.ndarray,
    device: torch.device,
    hidden_dim: int = 128,
    num_layers: int = 2,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    batch_size: int = 32,
) -> MLPPredictorConditioned:
    """
    Train an MLP predictor to map [source latents, treat_cond] to target latents.
    """
    latent_dim = source_latents.shape[1]
    treat_dim = treat_conds.shape[1]
    num_samples = source_latents.shape[0]
    
    print(f"Training conditioned MLP predictor...")
    print(f"  Training data shape: [{source_latents.shape}, {treat_conds.shape}] -> {target_latents.shape}")
    print(f"  Architecture: {latent_dim}+{treat_dim} -> {num_layers}x{hidden_dim} -> {latent_dim}")
    print(f"  lr={lr}, epochs={num_epochs}, batch_size={batch_size}")
    
    # Create model
    predictor = MLPPredictorConditioned(latent_dim, treat_dim, hidden_dim, num_layers).to(device)
    
    # Convert data to tensors
    source_tensor = torch.tensor(source_latents, dtype=torch.float32, device=device)
    treat_tensor = torch.tensor(treat_conds, dtype=torch.float32, device=device)
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
        treat_shuffled = treat_tensor[perm]
        target_shuffled = target_tensor[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_source = source_shuffled[i:i+batch_size]
            batch_treat = treat_shuffled[i:i+batch_size]
            batch_target = target_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = predictor(batch_source, batch_treat)
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
        pred = predictor(source_tensor, treat_tensor)
        final_loss = criterion(pred, target_tensor).item()
        
        ss_res = ((pred - target_tensor) ** 2).sum().item()
        ss_tot = ((target_tensor - target_tensor.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"  Final MSE: {final_loss:.6f}, R^2: {r2:.4f}")
    
    return predictor


# ============================================================================
# Main Evaluation Logic
# ============================================================================

def load_experiment(experiment_dir: str, device: torch.device):
    """Load the trained model, config, and instantiate the components."""
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Best model checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    
    # Instantiate dataset
    dataset = hydra.utils.instantiate(resolved_cfg.dataset)
    print(f"Instantiated dataset with {len(dataset.samples_train)} train samples and {len(dataset.samples_test)} test samples")
    
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
    
    # Check if model is conditioned (encoder in_dim > 43)
    encoder_in_dim = resolved_cfg.encoder.get('in_dim', 43)
    is_conditioned = encoder_in_dim > 43
    print(f"Model is {'CONDITIONED' if is_conditioned else 'NOT conditioned'} (encoder in_dim={encoder_in_dim})")
    
    return encoder, generator, dataset, cfg, is_conditioned


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor: Optional[Union[RidgePredictorConditioned, MLPPredictorConditioned]] = None,
    treat_cond: Optional[np.ndarray] = None,
    compute_baseline: bool = False,
    wasserstein_only: bool = False,
    max_samples_w1: Optional[int] = None,
):
    """
    Evaluate the model on a single sample using precomputed latents.
    
    For conditioned predictors, treat_cond is used to condition the prediction.
    """
    # Convert to tensors
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    # Compute baseline metrics if requested
    baseline_metrics = None
    if compute_baseline:
        baseline_metrics = compute_all_metrics(
            x0_tensor, x1_tensor, 
            wasserstein_only=wasserstein_only,
            max_samples_w1=max_samples_w1,
        )
    
    # Convert latents to tensors
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    # Get target latent: either from predictor or use precomputed
    if predictor is not None:
        # Use conditioned predictor: P([E(x0), treat_cond])
        predicted_target_latent = predictor.predict(source_latent, treat_cond)
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        # Use precomputed E(x1)
        target_latent_tensor = torch.tensor(target_latent, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Generate x1_pred from x0 using target latent
        x1_pred = generator.sample(
            x0_tensor,
            source_latent_tensor,
            target_latent_tensor,
        )
        x1_pred = x1_pred.squeeze(0)
    
    # Compute model metrics
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
    predictor_loss_weight: Optional[float] = None,
    seed: Optional[int] = None,
    selective_pairing_mode: Optional[str] = None,
    experiment_name: Optional[str] = None,
    num_epochs: Optional[int] = None,
) -> str:
    """Search through directories to find the experiment matching filter criteria."""
    if not os.path.exists(outputs_dir):
        raise ValueError(f"Outputs directory not found: {outputs_dir}")
    
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
    if num_epochs is not None:
        filters.append(f"num_epochs={num_epochs}")
    
    if not filters:
        raise ValueError("At least one filter criterion must be provided")
    
    print(f"Searching for experiment with: {', '.join(filters)}")
    print(f"Looking in: {outputs_dir}")
    
    matching_dirs = []
    
    for dirname in os.listdir(outputs_dir):
        dir_path = os.path.join(outputs_dir, dirname)
        
        if not os.path.isdir(dir_path):
            continue
        
        config_path = os.path.join(dir_path, "config.yaml")
        if not os.path.exists(config_path):
            continue
        
        try:
            cfg = OmegaConf.load(config_path)
            
            match = True
            
            if split_name is not None:
                cfg_split_name = cfg.get("experiment", {}).get("split_name")
                if cfg_split_name != split_name:
                    match = False
            
            if predictor_loss_weight is not None and match:
                cfg_weight = cfg.get("experiment", {}).get("predictor_loss_weight")
                if cfg_weight is None or abs(float(cfg_weight) - float(predictor_loss_weight)) >= 1e-6:
                    match = False
            
            if seed is not None and match:
                cfg_seed = cfg.get("seed")
                if cfg_seed is None or int(cfg_seed) != int(seed):
                    match = False
            
            if selective_pairing_mode is not None and match:
                cfg_pairing_mode = cfg.get("experiment", {}).get("selective_pairing_mode")
                if cfg_pairing_mode != selective_pairing_mode:
                    match = False
            
            if experiment_name is not None and match:
                cfg_exp_name = cfg.get("experiment", {}).get("name")
                if cfg_exp_name != experiment_name:
                    match = False
            
            if num_epochs is not None and match:
                cfg_num_epochs = cfg.get("training", {}).get("num_epochs")
                if cfg_num_epochs is None or int(cfg_num_epochs) != int(num_epochs):
                    match = False
            
            if match:
                matching_dirs.append(dir_path)
                print(f"  Found match: {dirname}")
                    
        except Exception as e:
            continue
    
    if len(matching_dirs) == 0:
        raise ValueError(f"No experiment found matching: {', '.join(filters)}")
    
    if len(matching_dirs) > 1:
        print(f"Warning: Multiple matching directories found, using the first one:")
        for d in matching_dirs:
            print(f"  - {d}")
    
    return matching_dirs[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Trellis model (with conditioning support)")
    parser.add_argument("--experiment_dir", type=str, default=None,
        help="Path to the experiment directory")
    parser.add_argument("--split_name", type=str, default=None,
        help="Split name to search for")
    parser.add_argument("--predictor_loss_weight", type=float, default=None,
        help="Predictor loss weight to search for")
    parser.add_argument("--seed", type=int, default=None,
        help="Random seed to search for")
    parser.add_argument("--selective_pairing_mode", type=str, default=None,
        help="Selective pairing mode to search for")
    parser.add_argument("--experiment_name", type=str, default=None,
        help="Experiment name to search for")
    parser.add_argument("--num_epochs", type=int, default=None,
        help="Number of training epochs to search for")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
        help="Directory containing experiment outputs")
    parser.add_argument("--compute_baseline", action="store_true",
        help="Compute and print baseline metrics (x0 vs x1)")
    parser.add_argument("--wasserstein_only", action="store_true",
        help="Only compute W1 and W2 scores")
    parser.add_argument("--max_samples_w1", type=int, default=3000,
        help="Maximum samples for W1 computation")
    parser.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on")
    parser.add_argument("--use_predictor", action="store_true",
        help="Use a predictor P to predict target latent")
    parser.add_argument("--predictor_type", type=str, choices=["ridge", "mlp"], default="ridge",
        help="Type of predictor to use")
    parser.add_argument("--ridge_alpha", type=float, default=1.0,
        help="Ridge regression regularization strength")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128,
        help="Hidden dimension for MLP predictor")
    parser.add_argument("--mlp_num_layers", type=int, default=2,
        help="Number of hidden layers for MLP predictor")
    parser.add_argument("--mlp_lr", type=float, default=1e-3,
        help="Learning rate for MLP predictor")
    parser.add_argument("--mlp_epochs", type=int, default=1000,
        help="Number of training epochs for MLP predictor")
    parser.add_argument("--mlp_batch_size", type=int, default=32,
        help="Batch size for MLP predictor training")
    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        has_filter = any([
            args.split_name is not None,
            args.predictor_loss_weight is not None,
            args.seed is not None,
            args.selective_pairing_mode is not None,
            args.experiment_name is not None,
            args.num_epochs is not None,
        ])
        if not has_filter:
            parser.error("Must provide either --experiment_dir or at least one filter criterion")
        
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            split_name=args.split_name, 
            predictor_loss_weight=args.predictor_loss_weight,
            seed=args.seed,
            selective_pairing_mode=args.selective_pairing_mode,
            experiment_name=args.experiment_name,
            num_epochs=args.num_epochs,
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load experiment (includes is_conditioned flag)
    encoder, generator, dataset, cfg, is_conditioned = load_experiment(args.experiment_dir, device)
    
    # Compute and cache test latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING LATENTS")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents, test_target_latents, test_treat_conds = compute_and_cache_latents_conditioned(
        encoder, dataset.samples_test, device, test_cache_path, 
        split_name="test", is_conditioned=is_conditioned
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print(f"TRAINING CONDITIONED PREDICTOR ({args.predictor_type.upper()})")
        print("=" * 80)
        
        # Compute and cache training latents
        train_cache_path = get_latent_cache_path(args.experiment_dir, "train")
        train_source_latents, train_target_latents, train_treat_conds = compute_and_cache_latents_conditioned(
            encoder, dataset.samples_train, device, train_cache_path,
            split_name="train", is_conditioned=is_conditioned
        )
        
        # Train conditioned predictor
        if args.predictor_type == "ridge":
            predictor = train_ridge_predictor_conditioned(
                train_source_latents, 
                train_treat_conds,
                train_target_latents, 
                alpha=args.ridge_alpha
            )
        elif args.predictor_type == "mlp":
            predictor = train_mlp_predictor_conditioned(
                train_source_latents,
                train_treat_conds,
                train_target_latents,
                device=device,
                hidden_dim=args.mlp_hidden_dim,
                num_layers=args.mlp_num_layers,
                lr=args.mlp_lr,
                num_epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
            )
        
        print(f"\nUsing conditioned {args.predictor_type} predictor: G(x0, E(x0), P([E(x0), treat_cond]))")
    else:
        print(f"\nUsing oracle target: G(x0, E(x0), E(x1))")
    
    # Evaluate each sample
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if args.use_predictor:
        print(f"Mode: Using conditioned {args.predictor_type.upper()} predictor P([E(x0), treat_cond]) as target latent")
    else:
        print("Mode: Using oracle E(x1) as target latent")
    if args.wasserstein_only:
        print("Computing: W1, W2 only (skipping MMD and r2)")
    if args.compute_baseline:
        print("Computing baseline: x0 vs x1")
    
    max_samples_w1 = args.max_samples_w1 if args.max_samples_w1 > 0 else None
    if max_samples_w1 is not None:
        print(f"W1 subsampling: max {max_samples_w1} samples from source/target")
    print("=" * 80)
    
    # Determine metrics to track
    if args.wasserstein_only:
        metric_names = ['W1', 'W2']
    else:
        metric_names = ['W1', 'W2', 'MMD', 'r2']
    
    # Track metrics
    all_model_metrics = {name: [] for name in metric_names}
    all_baseline_metrics = {name: [] for name in metric_names} if args.compute_baseline else None
    
    for i, sample in enumerate(dataset.samples_test):
        # Handle both conditioned and non-conditioned samples
        if len(sample) == 7:
            culture, x0, x1, cell_cond_x0, cell_cond_x1, treat_cond, patient = sample
        else:
            culture, x0, x1, cell_cond, treat_cond, patient = sample
        
        # Get precomputed latents for this sample
        source_latent = test_source_latents[i:i+1]
        target_latent = test_target_latents[i:i+1]
        sample_treat_cond = test_treat_conds[i:i+1]
        
        print(f"\nSample {i + 1}/{len(dataset.samples_test)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, device, 
            predictor=predictor,
            treat_cond=sample_treat_cond,
            compute_baseline=args.compute_baseline,
            wasserstein_only=args.wasserstein_only,
            max_samples_w1=max_samples_w1,
        )
        
        model = results['model']
        
        # Print results
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
