"""
Unified evaluation script for all Trellis model variants.

Supports all training configurations:
- run_trellis.sh (standard trellis)
- run_trellis_a2a.sh (a2a variant)
- run_trellis_conditioned.sh (conditioned encoder)
- run_trellis_conditioned_a2a.sh (conditioned encoder + a2a)
- run_trellis_mfm.sh (source-only MFM variant - no target latent)

Key features:
1. Automatic detection of model type (a2a vs standard, conditioned vs non-conditioned, source-only)
2. Support for different predictor types: ridge, mlp, conditioned_ridge, conditioned_mlp
   (Note: predictors are NOT used for source-only models since they don't use target latent)
3. Proper normalization of predicted latents before passing to generator
4. W1 subsampling for faster computation
5. Flexible experiment filtering via config keywords

Usage examples:
    # Standard trellis with ridge predictor
    python evaluate_trellis_unified.py --experiment_name trellis --split_name replicas-1 --use_predictor --predictor_type ridge
    
    # A2A with conditioned MLP predictor
    python evaluate_trellis_unified.py --experiment_name trellis_a2a --split_name pdo21 --use_predictor --predictor_type conditioned_mlp
    
    # Conditioned model with conditioned predictor
    python evaluate_trellis_unified.py --experiment_name trellis_conditioned --use_predictor --predictor_type conditioned_mlp
    
    # Source-only MFM model (no predictor needed/used)
    python evaluate_trellis_unified.py --experiment_name trellis_mfm_gnn --split_name replicas-1
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
# Latent Normalization (from utils/latents.py)
# ============================================================================

def normalize_latent(latent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a batch of latent vectors along the last dimension.
    """
    if latent is None:
        raise ValueError("normalize_latent expected a Tensor, got None")
    denom = torch.norm(latent, dim=-1, keepdim=True).clamp_min(eps)
    return latent / denom


def normalize_latent_np(latent: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize a batch of latent vectors (numpy version).
    """
    denom = np.linalg.norm(latent, axis=-1, keepdims=True)
    denom = np.clip(denom, a_min=eps, a_max=None)
    return latent / denom


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
        self.latent_dim = latent_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Sklearn-compatible predict method for numpy arrays."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            output = self.forward(x_tensor)
            return output.cpu().numpy()


class ConditionedMLPPredictor(nn.Module):
    """
    Conditioned MLP predictor that takes source latent and treat_cond as input.
    Maps (source_latent || treat_cond) -> target_latent
    """
    
    def __init__(self, latent_dim: int, treat_dim: int = 11, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        
        input_dim = latent_dim + treat_dim
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.network = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.treat_dim = treat_dim
    
    def forward(self, source_latent: torch.Tensor, treat_cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([source_latent, treat_cond], dim=-1)
        return self.network(x)
    
    def predict(self, source_latent: np.ndarray, treat_cond: np.ndarray) -> np.ndarray:
        """Predict with conditioning - compatible interface for evaluation."""
        self.eval()
        with torch.no_grad():
            source_tensor = torch.tensor(source_latent, dtype=torch.float32, device=next(self.parameters()).device)
            treat_tensor = torch.tensor(treat_cond, dtype=torch.float32, device=next(self.parameters()).device)
            output = self.forward(source_tensor, treat_tensor)
            return output.cpu().numpy()


class ConditionedRidgePredictor:
    """
    Conditioned Ridge regressor that concatenates treat_cond with source latent.
    Maps (source_latent || treat_cond) -> target_latent
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=alpha)
        self.fitted = False
    
    def fit(self, source_latents: np.ndarray, target_latents: np.ndarray, treat_conds: np.ndarray):
        """
        Fit the ridge regressor.
        
        Args:
            source_latents: [num_samples, latent_dim]
            target_latents: [num_samples, latent_dim]
            treat_conds: [num_samples, treat_dim]
        """
        # Concatenate source latent with treat_cond
        X = np.concatenate([source_latents, treat_conds], axis=-1)
        self.model.fit(X, target_latents)
        self.fitted = True
        
        # Compute training R^2
        train_r2 = self.model.score(X, target_latents)
        return train_r2
    
    def predict(self, source_latent: np.ndarray, treat_cond: np.ndarray) -> np.ndarray:
        """Predict with conditioning."""
        if not self.fitted:
            raise RuntimeError("Model not fitted yet")
        X = np.concatenate([source_latent, treat_cond], axis=-1)
        return self.model.predict(X)


# ============================================================================
# Metric Functions
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
    """
    Compute all metrics between two distributions.
    """
    # Subsample for W1 computation if needed
    if max_samples_w1 is not None:
        pred_w1 = pred
        target_w1 = target
        
        # TODO: is this subsampling necessary?
        if pred.shape[0] > max_samples_w1:
            indices = torch.randperm(pred.shape[0])[:max_samples_w1]
            pred_w1 = pred[indices]
        
        if target.shape[0] > max_samples_w1:
            indices = torch.randperm(target.shape[0])[:max_samples_w1]
            target_w1 = target[indices]
        
        w1 = wasserstein(pred_w1, target_w1, power=1)
    else:
        w1 = wasserstein(pred, target, power=1)
    
    w2 = 0.0  # Skip W2 for speed
    
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
# Dataset Utilities
# ============================================================================

def unpack_sample(sample):
    """
    Unpack a sample tuple, handling both old (6-element) and new (7-element) formats.
    
    Old format (a2a non-conditioned): (culture, x0, x1, cell_cond, treat_cond, patient)
    New format (standard and a2a_conditioned): (culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient)
    
    Returns:
        (culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient)
    """
    if len(sample) == 6:
        # Old format: single cell_cond for both source and target
        culture, x0, x1, cell_cond, treat_cond, patient = sample
        return culture, x0, x1, cell_cond, cell_cond, treat_cond, patient
    elif len(sample) == 7:
        # New format: separate cell_cond_source and cell_cond_target
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
        return culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient
    else:
        raise ValueError(f"Unexpected sample format with {len(sample)} elements")


def get_test_samples_from_dataset(dataset, is_a2a: bool):
    """
    Get test samples from dataset, handling both standard and a2a dataset structures.
    
    Args:
        dataset: The dataset object
        is_a2a: Whether this is an a2a-style dataset
        
    Returns:
        List of samples in format: (culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient)
    """
    if is_a2a:
        # A2A datasets have samples_test attribute
        return dataset.samples_test
    else:
        # Standard datasets have samples attribute
        return dataset.samples


def get_train_samples_from_dataset(dataset, is_a2a: bool):
    """
    Get training samples from dataset, handling both standard and a2a dataset structures.
    """
    if is_a2a:
        return dataset.samples_train
    else:
        return dataset.samples


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
    is_conditioned_encoder: bool = False,
) -> tuple:
    """
    Compute E(x0) and E(x1) for all samples.
    
    Args:
        encoder: Trained encoder
        samples: List of samples
        device: Device to run on
        cache_path: Path to save/load cached latents
        split_name: Name of split for logging
        is_conditioned_encoder: If True, concatenate cell_cond with samples before encoding
        
    Returns:
        Tuple of (source_latents, target_latents, treat_conds) as numpy arrays
    """
    print(f"Computing {split_name} latents for {len(samples)} samples...")
    if is_conditioned_encoder:
        print(f"  Using conditioned encoding (concatenating cell_cond with samples)")
    
    source_latents = []
    target_latents = []
    treat_conds = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = unpack_sample(sample)
            
            # Convert to tensors and add batch dimension
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)
            
            if is_conditioned_encoder:
                # Concatenate cell_cond with samples for conditioned encoder
                cell_cond_source_tensor = torch.tensor(cell_cond_source, dtype=torch.float32, device=device).unsqueeze(0)
                cell_cond_target_tensor = torch.tensor(cell_cond_target, dtype=torch.float32, device=device).unsqueeze(0)
                
                # x0_tensor: (1, N, 43) + cell_cond_source: (1, N, 2) -> (1, N, 45)
                x0_input = torch.cat([x0_tensor, cell_cond_source_tensor], dim=-1)
                x1_input = torch.cat([x1_tensor, cell_cond_target_tensor], dim=-1)
            else:
                x0_input = x0_tensor
                x1_input = x1_tensor
            
            # Encode
            source_latent = encoder(x0_input).cpu().numpy()  # [1, latent_dim]
            target_latent = encoder(x1_input).cpu().numpy()  # [1, latent_dim]
            
            source_latents.append(source_latent)
            target_latents.append(target_latent)
            
            # Store treat_cond (take first row since all rows are the same for a sample)
            treat_conds.append(treat_cond[0:1])  # [1, treat_dim]
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")
    
    # Stack into arrays: [num_samples, latent_dim]
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    treat_conds = np.vstack(treat_conds)
    
    # Cache to disk
    print(f"Saving {split_name} latents to {cache_path}")
    torch.save({
        'source_latents': source_latents,
        'target_latents': target_latents,
        'treat_conds': treat_conds,
    }, cache_path)
    
    return source_latents, target_latents, treat_conds


def train_ridge_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    alpha: float = 1.0,
) -> Ridge:
    """Train a ridge regression predictor to map source latents to target latents."""
    print(f"Training ridge regression predictor (alpha={alpha})...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    
    predictor = Ridge(alpha=alpha)
    predictor.fit(source_latents, target_latents)
    
    train_r2 = predictor.score(source_latents, target_latents)
    print(f"  Training R^2: {train_r2:.4f}")
    
    return predictor


def train_conditioned_ridge_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    treat_conds: np.ndarray,
    alpha: float = 1.0,
) -> ConditionedRidgePredictor:
    """Train a conditioned ridge regression predictor."""
    print(f"Training CONDITIONED ridge regression predictor (alpha={alpha})...")
    print(f"  Training data shape: {source_latents.shape} + {treat_conds.shape} -> {target_latents.shape}")
    
    predictor = ConditionedRidgePredictor(alpha=alpha)
    train_r2 = predictor.fit(source_latents, target_latents, treat_conds)
    
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
    """Train an MLP predictor to map source latents to target latents."""
    latent_dim = source_latents.shape[1]
    num_samples = source_latents.shape[0]
    
    print(f"Training MLP predictor...")
    print(f"  Training data shape: {source_latents.shape} -> {target_latents.shape}")
    print(f"  Architecture: {latent_dim} -> {num_layers}x{hidden_dim} -> {latent_dim}")
    print(f"  lr={lr}, epochs={num_epochs}, batch_size={batch_size}")
    
    predictor = MLPPredictor(latent_dim, hidden_dim, num_layers).to(device)
    
    source_tensor = torch.tensor(source_latents, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_latents, dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    predictor.train()
    
    for epoch in range(num_epochs):
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
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
    predictor.eval()
    with torch.no_grad():
        pred = predictor(source_tensor)
        final_loss = criterion(pred, target_tensor).item()
        ss_res = ((pred - target_tensor) ** 2).sum().item()
        ss_tot = ((target_tensor - target_tensor.mean(dim=0)) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"  Final MSE: {final_loss:.6f}, R^2: {r2:.4f}")
    
    return predictor


def train_conditioned_mlp_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    treat_conds: np.ndarray,
    device: torch.device,
    hidden_dim: int = 128,
    num_layers: int = 2,
    lr: float = 1e-3,
    num_epochs: int = 1000,
    batch_size: int = 32,
) -> ConditionedMLPPredictor:
    """Train a conditioned MLP predictor that maps (source_latent, treat_cond) -> target_latent."""
    latent_dim = source_latents.shape[1]
    treat_dim = treat_conds.shape[1]
    num_samples = source_latents.shape[0]
    
    print(f"Training CONDITIONED MLP predictor...")
    print(f"  Training data shape: {source_latents.shape} + {treat_conds.shape} -> {target_latents.shape}")
    print(f"  Architecture: ({latent_dim} + {treat_dim}) -> {num_layers}x{hidden_dim} -> {latent_dim}")
    print(f"  lr={lr}, epochs={num_epochs}, batch_size={batch_size}")
    
    predictor = ConditionedMLPPredictor(latent_dim, treat_dim, hidden_dim, num_layers).to(device)
    
    source_tensor = torch.tensor(source_latents, dtype=torch.float32, device=device)
    target_tensor = torch.tensor(target_latents, dtype=torch.float32, device=device)
    treat_tensor = torch.tensor(treat_conds, dtype=torch.float32, device=device)
    
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    predictor.train()
    
    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        source_shuffled = source_tensor[perm]
        target_shuffled = target_tensor[perm]
        treat_shuffled = treat_tensor[perm]
        
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_samples, batch_size):
            batch_source = source_shuffled[i:i+batch_size]
            batch_target = target_shuffled[i:i+batch_size]
            batch_treat = treat_shuffled[i:i+batch_size]
            
            optimizer.zero_grad()
            pred = predictor(batch_source, batch_treat)
            loss = criterion(pred, batch_target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
    
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

def load_experiment(experiment_dir: str, device: torch.device, load_train_data: bool = False):
    """
    Load the trained model, config, and instantiate the components.
    
    Returns:
        encoder, generator, test_samples, config, train_samples (or None), is_conditioned_encoder, is_a2a, is_source_only
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
    
    # Resolve config references
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    
    # Detect model type
    encoder_in_dim = resolved_cfg.get("encoder", {}).get("in_dim", 43)
    is_conditioned_encoder = (encoder_in_dim == 45)
    
    # Detect a2a dataset
    dataset_target = resolved_cfg.get("dataset", {}).get("_target_", "")
    is_a2a = "a2a" in dataset_target.lower()
    
    # Detect source-only model (MFM variant that doesn't use target latent)
    # Check model config for source_only flag, or check loss type
    model_source_only = resolved_cfg.get("model", {}).get("source_only", False)
    loss_target = resolved_cfg.get("loss", {}).get("_target_", "")
    is_source_only = model_source_only or "source_only" in loss_target.lower()
    
    if is_conditioned_encoder:
        print(f"Detected CONDITIONED encoder (encoder.in_dim={encoder_in_dim})")
    else:
        print(f"Detected NON-CONDITIONED encoder (encoder.in_dim={encoder_in_dim})")
    
    if is_a2a:
        print(f"Detected A2A dataset: {dataset_target}")
    else:
        print(f"Detected standard dataset: {dataset_target}")
    
    if is_source_only:
        print(f"Detected SOURCE-ONLY model (no target latent used)")
        print(f"  model.source_only={model_source_only}, loss={loss_target}")
    
    # Instantiate datasets
    train_samples = None
    test_samples = None
    
    if is_a2a:
        # A2A datasets don't have split_mode, they have internal train/test
        dataset = hydra.utils.instantiate(resolved_cfg.dataset)
        test_samples = dataset.samples_test
        train_samples = dataset.samples_train if load_train_data else None
        print(f"Instantiated A2A dataset: {len(dataset.samples_test)} test samples, {len(dataset.samples_train)} train samples")
    else:
        # Standard dataset: instantiate test split
        test_cfg = OmegaConf.create(OmegaConf.to_container(resolved_cfg, resolve=True))
        test_cfg.dataset.split_mode = "test"
        test_dataset = hydra.utils.instantiate(test_cfg.dataset)
        test_samples = test_dataset.samples
        print(f"Instantiated test dataset with split_mode='test', {len(test_samples)} samples")
        
        if load_train_data:
            train_cfg = OmegaConf.create(OmegaConf.to_container(resolved_cfg, resolve=True))
            train_cfg.dataset.split_mode = "train"
            train_dataset = hydra.utils.instantiate(train_cfg.dataset)
            train_samples = train_dataset.samples
            print(f"Instantiated train dataset with split_mode='train', {len(train_samples)} samples")
    
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
    
    return encoder, generator, test_samples, cfg, train_samples, is_conditioned_encoder, is_a2a, is_source_only


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor = None,
    treat_cond: Optional[np.ndarray] = None,
    compute_baseline: bool = False,
    wasserstein_only: bool = False,
    max_samples_w1: Optional[int] = None,
    normalize_predicted_latent: bool = True,
    is_source_only: bool = False,
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
        predictor: Optional predictor. If provided, uses predicted latent instead of E(x1)
        treat_cond: Treatment conditioning [1, treat_dim], required for conditioned predictors
        compute_baseline: If True, compute baseline metrics (x0 vs x1)
        wasserstein_only: If True, only compute W1 and W2
        max_samples_w1: If provided, subsample for W1 computation
        normalize_predicted_latent: If True, normalize predicted latent before passing to generator
        is_source_only: If True, model doesn't use target latent (pass None to generator)
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
    
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    # For source-only models, target_latent is not used
    if is_source_only:
        target_latent_tensor = None
    elif predictor is not None:
        # Get target latent from predictor
        # Determine predictor type and get prediction
        if isinstance(predictor, (ConditionedMLPPredictor, ConditionedRidgePredictor)):
            if treat_cond is None:
                raise ValueError("treat_cond is required for conditioned predictors")
            predicted_target_latent = predictor.predict(source_latent, treat_cond)
        elif isinstance(predictor, MLPPredictor):
            predicted_target_latent = predictor.predict(source_latent)
        elif hasattr(predictor, 'predict'):
            # Ridge regressor (sklearn)
            predicted_target_latent = predictor.predict(source_latent)
        else:
            raise ValueError(f"Unknown predictor type: {type(predictor)}")
        
        # Normalize predicted latent
        if normalize_predicted_latent:
            predicted_target_latent = normalize_latent_np(predicted_target_latent)
        
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        # Use precomputed (oracle) target latent
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
    """
    Search through directories in outputs_dir to find the experiment matching
    the given filter criteria.
    """
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
        # Handle "null" or "none" string to mean actual None value
        display_value = selective_pairing_mode if selective_pairing_mode.lower() not in ("null", "none") else "None"
        filters.append(f"selective_pairing_mode={display_value}")
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
                # Handle "null" or "none" string to mean actual None value
                if selective_pairing_mode.lower() in ("null", "none"):
                    if cfg_pairing_mode is not None:
                        match = False
                elif cfg_pairing_mode != selective_pairing_mode:
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
    parser = argparse.ArgumentParser(description="Unified evaluation for all Trellis model variants")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory. If not provided, searches based on filter criteria."
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default=None,
        help="Split name to search for (e.g., 'replicas-1', 'pdo21')"
    )
    parser.add_argument(
        "--predictor_loss_weight",
        type=float,
        default=None,
        help="Predictor loss weight to search for"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to search for"
    )
    parser.add_argument(
        "--selective_pairing_mode",
        type=str,
        default=None,
        help="Selective pairing mode to search for. Use 'null' or 'none' to find experiments where selective_pairing_mode was None/null"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name to search for (e.g., 'trellis', 'trellis_a2a', 'trellis_conditioned', 'trellis_a2a_conditioned')"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of training epochs to search for"
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
        "--wasserstein_only",
        action="store_true",
        help="Only compute W1 and W2 (skip R2 and MMD)"
    )
    parser.add_argument(
        "--max_samples_w1",
        type=int,
        default=30000,
        help="Max samples for W1 computation. Set to 0 to disable subsampling."
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
        help="Use a predictor P to predict target latent from source latent"
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["ridge", "mlp", "conditioned_ridge", "conditioned_mlp"],
        default="ridge",
        help="Type of predictor: 'ridge', 'mlp', 'conditioned_ridge', 'conditioned_mlp'"
    )
    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
        help="Ridge regression regularization strength"
    )
    parser.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for MLP predictor"
    )
    parser.add_argument(
        "--mlp_num_layers",
        type=int,
        default=2,
        help="Number of hidden layers for MLP predictor"
    )
    parser.add_argument(
        "--mlp_lr",
        type=float,
        default=1e-3,
        help="Learning rate for MLP predictor"
    )
    parser.add_argument(
        "--mlp_epochs",
        type=int,
        default=1000,
        help="Number of training epochs for MLP predictor"
    )
    parser.add_argument(
        "--mlp_batch_size",
        type=int,
        default=32,
        help="Batch size for MLP predictor training"
    )
    parser.add_argument(
        "--no_normalize_predicted_latent",
        action="store_true",
        help="Disable normalization of predicted latent before passing to generator"
    )
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
    
    # First load experiment without training data to check if source-only
    encoder, generator, test_samples, cfg, _, is_conditioned_encoder, is_a2a, is_source_only = load_experiment(
        args.experiment_dir, device, load_train_data=False
    )
    
    # For source-only models, predictors are not applicable
    if is_source_only and args.use_predictor:
        print("\n" + "=" * 80)
        print("WARNING: --use_predictor was specified but this is a SOURCE-ONLY model.")
        print("Source-only models do not use target latent, so predictors are not applicable.")
        print("Ignoring --use_predictor flag.")
        print("=" * 80)
        args.use_predictor = False
    
    # Now load training data if needed for predictor
    train_samples = None
    if args.use_predictor:
        _, _, _, _, train_samples, _, _, _ = load_experiment(
            args.experiment_dir, device, load_train_data=True
        )
    
    # Determine if predictor type is conditioned
    is_conditioned_predictor = args.predictor_type in ["conditioned_ridge", "conditioned_mlp"]
    
    # Compute and cache latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING LATENTS")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents, test_target_latents, test_treat_conds = compute_and_cache_latents(
        encoder, test_samples, device, test_cache_path, split_name="test",
        is_conditioned_encoder=is_conditioned_encoder
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print(f"TRAINING PREDICTOR ({args.predictor_type.upper()})")
        print("=" * 80)
        
        train_cache_path = get_latent_cache_path(args.experiment_dir, "train")
        train_source_latents, train_target_latents, train_treat_conds = compute_and_cache_latents(
            encoder, train_samples, device, train_cache_path, split_name="train",
            is_conditioned_encoder=is_conditioned_encoder
        )
        
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
        elif args.predictor_type == "conditioned_ridge":
            predictor = train_conditioned_ridge_predictor(
                train_source_latents,
                train_target_latents,
                train_treat_conds,
                alpha=args.ridge_alpha,
            )
        elif args.predictor_type == "conditioned_mlp":
            predictor = train_conditioned_mlp_predictor(
                train_source_latents,
                train_target_latents,
                train_treat_conds,
                device=device,
                hidden_dim=args.mlp_hidden_dim,
                num_layers=args.mlp_num_layers,
                lr=args.mlp_lr,
                num_epochs=args.mlp_epochs,
                batch_size=args.mlp_batch_size,
            )
        
        if is_conditioned_predictor:
            print(f"\nUsing {args.predictor_type} predictor: G(x0, E(x0), P(E(x0), treat_cond))")
        else:
            print(f"\nUsing {args.predictor_type} predictor: G(x0, E(x0), P(E(x0)))")
    elif is_source_only:
        print(f"\nUsing source-only mode: G(x0, E(x0)) - no target latent")
    else:
        print(f"\nUsing oracle target: G(x0, E(x0), E(x1))")
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if is_source_only:
        print("Mode: SOURCE-ONLY (no target latent used)")
    elif args.use_predictor:
        print(f"Mode: Using {args.predictor_type.upper()} predictor as target latent")
        if not args.no_normalize_predicted_latent:
            print("  Predicted latents will be normalized before generation")
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
    
    metric_names = ['W1', 'W2'] if args.wasserstein_only else ['W1', 'W2', 'MMD', 'r2']
    
    all_model_metrics = {name: [] for name in metric_names}
    all_baseline_metrics = {name: [] for name in metric_names} if args.compute_baseline else None
    
    for i, sample in enumerate(test_samples):
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = unpack_sample(sample)
        
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
            treat_cond=sample_treat_cond if is_conditioned_predictor else None,
            compute_baseline=args.compute_baseline,
            wasserstein_only=args.wasserstein_only,
            max_samples_w1=max_samples_w1,
            normalize_predicted_latent=not args.no_normalize_predicted_latent,
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
