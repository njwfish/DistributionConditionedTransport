"""
Predictor training utilities that match the loss function used during training.
Supports both cosine similarity loss and MSE loss with ridge regularization.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Literal
from sklearn.linear_model import Ridge


def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    two_step: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset of latent transitions for training the predictor.
    
    Feeds all data at each timepoint into the encoder at once (no subsetting).
    
    Returns (X, Y) of source/target latents built from training timepoints only.

    Modes:
    - two_step=False (default): input = latent at t; output = latent at t+1
    - two_step=True: input = concat(latent at t, latent at t+1); output = latent at t+2

    The transition into any held-out final timepoint is excluded.
    """
    encoder.eval()
    
    Xs = data['Xs']
    with torch.no_grad():
        # Collect one latent per timepoint by encoding all samples at once
        latents_by_time: List[np.ndarray] = []
        for X_np in list(Xs[:-1]):
            # Feed all samples at this timepoint into encoder at once
            X_tensor = torch.tensor(X_np, dtype=torch.float32, device=device).unsqueeze(0)
            lat = encoder(X_tensor).cpu().numpy()
            lat = np.squeeze(lat, axis=0).astype(np.float64)
            latents_by_time.append(lat)
            
    # Build source/target pairs from TRAINING timepoints only (exclude transition into held-out final)
    X_pairs, y_pairs = [], []
    
    if not two_step:
        for t in range(len(latents_by_time) - 1):
            cur, nxt = latents_by_time[t], latents_by_time[t + 1]
            X_pairs.append(cur)
            y_pairs.append(nxt)
    else:
        # Need triples of consecutive timepoints (t, t+1, t+2)
        for t in range(len(latents_by_time) - 2):
            cur, nxt, nxt2 = latents_by_time[t], latents_by_time[t + 1], latents_by_time[t + 2]
            # Concatenate latents at t and t+1 as input, latent at t+2 as output
            ab = np.concatenate([cur, nxt], axis=-1)
            X_pairs.append(ab)
            y_pairs.append(nxt2)

    return np.vstack(X_pairs), np.vstack(y_pairs)


class LinearPredictorCosine(nn.Module):
    """
    A linear predictor trained with cosine similarity loss and L2 regularization.
    This matches the training loss used for the predictor during model training.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.similarity = nn.CosineSimilarity(dim=-1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scikit-learn style predict interface."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            return self.forward(x_tensor).cpu().numpy()
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, ridge_alpha: float = 0.0) -> torch.Tensor:
        """Compute cosine loss with optional L2 regularization."""
        cosine_loss = (1 - self.similarity(pred, target)).mean()
        
        if ridge_alpha > 0:
            # L2 regularization on weights (same as training code: ridge_alpha * sum(W^2))
            l2_reg = ridge_alpha * torch.sum(self.linear.weight ** 2)
            return cosine_loss + l2_reg
        return cosine_loss


class LinearPredictorMSE(nn.Module):
    """
    A linear predictor trained with MSE loss and L2 regularization.
    """
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scikit-learn style predict interface."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            return self.forward(x_tensor).cpu().numpy()
    
    def compute_loss(self, pred: torch.Tensor, target: torch.Tensor, ridge_alpha: float = 0.0) -> torch.Tensor:
        """Compute MSE loss with optional L2 regularization."""
        mse_loss = (pred - target).pow(2).mean()
        
        if ridge_alpha > 0:
            # L2 regularization on weights
            l2_reg = ridge_alpha * torch.sum(self.linear.weight ** 2)
            return mse_loss + l2_reg
        return mse_loss


def train_linear_predictor_cosine(
    X: np.ndarray,
    Y: np.ndarray,
    ridge_alpha: float = 1e-3,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> LinearPredictorCosine:
    """
    Train a linear predictor using cosine similarity loss with L2 regularization.
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        ridge_alpha: L2 regularization weight
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Whether to print progress
        
    Returns:
        Trained LinearPredictorCosine model
    """
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    
    model = LinearPredictorCosine(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=device)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = model.compute_loss(pred, Y_tensor, ridge_alpha)
        loss.backward()
        optimizer.step()
        
        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
    
    model.eval()
    return model


def train_linear_predictor_mse(
    X: np.ndarray,
    Y: np.ndarray,
    ridge_alpha: float = 1e-3,
    seed: int = 0,
) -> Ridge:
    """
    Train a linear predictor using MSE loss with L2 regularization.
    Uses sklearn's Ridge for closed-form solution.
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        ridge_alpha: L2 regularization weight
        seed: Random seed
        
    Returns:
        Trained sklearn Ridge model
    """
    model = Ridge(alpha=ridge_alpha, random_state=seed).fit(X, Y)
    return model


def get_matched_predictor(
    encoder: torch.nn.Module,
    data: dict,
    loss_type: Literal["cosine", "MSE"] = "cosine",
    ridge_alpha: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    two_step: bool = False,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    verbose: bool = False,
):
    """
    Train a linear predictor that matches the loss function used during training.
    
    Feeds all data at each timepoint into the encoder at once (no subsetting).
    
    Args:
        encoder: Trained encoder model
        data: Dataset dictionary with 'Xs' key
        loss_type: "cosine" or "MSE" - should match predictor.loss_type from training config
        ridge_alpha: L2 regularization weight - should match predictor.model_args.ridge_alpha from training config
        device: Device to train on
        seed: Random seed (only used for MSE loss)
        two_step: Whether to use two-step prediction (input = concat of t and t+1 latents)
        num_epochs: Number of training epochs (only used for cosine loss)
        lr: Learning rate (only used for cosine loss)
        verbose: Whether to print progress
        
    Returns:
        Trained predictor model with scikit-learn style .predict() interface
    """
    # Build the latent transition dataset (feeds all data at once, no subsetting)
    X, Y = build_latent_transition_dataset(
        encoder, data, device=device, two_step=two_step
    )
    
    if loss_type == "cosine":
        model = train_linear_predictor_cosine(
            X, Y, ridge_alpha=ridge_alpha, num_epochs=num_epochs, 
            lr=lr, device=device, verbose=verbose
        )
    elif loss_type == "MSE":
        model = train_linear_predictor_mse(X, Y, ridge_alpha=ridge_alpha, seed=seed)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Expected 'cosine' or 'MSE'.")
    
    return model


def get_predictor_config_from_checkpoint(cfg) -> Tuple[str, float]:
    """
    Extract predictor loss type and ridge alpha from a loaded config.
    
    Args:
        cfg: OmegaConf config loaded from checkpoint directory
        
    Returns:
        Tuple of (loss_type, ridge_alpha)
    """
    from omegaconf import OmegaConf
    
    # Get loss_type from predictor config (default: cosine)
    loss_type = "cosine"
    try:
        loss_type = OmegaConf.select(cfg, "predictor.loss_type", default="cosine")
    except Exception:
        pass
    
    # Get ridge_alpha from experiment.predictor_model_args (default: 1e-3)
    ridge_alpha = 1e-3
    try:
        ridge_alpha = OmegaConf.select(cfg, "experiment.predictor_model_args.ridge_alpha", default=1e-3)
    except Exception:
        pass
    
    return loss_type, ridge_alpha
