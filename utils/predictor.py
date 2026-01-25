import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Literal, Dict, Any
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold


def _normalize_numpy(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize numpy array along last dimension to unit norm."""
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return arr / norms


def build_latent_transition_dataset(
    encoder: torch.nn.Module,
    data: dict,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    two_step: bool = False,
    residual_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dataset of latent transitions for training the predictor.
    
    Feeds all data at each timepoint into the encoder at once (no subsetting).
    
    Returns (X, Y) of source/target latents built from training timepoints only.
    When residual_mode=True, Y contains (target - source) residuals instead of target latents.

    Modes:
    - two_step=False (default): input = latent at t; output = latent at t+1
    - two_step=True: input = concat(latent at t, latent at t+1); output = latent at t+2
    
    Residual mode:
    - residual_mode=False (default): Y = target_latent
    - residual_mode=True: Y = (target_latent - source_latent)
      For two_step, source_latent is the latent at t+1 (the most recent input)

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
            if residual_mode:
                y_pairs.append(nxt - cur)  # Learn the residual
            else:
                y_pairs.append(nxt)
    else:
        # Need triples of consecutive timepoints (t, t+1, t+2)
        for t in range(len(latents_by_time) - 2):
            cur, nxt, nxt2 = latents_by_time[t], latents_by_time[t + 1], latents_by_time[t + 2]
            # Concatenate latents at t and t+1 as input, latent at t+2 as output
            ab = np.concatenate([cur, nxt], axis=-1)
            X_pairs.append(ab)
            if residual_mode:
                y_pairs.append(nxt2 - nxt)  # Learn the residual from t+1 to t+2
            else:
                y_pairs.append(nxt2)

    return np.vstack(X_pairs), np.vstack(y_pairs)


class ResidualPredictorWrapper:
    """
    Wrapper that adds residual prediction functionality to any predictor.
    When predicting, adds the raw prediction to the source latent and normalizes.
    
    For two_step mode, the source latent is extracted from the concatenated input
    (the second half, which is the latent at t+1).
    """
    
    def __init__(self, base_predictor, two_step: bool = False, input_dim: int = None):
        self._base = base_predictor
        self._two_step = two_step
        self._input_dim = input_dim  # Full input dimension (needed to extract source for two_step)
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the residual, add to source latent, and normalize.
        
        Args:
            x: Input latents. For two_step, this is concat(lat_t, lat_t+1)
        """
        # Get raw prediction (residual)
        residual = self._base.predict_raw(x)
        
        # Extract source latent to add residual to
        if self._two_step:
            # Source is the second half (lat_t+1) of the concatenated input
            half_dim = x.shape[-1] // 2
            src_latent = x[..., half_dim:]
        else:
            src_latent = x
            
        # Add residual to source and normalize
        output = src_latent + residual
        return _normalize_numpy(output)
    
    def __getattr__(self, name):
        """Delegate other attributes to the base predictor."""
        return getattr(self._base, name)


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
    
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Predict without normalization (for residual mode)."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            return self.forward(x_tensor).cpu().numpy()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scikit-learn style predict interface. Returns normalized latents."""
        return _normalize_numpy(self.predict_raw(x))
    
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
    
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Predict without normalization (for residual mode)."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32, device=next(self.parameters()).device)
            return self.forward(x_tensor).cpu().numpy()
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Scikit-learn style predict interface. Returns normalized latents."""
        return _normalize_numpy(self.predict_raw(x))
    
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


class NormalizedRidgeWrapper:
    """
    Wrapper for sklearn Ridge that normalizes predictions to unit norm.
    """
    
    def __init__(self, ridge_model: Ridge):
        self._model = ridge_model
    
    def predict_raw(self, x: np.ndarray) -> np.ndarray:
        """Predict without normalization (for residual mode)."""
        return self._model.predict(x)
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict and normalize output to unit norm."""
        return _normalize_numpy(self.predict_raw(x))
    
    def __getattr__(self, name):
        """Delegate other attributes to the underlying Ridge model."""
        return getattr(self._model, name)


def train_linear_predictor_mse(
    X: np.ndarray,
    Y: np.ndarray,
    ridge_alpha: float = 1e-3,
    seed: int = 0,
) -> NormalizedRidgeWrapper:
    """
    Train a linear predictor using MSE loss with L2 regularization.
    Uses sklearn's Ridge for closed-form solution.
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        ridge_alpha: L2 regularization weight
        seed: Random seed
        
    Returns:
        Wrapped sklearn Ridge model with normalized predictions
    """
    model = Ridge(alpha=ridge_alpha, random_state=seed).fit(X, Y)
    return NormalizedRidgeWrapper(model)


def _train_predictor_on_data(
    X: np.ndarray,
    Y: np.ndarray,
    loss_type: Literal["cosine", "MSE"],
    ridge_alpha: float,
    num_epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
    verbose: bool = False,
):
    """Internal helper to train a predictor on given X, Y data."""
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


def _compute_cv_loss(
    model,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    loss_type: Literal["cosine", "MSE"],
) -> float:
    """Compute validation loss for cross-validation."""
    pred = model.predict_raw(X_val)
    
    if loss_type == "cosine":
        # Cosine similarity loss: 1 - cos(pred, target)
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-12)
        Y_norm = Y_val / (np.linalg.norm(Y_val, axis=-1, keepdims=True) + 1e-12)
        cos_sim = np.sum(pred_norm * Y_norm, axis=-1)
        return float(np.mean(1 - cos_sim))
    else:  # MSE
        return float(np.mean((pred - Y_val) ** 2))


def cross_validate_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    loss_type: Literal["cosine", "MSE"] = "cosine",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 5,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Perform cross-validation to find optimal hyperparameters for the predictor.
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        loss_type: "cosine" or "MSE"
        param_grid: Dictionary of hyperparameters to search. 
                   Default: {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
                   For cosine loss, can also include 'lr' and 'num_epochs'
        n_folds: Number of cross-validation folds
        num_epochs: Default number of training epochs (for cosine loss)
        lr: Default learning rate (for cosine loss)
        device: Device to train on
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Dictionary with:
        - 'best_params': Dict of best hyperparameters
        - 'best_score': Best cross-validation score (lower is better)
        - 'cv_results': List of dicts with params and scores for each combination
    """
    if param_grid is None:
        param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    from itertools import product
    param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_values)]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    cv_results = []
    best_score = float('inf')
    best_params = None
    
    for params in param_combinations:
        fold_scores = []
        
        # Extract parameters
        ridge_alpha = params.get('ridge_alpha', 1e-3)
        current_lr = params.get('lr', lr)
        current_epochs = params.get('num_epochs', num_epochs)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # Train model on fold
            model = _train_predictor_on_data(
                X_train, Y_train, loss_type, ridge_alpha,
                current_epochs, current_lr, device, seed, verbose=False
            )
            
            # Compute validation loss
            val_loss = _compute_cv_loss(model, X_val, Y_val, loss_type)
            fold_scores.append(val_loss)
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        
        cv_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_scores': fold_scores,
        })
        
        if verbose:
            print(f"Params: {params} -> CV score: {mean_score:.6f} +/- {std_score:.6f}")
        
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    
    if verbose:
        print(f"\nBest params: {best_params} with score: {best_score:.6f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'cv_results': cv_results,
    }


def get_matched_predictor(
    encoder: torch.nn.Module,
    data: dict,
    loss_type: Literal["cosine", "MSE"] = "cosine",
    ridge_alpha: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    seed: int = 42,
    two_step: bool = False,
    residual_mode: bool = False,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    verbose: bool = False,
    use_cv: bool = False,
    cv_param_grid: Optional[Dict[str, List[Any]]] = None,
    n_cv_folds: int = 5,
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
        residual_mode: If True, predict (target - source) and add to source latent
        num_epochs: Number of training epochs (only used for cosine loss)
        lr: Learning rate (only used for cosine loss)
        verbose: Whether to print progress
        use_cv: If True, use cross-validation to find optimal hyperparameters
        cv_param_grid: Parameter grid for CV search. If None, uses default grid for ridge_alpha
        n_cv_folds: Number of CV folds
        
    Returns:
        Trained predictor model with scikit-learn style .predict() interface
        If residual_mode=True, returns a ResidualPredictorWrapper
    """
    # Build the latent transition dataset (feeds all data at once, no subsetting)
    X, Y = build_latent_transition_dataset(
        encoder, data, device=device, two_step=two_step, residual_mode=residual_mode
    )
    
    # Determine hyperparameters via CV or use provided values
    if use_cv:
        cv_result = cross_validate_predictor(
            X, Y, loss_type=loss_type, param_grid=cv_param_grid,
            n_folds=n_cv_folds, num_epochs=num_epochs, lr=lr,
            device=device, seed=seed, verbose=verbose
        )
        best_params = cv_result['best_params']
        ridge_alpha = best_params.get('ridge_alpha', ridge_alpha)
        lr = best_params.get('lr', lr)
        num_epochs = best_params.get('num_epochs', num_epochs)
        
        if verbose:
            print(f"Using CV-selected params: ridge_alpha={ridge_alpha}, lr={lr}, num_epochs={num_epochs}")
    
    # Train the final model on all data
    model = _train_predictor_on_data(
        X, Y, loss_type, ridge_alpha, num_epochs, lr, device, seed, verbose
    )
    
    # Wrap in residual predictor if needed
    if residual_mode:
        model = ResidualPredictorWrapper(model, two_step=two_step, input_dim=X.shape[1])
    
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
