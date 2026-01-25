import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, List, Any
from sklearn.model_selection import KFold
from itertools import product


class LinearPredictor(nn.Module):
    """Simple linear predictor with MSE or cosine loss."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        loss_type: Literal["mse", "cosine"] = "mse",
        ridge_alpha: float = 1e-3,
        num_epochs: int = 1000,
        lr: float = 1e-2,
        device: torch.device = None,
        verbose: bool = False,
    ) -> "LinearPredictor":
        """
        Fit the predictor to data.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            loss_type: "mse" or "cosine"
            ridge_alpha: L2 regularization weight
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on
            verbose: Print progress
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        Y_t = torch.tensor(Y, dtype=torch.float32, device=device)
        
        cos_sim = nn.CosineSimilarity(dim=-1)
        
        self.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            pred = self(X_t)
            
            if loss_type == "cosine":
                loss = (1 - cos_sim(pred, Y_t)).mean()
            else:  # mse
                loss = (pred - Y_t).pow(2).mean()
            
            if ridge_alpha > 0:
                loss = loss + ridge_alpha * torch.sum(self.linear.weight ** 2)
            
            loss.backward()
            optimizer.step()
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        self.eval()
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            return self(X_t).cpu().numpy()


def cross_validate_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    loss_type: Literal["mse", "cosine"] = "mse",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    n_folds: int = 5,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    device: torch.device = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Cross-validation to find optimal hyperparameters.
    
    Returns dict with 'best_params', 'best_score', 'cv_results'.
    """
    if param_grid is None:
        param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    param_names = list(param_grid.keys())
    param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_grid.values())]
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    cv_results = []
    best_score = float('inf')
    best_params = None
    
    for params in param_combinations:
        fold_scores = []
        ridge_alpha = params.get('ridge_alpha', 1e-3)
        current_lr = params.get('lr', lr)
        current_epochs = params.get('num_epochs', num_epochs)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            model = LinearPredictor(X.shape[1], Y.shape[1])
            model.fit(X_train, Y_train, loss_type=loss_type, ridge_alpha=ridge_alpha,
                      num_epochs=current_epochs, lr=current_lr, device=device)
            
            # Compute validation loss
            pred = model.predict(X_val)
            if loss_type == "cosine":
                pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-12)
                Y_norm = Y_val / (np.linalg.norm(Y_val, axis=-1, keepdims=True) + 1e-12)
                val_loss = float(np.mean(1 - np.sum(pred_norm * Y_norm, axis=-1)))
            else:
                val_loss = float(np.mean((pred - Y_val) ** 2))
            
            fold_scores.append(val_loss)
        
        mean_score = np.mean(fold_scores)
        cv_results.append({'params': params, 'mean_score': mean_score, 'std_score': np.std(fold_scores)})
        
        if verbose:
            print(f"Params: {params} -> CV score: {mean_score:.6f}")
        
        if mean_score < best_score:
            best_score = mean_score
            best_params = params
    
    if verbose:
        print(f"\nBest params: {best_params} with score: {best_score:.6f}")
    
    return {'best_params': best_params, 'best_score': best_score, 'cv_results': cv_results}
