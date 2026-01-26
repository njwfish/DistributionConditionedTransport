import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, List, Any, Union
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from itertools import product


# Type alias for all predictor types
PredictorType = Union["LinearPredictor", "RidgePredictor", "KernelRidgePredictor", "PLSPredictor", "MLPPredictor"]


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


class RidgePredictor:
    """
    Sklearn-based Ridge regression predictor with closed-form solution.
    
    This is equivalent to LinearPredictor with MSE loss, but uses sklearn's
    exact closed-form solution instead of gradient descent. Does not support
    cosine loss.
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        """
        Initialize the predictor. Dimensions are inferred from data during fit.
        """
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ridge_alpha: float = 1.0,
        **kwargs,  # Accept but ignore other arguments for compatibility
    ) -> "RidgePredictor":
        """
        Fit the predictor using sklearn's Ridge regression.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            ridge_alpha: Regularization strength (sklearn's alpha parameter)
        """
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        # Use Ridge regression with multi-output support
        self.model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        self.model.fit(X, Y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)


class KernelRidgePredictor:
    """
    Kernel Ridge Regression predictor using sklearn.
    
    Extends linear Ridge regression to capture nonlinear relationships via the 
    kernel trick. Uses RBF (Gaussian) kernel by default, which can model complex
    nonlinear mappings while maintaining good regularization properties.
    
    Well-suited for moderate sample sizes (~1000) with high-dimensional data,
    as it provides a closed-form solution and the kernel parameters can be
    cross-validated.
    
    Key hyperparameters:
        - alpha: Regularization strength (like Ridge)
        - gamma: RBF kernel bandwidth (controls smoothness)
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        """
        Initialize the predictor. Dimensions are inferred from data during fit.
        """
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        alpha: float = 1.0,
        gamma: float = None,
        **kwargs,  # Accept but ignore other arguments for compatibility
    ) -> "KernelRidgePredictor":
        """
        Fit the predictor using sklearn's Kernel Ridge regression.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            alpha: Regularization strength
            gamma: RBF kernel bandwidth. If None, defaults to 1/n_features.
        """
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        # Use Kernel Ridge with RBF kernel
        # gamma=None lets sklearn use 1/n_features as default
        self.model = KernelRidge(alpha=alpha, kernel='rbf', gamma=gamma)
        self.model.fit(X, Y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)


class PLSPredictor:
    """
    Partial Least Squares (PLS) Regression predictor using sklearn.
    
    PLS is specifically designed for situations with:
        - High-dimensional inputs and outputs
        - Potential multicollinearity in features
        - Limited sample sizes relative to dimensionality
        - Need to find underlying latent structure
    
    PLS finds latent components that explain variance in both X and Y
    simultaneously, making it well-suited for mapping between two high-dimensional
    spaces when the effective dimensionality is much lower.
    
    Key hyperparameter:
        - n_components: Number of latent components to extract (controls complexity)
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        """
        Initialize the predictor. Dimensions are inferred from data during fit.
        """
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_components: int = 10,
        **kwargs,  # Accept but ignore other arguments for compatibility
    ) -> "PLSPredictor":
        """
        Fit the predictor using sklearn's PLS regression.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            n_components: Number of latent components. Should be <= min(n_samples, n_features).
                          Higher values allow more complex mappings but risk overfitting.
        """
        self.input_dim = X.shape[1]
        self.output_dim = Y.shape[1]
        
        # Ensure n_components doesn't exceed the limits
        max_components = min(X.shape[0], X.shape[1], Y.shape[1])
        n_components = min(n_components, max_components)
        
        self.model = PLSRegression(n_components=n_components)
        self.model.fit(X, Y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)


class MLPPredictor(nn.Module):
    """
    Simple MLP predictor with dropout and weight decay regularization.
    
    Architecture: input -> hidden1 -> ReLU -> Dropout -> hidden2 -> ReLU -> Dropout -> output
    
    Designed for small datasets (~1000 samples) with moderate dimensionality (~50).
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        loss_type: Literal["mse", "cosine"] = "mse",
        weight_decay: float = 1e-3,
        num_epochs: int = 1000,
        lr: float = 1e-3,
        device: torch.device = None,
        verbose: bool = False,
        **kwargs,  # Accept but ignore other arguments for compatibility
    ) -> "MLPPredictor":
        """
        Fit the MLP predictor to data.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            loss_type: "mse" or "cosine"
            weight_decay: L2 regularization weight (Adam weight_decay parameter)
            num_epochs: Number of training epochs
            lr: Learning rate
            device: Device to train on
            verbose: Print progress
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        
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


def create_predictor(
    predictor_type: Literal["linear", "ridge", "kernel_ridge", "pls", "mlp"],
    input_dim: int,
    output_dim: int,
    **kwargs,
) -> PredictorType:
    """
    Factory function to create a predictor of the specified type.
    
    Args:
        predictor_type: Type of predictor ("linear", "ridge", "kernel_ridge", "pls", or "mlp")
        input_dim: Input dimension
        output_dim: Output dimension
        **kwargs: Additional arguments passed to predictor constructor
    
    Returns:
        Predictor instance
    """
    if predictor_type == "linear":
        return LinearPredictor(input_dim, output_dim)
    elif predictor_type == "ridge":
        return RidgePredictor(input_dim, output_dim)
    elif predictor_type == "kernel_ridge":
        return KernelRidgePredictor(input_dim, output_dim)
    elif predictor_type == "pls":
        return PLSPredictor(input_dim, output_dim)
    elif predictor_type == "mlp":
        hidden_dim = kwargs.get('hidden_dim', 128)
        dropout = kwargs.get('dropout', 0.1)
        return MLPPredictor(input_dim, output_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


def get_default_param_grid(predictor_type: Literal["linear", "ridge", "kernel_ridge", "pls", "mlp"]) -> Dict[str, List[Any]]:
    """
    Get the default parameter grid for cross-validation based on predictor type.
    
    Args:
        predictor_type: Type of predictor
    
    Returns:
        Dictionary of parameter names to lists of values to try
    """
    if predictor_type == "linear":
        return {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    elif predictor_type == "ridge":
        return {'ridge_alpha': [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
    elif predictor_type == "kernel_ridge":
        # alpha: regularization, gamma: RBF kernel bandwidth
        # gamma=None means sklearn uses 1/n_features
        return {
            'alpha': [1e-3, 1e-2, 1e-1, 1.0, 10.0],
            'gamma': [None, 1e-4, 1e-3, 1e-2, 1e-1],
        }
    elif predictor_type == "pls":
        # n_components: number of latent components
        # For ~250-dim data with ~1000 samples, reasonable range is 5-50
        return {
            'n_components': [5, 10, 20, 30, 50, 75, 100],
        }
    elif predictor_type == "mlp":
        return {
            'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
            'dropout': [0.0, 0.1, 0.2, 0.3],
        }
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


def cross_validate_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    predictor_type: Literal["linear", "ridge", "kernel_ridge", "pls", "mlp"] = "linear",
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
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        predictor_type: Type of predictor ("linear", "ridge", or "mlp")
        loss_type: Loss function ("mse" or "cosine"). Note: "ridge" only supports "mse".
        param_grid: Dictionary of hyperparameter lists to search. If None, uses defaults.
        n_folds: Number of cross-validation folds
        num_epochs: Number of training epochs (ignored for "ridge")
        lr: Learning rate (ignored for "ridge")
        device: Device for training (ignored for "ridge")
        seed: Random seed for reproducibility
        verbose: Print detailed progress
    
    Returns:
        Dict with 'best_params', 'best_score', 'cv_results'.
    """
    # sklearn-based predictors don't support cosine loss
    sklearn_predictors = ["ridge", "kernel_ridge", "pls"]
    if predictor_type in sklearn_predictors and loss_type == "cosine":
        raise ValueError(f"{predictor_type} predictor does not support cosine loss. Use 'linear' or 'mlp' instead.")
    
    if param_grid is None:
        param_grid = get_default_param_grid(predictor_type)
    
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
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            Y_train, Y_val = Y[train_idx], Y[val_idx]
            
            # For MLP, separate constructor params (dropout) from fit params (weight_decay)
            if predictor_type == "mlp":
                constructor_params = {k: v for k, v in params.items() if k in ['dropout', 'hidden_dim']}
                fit_params = {k: v for k, v in params.items() if k not in ['dropout', 'hidden_dim']}
            else:
                constructor_params = {}
                fit_params = params
            
            # Create model with constructor params
            model = create_predictor(predictor_type, X.shape[1], Y.shape[1], **constructor_params)
            
            # sklearn-based predictors (ridge, kernel_ridge, pls) just need fit params
            if predictor_type in ["ridge", "kernel_ridge", "pls"]:
                model.fit(X_train, Y_train, **fit_params)
            elif predictor_type == "mlp":
                model.fit(
                    X_train, Y_train,
                    loss_type=loss_type,
                    num_epochs=num_epochs,
                    lr=lr,
                    device=device,
                    **fit_params
                )
            else:  # linear
                model.fit(
                    X_train, Y_train,
                    loss_type=loss_type,
                    num_epochs=num_epochs,
                    lr=lr,
                    device=device,
                    **fit_params
                )
            
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


def cross_validate_predictor_by_patient(
    X: np.ndarray,
    Y: np.ndarray,
    patient_ids: np.ndarray,
    predictor_type: Literal["linear", "ridge", "kernel_ridge", "pls", "mlp"] = "linear",
    loss_type: Literal["mse", "cosine"] = "mse",
    param_grid: Optional[Dict[str, List[Any]]] = None,
    holdout_fraction: float = 1.0,
    folds_per_patient: int = 1,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    device: torch.device = None,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Patient-based cross-validation to find optimal hyperparameters.
    
    Instead of random k-fold splits, this function holds out samples from one
    patient at a time. This ensures that the validation set contains samples
    from a single patient, testing the model's ability to generalize across patients.
    
    Args:
        X: Input features [n_samples, input_dim]
        Y: Target features [n_samples, output_dim]
        patient_ids: Patient ID for each sample [n_samples]
        predictor_type: Type of predictor ("linear", "ridge", or "mlp")
        loss_type: "mse" or "cosine". Note: sklearn predictors (ridge, kernel_ridge, pls) only support "mse".
        param_grid: Dictionary of hyperparameter lists to search. If None, uses defaults.
        holdout_fraction: Fraction of each patient's samples to hold out (0.0-1.0).
                          If 1.0, all samples from the patient are held out.
        folds_per_patient: Number of CV folds per patient. When holdout_fraction < 1.0,
                           different random subsets of the patient's samples are held out
                           for each fold. When holdout_fraction == 1.0, this is effectively 1.
        num_epochs: Number of training epochs per fold (ignored for sklearn predictors)
        lr: Learning rate (ignored for sklearn predictors)
        device: Device for training (ignored for sklearn predictors)
        seed: Random seed for reproducibility
        verbose: Print detailed progress
    
    Returns:
        Dictionary with 'best_params', 'best_score', 'cv_results', 'fold_details'
    """
    # sklearn-based predictors don't support cosine loss
    sklearn_predictors = ["ridge", "kernel_ridge", "pls"]
    if predictor_type in sklearn_predictors and loss_type == "cosine":
        raise ValueError(f"{predictor_type} predictor does not support cosine loss. Use 'linear' or 'mlp' instead.")
    
    if param_grid is None:
        param_grid = get_default_param_grid(predictor_type)
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    if verbose:
        print(f"Patient-based CV: {n_patients} unique patients")
        print(f"  Predictor type: {predictor_type}")
        print(f"  Holdout fraction: {holdout_fraction}")
        print(f"  Folds per patient: {folds_per_patient}")
    
    # When holdout_fraction is 1.0, multiple folds per patient are redundant
    effective_folds_per_patient = 1 if holdout_fraction >= 1.0 else folds_per_patient
    total_folds = n_patients * effective_folds_per_patient
    
    if verbose:
        print(f"  Total folds: {total_folds}")
    
    param_names = list(param_grid.keys())
    param_combinations = [dict(zip(param_names, combo)) for combo in product(*param_grid.values())]
    
    rng = np.random.default_rng(seed)
    
    cv_results = []
    best_score = float('inf')
    best_params = None
    
    for params in param_combinations:
        fold_scores = []
        fold_details = []
        
        for patient in unique_patients:
            # Get indices for this patient
            patient_mask = patient_ids == patient
            patient_indices = np.where(patient_mask)[0]
            other_indices = np.where(~patient_mask)[0]
            n_patient_samples = len(patient_indices)
            
            for fold_idx in range(effective_folds_per_patient):
                # Determine how many samples to hold out
                n_holdout = max(1, int(n_patient_samples * holdout_fraction))
                
                if holdout_fraction >= 1.0:
                    # Hold out all samples from this patient
                    val_indices = patient_indices
                    # Training set is all samples from other patients
                    train_indices = other_indices
                else:
                    # Randomly select a subset of this patient's samples to hold out
                    # Use a different random subset for each fold
                    fold_seed = seed + hash(str(patient)) + fold_idx
                    fold_rng = np.random.default_rng(fold_seed)
                    
                    holdout_patient_indices = fold_rng.choice(
                        patient_indices, size=n_holdout, replace=False
                    )
                    val_indices = holdout_patient_indices
                    
                    # Training set: other patients + non-held-out samples from this patient
                    non_holdout_patient_indices = np.setdiff1d(patient_indices, holdout_patient_indices)
                    train_indices = np.concatenate([other_indices, non_holdout_patient_indices])
                
                X_train, X_val = X[train_indices], X[val_indices]
                Y_train, Y_val = Y[train_indices], Y[val_indices]
                
                # For MLP, separate constructor params (dropout) from fit params (weight_decay)
                if predictor_type == "mlp":
                    constructor_params = {k: v for k, v in params.items() if k in ['dropout', 'hidden_dim']}
                    fit_params = {k: v for k, v in params.items() if k not in ['dropout', 'hidden_dim']}
                else:
                    constructor_params = {}
                    fit_params = params
                
                # Create model with constructor params
                model = create_predictor(predictor_type, X.shape[1], Y.shape[1], **constructor_params)
                
                # sklearn-based predictors (ridge, kernel_ridge, pls) just need fit params
                if predictor_type in ["ridge", "kernel_ridge", "pls"]:
                    model.fit(X_train, Y_train, **fit_params)
                elif predictor_type == "mlp":
                    model.fit(
                        X_train, Y_train,
                        loss_type=loss_type,
                        num_epochs=num_epochs,
                        lr=lr,
                        device=device,
                        **fit_params
                    )
                else:  # linear
                    model.fit(
                        X_train, Y_train,
                        loss_type=loss_type,
                        num_epochs=num_epochs,
                        lr=lr,
                        device=device,
                        **fit_params
                    )
                
                # Compute validation loss
                pred = model.predict(X_val)
                if loss_type == "cosine":
                    pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-12)
                    Y_norm = Y_val / (np.linalg.norm(Y_val, axis=-1, keepdims=True) + 1e-12)
                    val_loss = float(np.mean(1 - np.sum(pred_norm * Y_norm, axis=-1)))
                else:
                    val_loss = float(np.mean((pred - Y_val) ** 2))
                
                fold_scores.append(val_loss)
                fold_details.append({
                    'patient': patient,
                    'fold_idx': fold_idx,
                    'n_train': len(train_indices),
                    'n_val': len(val_indices),
                    'val_loss': val_loss,
                })
                
                if verbose:
                    print(f"  Patient {patient}, Fold {fold_idx + 1}: "
                          f"train={len(train_indices)}, val={len(val_indices)}, loss={val_loss:.6f}")
        
        mean_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)
        cv_results.append({
            'params': params,
            'mean_score': mean_score,
            'std_score': std_score,
            'fold_details': fold_details,
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
        'n_patients': n_patients,
        'unique_patients': unique_patients.tolist(),
    }
