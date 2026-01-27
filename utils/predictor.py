import numpy as np
import torch
import torch.nn as nn
from typing import Literal, Optional, Dict, List, Any
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from itertools import product


class LinearPredictor(nn.Module):
    """Simple linear predictor with MSE or cosine loss (gradient descent based)."""
    
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
        Fit the predictor to data using gradient descent.
        
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
    Linear predictor using sklearn's Ridge regression (exact closed-form solution).
    
    This uses the exact analytical solution to ridge regression rather than 
    gradient descent, which is faster and more accurate for linear models.
    Only supports MSE loss (inherent to ridge regression).
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        """
        Initialize the Ridge predictor.
        
        Note: input_dim and output_dim are accepted for API compatibility
        but are not used (sklearn infers dimensions from data).
        """
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_delta = False  # Will be set by trainer
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        ridge_alpha: float = 1e-3,
        verbose: bool = False,
        **kwargs,  # Accept and ignore other args for API compatibility
    ) -> "RidgePredictor":
        """
        Fit the ridge regression model using sklearn's closed-form solution.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            ridge_alpha: L2 regularization parameter (alpha in sklearn)
            verbose: Print fitting info
            **kwargs: Ignored (for API compatibility with LinearPredictor)
        
        Returns:
            self
        """
        if verbose:
            print(f"Fitting Ridge regression (alpha={ridge_alpha})...")
        
        # sklearn Ridge natively supports multi-output regression
        self.model = Ridge(alpha=ridge_alpha, fit_intercept=True)
        self.model.fit(X, Y)
        
        if verbose:
            # Compute training MSE
            train_pred = self.model.predict(X)
            train_mse = np.mean((train_pred - Y) ** 2)
            print(f"  Training MSE: {train_mse:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)


class RandomForestPredictor:
    """
    Predictor using sklearn's Random Forest Regressor.
    
    For multi-output regression, this uses MultiOutputRegressor wrapper
    which fits one random forest per output dimension.
    Only supports MSE loss.
    """
    
    def __init__(self, input_dim: int = None, output_dim: int = None):
        """
        Initialize the Random Forest predictor.
        
        Note: input_dim and output_dim are accepted for API compatibility
        but are not used (sklearn infers dimensions from data).
        """
        self.model = None
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.predict_delta = False  # Will be set by trainer
    
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = False,
        **kwargs,  # Accept and ignore other args for API compatibility
    ) -> "RandomForestPredictor":
        """
        Fit the random forest regression model.
        
        Args:
            X: Input features [n_samples, input_dim]
            Y: Target features [n_samples, output_dim]
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Min samples required to split a node
            min_samples_leaf: Min samples required at a leaf node
            n_jobs: Number of parallel jobs (-1 = all cores)
            random_state: Random seed for reproducibility
            verbose: Print fitting info
            **kwargs: Ignored (for API compatibility)
        
        Returns:
            self
        """
        if verbose:
            print(f"Fitting Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
        
        # RandomForestRegressor natively supports multi-output
        base_rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        
        # Use native multi-output support
        self.model = base_rf
        self.model.fit(X, Y)
        
        if verbose:
            # Compute training MSE
            train_pred = self.model.predict(X)
            train_mse = np.mean((train_pred - Y) ** 2)
            print(f"  Training MSE: {train_mse:.6f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on numpy array input."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        return self.model.predict(X)


def _create_predictor(predictor_type: str, input_dim: int, output_dim: int):
    """Factory function to create a predictor of the specified type."""
    if predictor_type == "linear":
        return LinearPredictor(input_dim, output_dim)
    elif predictor_type == "ridge":
        return RidgePredictor(input_dim, output_dim)
    elif predictor_type == "random_forest":
        return RandomForestPredictor(input_dim, output_dim)
    else:
        raise ValueError(f"Unknown predictor type: {predictor_type}")


def _compute_validation_loss(pred: np.ndarray, Y_val: np.ndarray, loss_type: str) -> float:
    """Compute validation loss (MSE or cosine)."""
    if loss_type == "cosine":
        pred_norm = pred / (np.linalg.norm(pred, axis=-1, keepdims=True) + 1e-12)
        Y_norm = Y_val / (np.linalg.norm(Y_val, axis=-1, keepdims=True) + 1e-12)
        return float(np.mean(1 - np.sum(pred_norm * Y_norm, axis=-1)))
    else:  # mse
        return float(np.mean((pred - Y_val) ** 2))


def cross_validate_predictor(
    X: np.ndarray,
    Y: np.ndarray,
    predictor_type: str = "linear",
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
        predictor_type: Type of predictor ("linear", "ridge", "random_forest")
        loss_type: "mse" or "cosine" (cosine only for linear predictor)
        param_grid: Dictionary of hyperparameters to search
        n_folds: Number of cross-validation folds
        num_epochs: Training epochs (for linear predictor)
        lr: Learning rate (for linear predictor)
        device: Device for training (for linear predictor)
        seed: Random seed
        verbose: Print progress
    
    Returns:
        dict with 'best_params', 'best_score', 'cv_results'.
    """
    # Default param grids by predictor type
    if param_grid is None:
        if predictor_type == "linear":
            param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
        elif predictor_type == "ridge":
            param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}
        elif predictor_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            }
        else:
            param_grid = {'ridge_alpha': [1e-3]}
    
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
            
            model = _create_predictor(predictor_type, X.shape[1], Y.shape[1])
            
            # Fit based on predictor type
            if predictor_type == "linear":
                ridge_alpha = params.get('ridge_alpha', 1e-3)
                current_lr = params.get('lr', lr)
                current_epochs = params.get('num_epochs', num_epochs)
                model.fit(X_train, Y_train, loss_type=loss_type, ridge_alpha=ridge_alpha,
                          num_epochs=current_epochs, lr=current_lr, device=device)
            elif predictor_type == "ridge":
                ridge_alpha = params.get('ridge_alpha', 1e-3)
                model.fit(X_train, Y_train, ridge_alpha=ridge_alpha)
            elif predictor_type == "random_forest":
                model.fit(X_train, Y_train, **params)
            
            # Compute validation loss
            pred = model.predict(X_val)
            val_loss = _compute_validation_loss(pred, Y_val, loss_type)
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
    predictor_type: str = "linear",
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
        predictor_type: Type of predictor ("linear", "ridge", "random_forest")
        loss_type: "mse" or "cosine" (cosine only for linear predictor)
        param_grid: Dictionary of hyperparameter lists to search
        holdout_fraction: Fraction of each patient's samples to hold out (0.0-1.0).
                          If 1.0, all samples from the patient are held out.
        folds_per_patient: Number of CV folds per patient. When holdout_fraction < 1.0,
                           different random subsets of the patient's samples are held out
                           for each fold. When holdout_fraction == 1.0, this is effectively 1.
        num_epochs: Number of training epochs per fold (for linear predictor)
        lr: Learning rate (for linear predictor)
        device: Device for training (for linear predictor)
        seed: Random seed for reproducibility
        verbose: Print detailed progress
    
    Returns:
        Dictionary with 'best_params', 'best_score', 'cv_results', 'fold_details'
    """
    # Default param grids by predictor type
    if param_grid is None:
        if predictor_type == "linear":
            param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]}
        elif predictor_type == "ridge":
            param_grid = {'ridge_alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]}
        elif predictor_type == "random_forest":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
            }
        else:
            param_grid = {'ridge_alpha': [1e-3]}
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get unique patients
    unique_patients = np.unique(patient_ids)
    n_patients = len(unique_patients)
    
    # Check for edge case: can't do patient-based CV with only 1 patient
    if n_patients < 2 and holdout_fraction >= 1.0:
        raise ValueError(
            f"Patient-based cross-validation requires at least 2 unique patients when "
            f"holdout_fraction=1.0, but only found {n_patients} patient(s): {unique_patients.tolist()}. "
            f"Either use regular cross-validation (--cross_validate without --patient_cv), "
            f"or set holdout_fraction < 1.0 to allow partial holdout within the single patient."
        )
    
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
                
                model = _create_predictor(predictor_type, X.shape[1], Y.shape[1])
                
                # Fit based on predictor type
                if predictor_type == "linear":
                    ridge_alpha = params.get('ridge_alpha', 1e-3)
                    current_lr = params.get('lr', lr)
                    current_epochs = params.get('num_epochs', num_epochs)
                    model.fit(
                        X_train, Y_train,
                        loss_type=loss_type,
                        ridge_alpha=ridge_alpha,
                        num_epochs=current_epochs,
                        lr=current_lr,
                        device=device
                    )
                elif predictor_type == "ridge":
                    ridge_alpha = params.get('ridge_alpha', 1e-3)
                    model.fit(X_train, Y_train, ridge_alpha=ridge_alpha)
                elif predictor_type == "random_forest":
                    model.fit(X_train, Y_train, **params)
                
                # Compute validation loss
                pred = model.predict(X_val)
                val_loss = _compute_validation_loss(pred, Y_val, loss_type)
                
                fold_scores.append(val_loss)
                fold_details.append({
                    'patient': patient if not isinstance(patient, np.generic) else patient.item(),
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
        'unique_patients': [p.item() if isinstance(p, np.generic) else p for p in unique_patients],
    }