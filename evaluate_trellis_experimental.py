import os
import sys
import argparse
import hashlib
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Literal

from utils.latents import normalize_latent, normalize_latent_np
from utils.predictor import (
    LinearPredictor,
    RidgePredictor,
    RandomForestPredictor,
    cross_validate_predictor,
    cross_validate_predictor_by_patient,
)
from utils.experiment_utils import (
    load_config,
    load_experiment,
    is_mfm_model,
    find_experiment_dir,
)
from generator.losses import wasserstein, mmd, sliced_wasserstein_distance

# ============================================================================
# Checkpointing Utilities
# ============================================================================

CHECKPOINT_DIR = "trellis_running_eval"


def compute_args_hash(args: argparse.Namespace) -> str:
    """
    Compute a deterministic hash of the relevant arguments.
    
    This hash is used to uniquely identify a run configuration, so that
    different runs don't get mixed up when checkpointing.
    """
    # Extract relevant arguments that affect the evaluation results
    args_dict = {
        'experiment_dir': args.experiment_dir,
        'match': sorted(args.match) if args.match else [],
        'outputs_dir': args.outputs_dir,
        'compute_baseline': args.compute_baseline,
        'metric': args.metric,
        'use_predictor': args.use_predictor,
        'predictor_type': args.predictor_type,
        'cross_validate': args.cross_validate,
        'predict_delta': args.predict_delta,
        'predictor_loss': args.predictor_loss,
        'patient_cv': args.patient_cv,
        'patient_holdout_fraction': args.patient_holdout_fraction,
        'folds_per_patient': args.folds_per_patient,
        'cheat_mode': args.cheat_mode,
    }
    
    # Create a deterministic JSON string and hash it
    args_json = json.dumps(args_dict, sort_keys=True)
    return hashlib.md5(args_json.encode()).hexdigest()[:12]


def get_checkpoint_path(args_hash: str) -> str:
    """Get the checkpoint file path for a given args hash."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"checkpoint_{args_hash}.json")


def load_checkpoint(checkpoint_path: str) -> dict:
    """
    Load checkpoint from file if it exists.
    
    Returns:
        Dictionary with 'completed_indices', 'model_metrics', 'baseline_metrics', 
        'sample_info', or empty dict if no checkpoint exists.
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  {len(checkpoint.get('completed_indices', []))} samples already completed")
        return checkpoint
    return {
        'completed_indices': [],
        'model_metrics': {},
        'baseline_metrics': {},
        'sample_info': {},
    }


def save_checkpoint(checkpoint_path: str, checkpoint: dict):
    """Save checkpoint to file."""
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def is_conditioned_encoder(cfg) -> bool:
    """
    Detect if the encoder uses cell_cond conditioning based on the config.
    
    This checks if the loss target contains "conditioned", which indicates
    that cell_cond should be concatenated with input features before encoding.
    """
    loss_target = cfg.get("loss", {}).get("_target_", "")
    return "conditioned" in loss_target.lower()


def compute_metric(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    metric: Literal["w1", "mmd_energy", "mmd_rbf", "swd"] = "w1",
    swd_subsample_rounds: int = 100,
) -> float:
    """Compute distance metric between two distributions.
    
    Args:
        pred: Predicted samples tensor
        target: Target samples tensor
        metric: Metric to compute:
            - "w1": Wasserstein-1 distance (exact OT)
            - "mmd_energy": MMD with energy kernel (parameter-free)
            - "mmd_rbf": MMD with RBF kernel (uses median heuristic for bandwidth)
            - "swd": Sliced Wasserstein Distance
        swd_subsample_rounds: Number of subsampling rounds for SWD when sample sizes
            differ. The larger distribution is subsampled to match the smaller one,
            SWD is computed, and this is averaged over multiple rounds. Default: 100.
    
    Returns:
        Computed metric value
    """
    if metric == "w1":
        return wasserstein(pred, target, p=1)
    elif metric == "mmd_energy":
        return mmd(pred, target, kernel='energy').item()
    elif metric == "mmd_rbf":
        return mmd(pred, target, kernel='rbf').item()
    elif metric == "swd":
        n_pred, n_target = pred.shape[0], target.shape[0]
        
        # If sample sizes match, compute SWD directly
        if n_pred == n_target:
            return sliced_wasserstein_distance(pred, target).item()
        
        # Otherwise, subsample the larger distribution multiple times and average
        min_size = min(n_pred, n_target)
        swd_values = []
        
        for _ in range(swd_subsample_rounds):
            # Subsample the larger distribution
            if n_pred > n_target:
                indices = torch.randperm(n_pred, device=pred.device)[:min_size]
                pred_sub = pred[indices]
                target_sub = target
            else:
                pred_sub = pred
                indices = torch.randperm(n_target, device=target.device)[:min_size]
                target_sub = target[indices]
            
            swd_val = sliced_wasserstein_distance(pred_sub, target_sub).item()
            swd_values.append(swd_val)
        
        return np.mean(swd_values)
    else:
        raise ValueError(f"Unknown metric: {metric}")

# ============================================================================
# Latent Computation and Predictor Training
# ============================================================================

def compute_latents(
    encoder: torch.nn.Module,
    samples: list,
    device: torch.device,
    split_name: str = "dataset",
    use_cell_cond: bool = False,
) -> tuple:
    """
    Compute E(x0) and E(x1) for all samples.
    
    Args:
        encoder: The encoder model
        samples: List of samples from the dataset
        device: Device to use for computation
        split_name: Name of the split (for logging)
        use_cell_cond: If True, concatenate cell_cond with x0/x1 before encoding
                       (for models trained with conditioned encoder)
    
    Returns:
        (source_latents, target_latents, treat_conds)
    """
    print(f"Computing {split_name} latents for {len(samples)} samples...")
    if use_cell_cond:
        print(f"  Using cell_cond conditioning for encoder input")
    
    source_latents = []
    target_latents = []
    treat_conds = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
            
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)
            
            if use_cell_cond:
                # Concatenate cell_cond with samples, just like during training
                # x0: (1, num_cells, 43) -> (1, num_cells, 45)
                cell_cond_source_tensor = torch.tensor(cell_cond_source, dtype=torch.float32, device=device).unsqueeze(0)
                cell_cond_target_tensor = torch.tensor(cell_cond_target, dtype=torch.float32, device=device).unsqueeze(0)
                x0_input = torch.cat([x0_tensor, cell_cond_source_tensor], dim=-1)
                x1_input = torch.cat([x1_tensor, cell_cond_target_tensor], dim=-1)
            else:
                x0_input = x0_tensor
                x1_input = x1_tensor
            
            source_latents.append(encoder(x0_input).cpu().numpy())
            target_latents.append(encoder(x1_input).cpu().numpy())
            
            # TODO: make sure this is correct.
            # treat_cond is the same for all cells in a sample, so take first row
            treat_conds.append(treat_cond[0:1])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")
    
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    treat_conds = np.vstack(treat_conds)
    
    return source_latents, target_latents, treat_conds


def train_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    treat_conds: np.ndarray,
    device: torch.device,
    predictor_type: str = "linear",
    ridge_alpha: float = 1e-3,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    verbose: bool = True,
    cross_validate: bool = False,
    predict_delta: bool = False,
    loss_type: str = "mse",
    patient_ids: np.ndarray = None,
    patient_cv: bool = False,
    patient_holdout_fraction: float = 1.0,
    folds_per_patient: int = 1,
):
    """
    Train a predictor to map source latents (conditioned on treatment) to target latents.
    
    The predictor is always conditioned on the treatment condition by concatenating
    the one-hot encoded treatment vector with the source latents.
    
    Args:
        source_latents: Source latents from encoder, shape (N, latent_dim)
        target_latents: Target latents from encoder, shape (N, latent_dim)
        treat_conds: Treatment conditions (one-hot), shape (N, num_treatments)
        device: Device for training
        predictor_type: Type of predictor ("linear", "ridge", "random_forest")
        ridge_alpha: Ridge regularization coefficient (for linear/ridge)
        num_epochs: Number of training epochs (for linear)
        lr: Learning rate (for linear)
        verbose: Whether to print training progress
        cross_validate: Whether to use cross-validation to find optimal hyperparameters
        predict_delta: Whether to predict (target - source) instead of target directly
        loss_type: Loss function for training ("mse" or "cosine", cosine only for linear)
        patient_ids: Patient ID for each sample, shape (N,). Required if patient_cv=True.
        patient_cv: Whether to use patient-based cross-validation instead of random k-fold.
        patient_holdout_fraction: Fraction of each patient's samples to hold out (0.0-1.0).
        folds_per_patient: Number of CV folds per patient when patient_holdout_fraction < 1.0.
    
    Returns:
        Trained predictor (LinearPredictor, RidgePredictor, or RandomForestPredictor)
    """
    # Validate loss_type for non-linear predictors
    if predictor_type in ["ridge", "random_forest"] and loss_type != "mse":
        print(f"  WARNING: {predictor_type} predictor only supports MSE loss. Ignoring loss_type='{loss_type}'.")
        loss_type = "mse"
    
    # Concatenate source latents with treatment condition
    # This conditions the predictor on the treatment
    source_latents_conditioned = np.concatenate([source_latents, treat_conds], axis=1)
    
    # Determine prediction target
    predictor_type_name = predictor_type.upper()
    if predict_delta:
        prediction_target = target_latents - source_latents
        print(f"Training {predictor_type_name} predictor (DELTA mode: predicting target - source)...")
    else:
        prediction_target = target_latents
        print(f"Training {predictor_type_name} predictor...")
    
    print(f"  Source latents shape: {source_latents.shape}")
    print(f"  Treatment conditions shape: {treat_conds.shape}")
    print(f"  Conditioned input shape: {source_latents_conditioned.shape} -> {prediction_target.shape}")
    
    input_dim = source_latents_conditioned.shape[1]
    output_dim = prediction_target.shape[1]
    
    # Set up default param grid based on predictor type
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
        raise ValueError(f"Unknown predictor type: {predictor_type}")
    
    best_params = {}
    
    # Cross-validation for hyperparameter tuning
    if cross_validate:
        if patient_cv:
            # Patient-based cross-validation
            if patient_ids is None:
                raise ValueError("patient_ids must be provided when patient_cv=True")
            
            print(f"  Running PATIENT-BASED cross-validation to find optimal hyperparameters...")
            print(f"    Holdout fraction: {patient_holdout_fraction}")
            print(f"    Folds per patient: {folds_per_patient}")
            
            cv_results = cross_validate_predictor_by_patient(
                source_latents_conditioned,
                prediction_target,
                patient_ids=patient_ids,
                predictor_type=predictor_type,
                loss_type=loss_type,
                param_grid=param_grid,
                holdout_fraction=patient_holdout_fraction,
                folds_per_patient=folds_per_patient,
                num_epochs=num_epochs,
                lr=lr,
                device=device,
                verbose=verbose,
            )
            print(f"  Patient-based CV complete. {cv_results['n_patients']} patients evaluated.")
            print(f"  Patients: {cv_results['unique_patients']}")
        else:
            # Standard random k-fold cross-validation
            print(f"  Running cross-validation to find optimal hyperparameters...")
            cv_results = cross_validate_predictor(
                source_latents_conditioned,
                prediction_target,
                predictor_type=predictor_type,
                loss_type=loss_type,
                param_grid=param_grid,
                n_folds=10,
                num_epochs=num_epochs,
                lr=lr,
                device=device,
                verbose=verbose,
            )
        
        best_params = cv_results['best_params']
        print(f"  Cross-validation complete. Best params: {best_params}")
    
    # Create and train the predictor
    if predictor_type == "linear":
        ridge_alpha_final = best_params.get('ridge_alpha', ridge_alpha)
        print(f"  loss_type={loss_type}, ridge_alpha={ridge_alpha_final}, lr={lr}, epochs={num_epochs}")
        
        predictor = LinearPredictor(input_dim, output_dim)
        predictor.fit(
            source_latents_conditioned,
            prediction_target,
            loss_type=loss_type,
            ridge_alpha=ridge_alpha_final,
            num_epochs=num_epochs,
            lr=lr,
            device=device,
            verbose=verbose,
        )
    elif predictor_type == "ridge":
        ridge_alpha_final = best_params.get('ridge_alpha', ridge_alpha)
        print(f"  ridge_alpha={ridge_alpha_final}")
        
        predictor = RidgePredictor(input_dim, output_dim)
        predictor.fit(
            source_latents_conditioned,
            prediction_target,
            ridge_alpha=ridge_alpha_final,
            verbose=verbose,
        )
    elif predictor_type == "random_forest":
        n_estimators = best_params.get('n_estimators', 100)
        max_depth = best_params.get('max_depth', None)
        print(f"  n_estimators={n_estimators}, max_depth={max_depth}")
        
        predictor = RandomForestPredictor(input_dim, output_dim)
        predictor.fit(
            source_latents_conditioned,
            prediction_target,
            n_estimators=n_estimators,
            max_depth=max_depth,
            verbose=verbose,
        )
    
    # Store metadata for inference
    predictor.predict_delta = predict_delta
    
    return predictor


# Keep old function name as alias for backwards compatibility
def train_linear_predictor(*args, **kwargs):
    """Deprecated: Use train_predictor instead."""
    return train_predictor(*args, **kwargs)


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    treat_cond: np.ndarray,
    device: torch.device,
    metric: Literal["w1", "mmd_energy", "mmd_rbf", "swd"] = "w1",
    predictor = None,
    compute_baseline: bool = False,
    normalize_predicted_latent: bool = True,
    is_mfm: bool = False,
):
    """
    Evaluate the model on a single sample using precomputed latents.
    
    Args:
        generator: The generator model
        x0: Source samples, shape (num_cells, data_dim)
        x1: Target samples, shape (num_cells, data_dim)
        source_latent: Precomputed source latent, shape (1, latent_dim)
        target_latent: Precomputed target latent, shape (1, latent_dim)
        treat_cond: Treatment condition (one-hot), shape (1, num_treatments)
        device: Device for computation
        metric: Metric to compute ("w1", "mmd_energy", "mmd_rbf", or "swd")
        predictor: Optional predictor to predict target latent from source latent + treat_cond
        compute_baseline: Whether to compute baseline metric (x0 vs x1)
        normalize_predicted_latent: Whether to normalize predicted latent
        is_mfm: Whether the model is MFM variant.
                If True, zeros are used for target_latent and treat_cond is passed to generator.
    
    Returns:
        Dictionary with 'x1_pred', 'model_metric', and optionally 'baseline_metric'
    """
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    baseline_metric = None
    if compute_baseline:
        baseline_metric = compute_metric(x0_tensor, x1_tensor, metric=metric)
    
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    treat_cond_tensor = torch.tensor(treat_cond, dtype=torch.float32, device=device)
    
    if is_mfm:
        # MFM models use zeros for target latent (same as during training)
        target_latent_tensor = torch.zeros_like(source_latent_tensor)
    elif predictor is not None:
        # Concatenate source latent with treatment condition for the predictor
        # The predictor was trained with this conditioning
        source_latent_conditioned = np.concatenate([source_latent, treat_cond], axis=1)
        predicted = predictor.predict(source_latent_conditioned)
        
        # If predictor was trained in delta mode, add the predicted delta to source
        if getattr(predictor, 'predict_delta', False):
            predicted_target_latent = source_latent + predicted
        else:
            predicted_target_latent = predicted
        
        if normalize_predicted_latent:
            predicted_target_latent = normalize_latent_np(predicted_target_latent)
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
        target_latent_tensor = torch.tensor(target_latent, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        # Generate x1_pred from x0 using source latent (and target latent)
        if is_mfm:
            # MFM generator requires treat_cond
            x1_pred = generator.sample(
                x0_tensor,              # [N, dim] - source samples
                source_latent_tensor,   # [1, latent_dim]
                target_latent_tensor,   # [1, latent_dim] - zeros for MFM
                treat_cond_tensor,      # [1, treat_dim] - treatment condition
            )
        else:
            # Standard generator doesn't use treat_cond
            x1_pred = generator.sample(
                x0_tensor,              # [N, dim] - source samples
                source_latent_tensor,   # [1, latent_dim]
                target_latent_tensor,   # [1, latent_dim]
            )
        
        # Reshape: generator returns [num_sets, num_samples, dim], we want [num_samples, dim]
        x1_pred = x1_pred.squeeze(0)  # [N, dim]
    
    # Compute model metric
    model_metric = compute_metric(x1_pred, x1_tensor, metric=metric)
    
    result = {
        'x1_pred': x1_pred.cpu().numpy(),
        'model_metric': model_metric,
    }
    
    if compute_baseline:
        result['baseline_metric'] = baseline_metric
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Unified evaluation for all Trellis model variants",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default=None,
        help="Path to experiment directory. If not provided, searches based on --match criteria."
    )
    parser.add_argument(
        "--match",
        type=str,
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Filter criterion in 'key=value' format. Keys use dot notation for nested access "
             "(e.g., 'experiment.split_name=replicas-1', 'seed=42'). "
             "Use 'null' or 'none' to match None values. Can be specified multiple times."
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
        "--metric",
        type=str,
        choices=["w1", "mmd_energy", "mmd_rbf", "swd"],
        default="w1",
        help="Metric to compute: 'w1' (Wasserstein-1), 'mmd_energy' (MMD with energy kernel), "
             "'mmd_rbf' (MMD with RBF kernel), or 'swd' (Sliced Wasserstein Distance)"
    )
    parser.add_argument(
        "--use_predictor",
        action="store_true",
        help="Use a predictor P to predict target latent from source latent"
    )
    parser.add_argument(
        "--predictor_type",
        type=str,
        choices=["linear", "ridge", "random_forest"],
        default="linear",
        help="Type of predictor to use: 'linear' (gradient descent), 'ridge' (sklearn exact solution), "
             "or 'random_forest' (sklearn random forest). Default: linear"
    )
    parser.add_argument(
        "--cross_validate",
        action="store_true",
        help="Use cross-validation to find optimal predictor hyperparameters (ridge_alpha)"
    )
    parser.add_argument(
        "--predict_delta",
        action="store_true",
        help="Train predictor to predict (target_latent - source_latent) instead of target_latent directly"
    )
    parser.add_argument(
        "--predictor_loss",
        type=str,
        choices=["mse", "cosine"],
        default="mse",
        help="Loss function for predictor training: 'mse' or 'cosine' (default: mse). "
             "Note: 'cosine' is only supported for 'linear' predictor type."
    )
    parser.add_argument(
        "--patient_cv",
        action="store_true",
        help="Use patient-based cross-validation instead of random k-fold. "
             "Each fold holds out samples from a single patient."
    )
    parser.add_argument(
        "--patient_holdout_fraction",
        type=float,
        default=1.0,
        help="Fraction of each patient's samples to hold out (0.0-1.0). "
             "If 1.0, all samples from the patient are held out. Default: 1.0"
    )
    parser.add_argument(
        "--folds_per_patient",
        type=int,
        default=1,
        help="Number of CV folds per patient. Only applicable when patient_holdout_fraction < 1.0. "
             "Different random subsets of the patient's samples are held out for each fold. Default: 1"
    )
    parser.add_argument(
        "--cheat_mode",
        action="store_true",
        help="CHEAT MODE: Train the predictor on BOTH training AND test data. "
             "This is useful to understand the upper bound of predictor performance "
             "(how good it could be if we had perfect training data)."
    )

    args = parser.parse_args()
    
    # Determine experiment directory
    if args.experiment_dir is None:
        if not args.match:
            parser.error("Must provide either --experiment_dir or at least one --match criterion")
        
        args.experiment_dir = find_experiment_dir(
            outputs_dir=args.outputs_dir,
            match_criteria=dict(item.split('=', 1) for item in args.match),
        )
    
    print(f"\nUsing experiment directory: {args.experiment_dir}")
    
    # Setup checkpointing
    args_hash = compute_args_hash(args)
    checkpoint_path = get_checkpoint_path(args_hash)
    checkpoint = load_checkpoint(checkpoint_path)
    print(f"Checkpoint file: {checkpoint_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experiment
    encoder, generator, dataset, cfg = load_experiment(args.experiment_dir, device)
    
    # Check model type (MFM uses zeros for target latent instead of encoder output)
    is_mfm = is_mfm_model(cfg)
    if is_mfm:
        print("Detected MFM model (zeros used for target latent)")
    
    if is_mfm and args.use_predictor:
        raise ValueError("MFM models use zeros for target latent, so predictors are not applicable.")
    
    if (args.cross_validate or args.predict_delta) and not args.use_predictor:
        print("WARNING: --cross_validate and --predict_delta have no effect without --use_predictor")
    
    if args.patient_cv and not args.cross_validate:
        print("WARNING: --patient_cv has no effect without --cross_validate")
    
    if args.patient_cv and not args.use_predictor:
        print("WARNING: --patient_cv has no effect without --use_predictor")
    
    if args.predictor_loss == "cosine" and args.predictor_type != "linear":
        print(f"WARNING: cosine loss is only supported for 'linear' predictor. "
              f"Using MSE for '{args.predictor_type}' predictor.")
    
    if args.cheat_mode:
        print("\n" + "!" * 80)
        print("! CHEAT MODE ENABLED !")
        print("! The predictor will be trained on BOTH training AND test data.")
        print("! This gives an upper bound on predictor performance, NOT realistic performance.")
        print("!" * 80)
    
    # Check if encoder uses cell_cond conditioning
    use_cell_cond = is_conditioned_encoder(cfg)
    if use_cell_cond:
        print("Detected CONDITIONED encoder (cell_cond will be concatenated with input)")
    
    test_samples = dataset.samples_test
    train_samples = dataset.samples_train
    
    # Compute latents
    print("\n" + "=" * 80)
    print("COMPUTING LATENTS")
    print("=" * 80)
    
    test_source_latents, test_target_latents, test_treat_conds = compute_latents(
        encoder, test_samples, device, split_name="test", use_cell_cond=use_cell_cond
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print(f"TRAINING {args.predictor_type.upper()} PREDICTOR")
        print("=" * 80)
        
        train_source_latents, train_target_latents, train_treat_conds = compute_latents(
            encoder, train_samples, device, split_name="train", use_cell_cond=use_cell_cond
        )
        
        # Handle cheat mode: combine training and test data
        if args.cheat_mode:
            print("\n  CHEAT MODE: Combining training and test data for predictor training")
            print(f"    Training samples: {len(train_source_latents)}")
            print(f"    Test samples: {len(test_source_latents)}")
            
            combined_source_latents = np.vstack([train_source_latents, test_source_latents])
            combined_target_latents = np.vstack([train_target_latents, test_target_latents])
            combined_treat_conds = np.vstack([train_treat_conds, test_treat_conds])
            
            print(f"    Combined samples: {len(combined_source_latents)}")
            
            # Use combined data for training
            predictor_source_latents = combined_source_latents
            predictor_target_latents = combined_target_latents
            predictor_treat_conds = combined_treat_conds
            
            # For patient-based CV in cheat mode, combine patient IDs too
            combined_patient_ids = None
            if args.patient_cv:
                train_patient_ids = np.array([sample[6] for sample in train_samples])
                test_patient_ids = np.array([sample[6] for sample in test_samples])
                combined_patient_ids = np.concatenate([train_patient_ids, test_patient_ids])
                unique_patients = np.unique(combined_patient_ids)
                print(f"    Patient-based CV enabled with {len(unique_patients)} unique patients: {unique_patients.tolist()}")
            
            predictor_patient_ids = combined_patient_ids
        else:
            # Normal mode: only use training data
            predictor_source_latents = train_source_latents
            predictor_target_latents = train_target_latents
            predictor_treat_conds = train_treat_conds
            
            # Extract patient IDs from training samples for patient-based CV
            predictor_patient_ids = None
            if args.patient_cv:
                predictor_patient_ids = np.array([sample[6] for sample in train_samples])  # patient is at index 6
                unique_patients = np.unique(predictor_patient_ids)
                print(f"  Patient-based CV enabled with {len(unique_patients)} unique patients: {unique_patients.tolist()}")
        
        # Train predictor with source latents conditioned on treatment
        predictor = train_predictor(
            predictor_source_latents, predictor_target_latents, predictor_treat_conds, 
            device=device,
            predictor_type=args.predictor_type,
            cross_validate=args.cross_validate,
            predict_delta=args.predict_delta,
            loss_type=args.predictor_loss,
            patient_ids=predictor_patient_ids,
            patient_cv=args.patient_cv,
            patient_holdout_fraction=args.patient_holdout_fraction,
            folds_per_patient=args.folds_per_patient,
        )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if is_mfm:
        print("Mode: MFM (zeros used for target latent)")
    elif args.use_predictor:
        mode_str = f"Mode: Using {args.predictor_type.upper()} predictor as target latent"
        if args.predictor_type == "linear":
            mode_str += f" (loss={args.predictor_loss})"
        if args.predict_delta:
            mode_str += " (DELTA prediction)"
        if args.cross_validate:
            mode_str += " (cross-validated)"
        if args.cheat_mode:
            mode_str += " [CHEAT MODE]"
        print(mode_str)
    else:
        print("Mode: Using oracle E(x1) as target latent")
    
    metric_name = args.metric.upper()
    print(f"Metric: {metric_name}")
    print("=" * 80)
    
    # Determine which samples still need to be computed
    completed_indices = set(checkpoint.get('completed_indices', []))
    remaining_indices = [i for i in range(len(test_samples)) if i not in completed_indices]
    
    if completed_indices:
        print(f"\nResuming from checkpoint: {len(completed_indices)} samples already done, {len(remaining_indices)} remaining")
    
    # Build running lists from checkpoint data
    all_model_metrics = []
    all_baseline_metrics = [] if args.compute_baseline else None
    
    # Add previously completed results (in order)
    for i in range(len(test_samples)):
        if i in completed_indices:
            all_model_metrics.append(checkpoint['model_metrics'][str(i)])
            if args.compute_baseline:
                all_baseline_metrics.append(checkpoint['baseline_metrics'][str(i)])
    
    for i in remaining_indices:
        sample = test_samples[i]
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
        
        source_latent = test_source_latents[i:i+1]
        target_latent = test_target_latents[i:i+1]
        sample_treat_cond = test_treat_conds[i:i+1]  # Already extracted first row during compute_latents
        
        # print(f"\nSample {i + 1}/{len(test_samples)} (computing):")
        # print(f"  Culture: {culture}, Patient: {patient}")
        # print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, sample_treat_cond, device,
            metric=args.metric,
            predictor=predictor,
            compute_baseline=args.compute_baseline,
            normalize_predicted_latent=False,  # Must match encoder's normalize_latent setting (False by default)
            is_mfm=is_mfm,
        )
        
        model_metric = results['model_metric']
        all_model_metrics.append(model_metric)
        
        # Save to checkpoint
        checkpoint['completed_indices'].append(i)
        checkpoint['model_metrics'][str(i)] = float(model_metric)
        checkpoint['sample_info'][str(i)] = {'culture': culture, 'patient': patient}
        
        if args.compute_baseline:
            baseline_metric = results['baseline_metric']
            all_baseline_metrics.append(baseline_metric)
            checkpoint['baseline_metrics'][str(i)] = float(baseline_metric)
        
        # Save checkpoint after each sample
        save_checkpoint(checkpoint_path, checkpoint)
        
        # Compute running averages (over all completed samples so far)
        # running_model_mean = np.mean(all_model_metrics)
        # 
        # if args.compute_baseline:
        #     running_baseline_mean = np.mean(all_baseline_metrics)
        #     print(f"  {metric_name:<6} Model: {model_metric:>12.6f}  Baseline: {baseline_metric:>12.6f}")
        #     print(f"  {'Avg':<6} Model: {running_model_mean:>12.6f}  Baseline: {running_baseline_mean:>12.6f}")
        # else:
        #     print(f"  {metric_name:<6} Model: {model_metric:>12.6f}")
        #     print(f"  {'Avg':<6} Model: {running_model_mean:>12.6f}")
    
    # Gather final results in proper order from checkpoint
    final_model_metrics = []
    final_baseline_metrics = [] if args.compute_baseline else None
    
    for i in range(len(test_samples)):
        if str(i) in checkpoint['model_metrics']:
            final_model_metrics.append(checkpoint['model_metrics'][str(i)])
            if args.compute_baseline:
                final_baseline_metrics.append(checkpoint['baseline_metrics'][str(i)])
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total samples evaluated: {len(final_model_metrics)}/{len(test_samples)}")
    print("=" * 80)
    
    if len(final_model_metrics) == 0:
        print("No samples evaluated yet.")
    else:
        model_mean = np.mean(final_model_metrics)
        model_std = np.std(final_model_metrics)
        model_str = f"{model_mean:.4f} +/- {model_std:.4f}"
        
        if args.compute_baseline:
            baseline_mean = np.mean(final_baseline_metrics)
            baseline_std = np.std(final_baseline_metrics)
            baseline_str = f"{baseline_mean:.4f} +/- {baseline_std:.4f}"
            print(f"{metric_name:<6} Model: {model_str:>20}  Baseline: {baseline_str:>20}")
        else:
            print(f"{metric_name:<6} Model: {model_str:>20}")


if __name__ == "__main__":
    main()