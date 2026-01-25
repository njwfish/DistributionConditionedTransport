import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from typing import Literal

from utils.latents import normalize_latent, normalize_latent_np
from utils.predictor import LinearPredictor
from utils.experiment_utils import (
    load_config,
    load_experiment,
    is_source_only_model,
    find_experiment_dir,
)
from generator.losses import wasserstein, mmd


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
    metric: Literal["w1", "mmd"] = "w1",
) -> float:
    """Compute distance metric between two distributions.
    
    Args:
        pred: Predicted samples tensor
        target: Target samples tensor
        metric: Metric to compute - "w1" (Wasserstein-1) or "mmd" (energy MMD)
    
    Returns:
        Computed metric value
    """
    if metric == "w1":
        return wasserstein(pred, target, p=1)
    elif metric == "mmd":
        return mmd(pred, target, kernel='energy').item()
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


def train_linear_predictor(
    source_latents: np.ndarray,
    target_latents: np.ndarray,
    treat_conds: np.ndarray,
    device: torch.device,
    ridge_alpha: float = 1e-3,
    num_epochs: int = 1000,
    lr: float = 1e-2,
    verbose: bool = True,
) -> LinearPredictor:
    """
    Train a linear predictor to map source latents (conditioned on treatment) to target latents.
    
    The predictor is always conditioned on the treatment condition by concatenating
    the one-hot encoded treatment vector with the source latents.
    
    Args:
        source_latents: Source latents from encoder, shape (N, latent_dim)
        target_latents: Target latents from encoder, shape (N, latent_dim)
        treat_conds: Treatment conditions (one-hot), shape (N, num_treatments)
        device: Device for training
        ridge_alpha: Ridge regularization coefficient
        num_epochs: Number of training epochs
        lr: Learning rate
        verbose: Whether to print training progress
    
    Returns:
        Trained LinearPredictor
    """
    # Concatenate source latents with treatment condition
    # This conditions the predictor on the treatment
    source_latents_conditioned = np.concatenate([source_latents, treat_conds], axis=1)
    
    print(f"Training linear predictor...")
    print(f"  Source latents shape: {source_latents.shape}")
    print(f"  Treatment conditions shape: {treat_conds.shape}")
    print(f"  Conditioned input shape: {source_latents_conditioned.shape} -> {target_latents.shape}")
    print(f"  ridge_alpha={ridge_alpha}, lr={lr}, epochs={num_epochs}")
    
    input_dim = source_latents_conditioned.shape[1]
    output_dim = target_latents.shape[1]
    
    predictor = LinearPredictor(input_dim, output_dim)
    predictor.fit(
        source_latents_conditioned,
        target_latents,
        loss_type="cosine",
        ridge_alpha=ridge_alpha,
        num_epochs=num_epochs,
        lr=lr,
        device=device,
        verbose=verbose,
    )
    
    return predictor


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    treat_cond: np.ndarray,
    device: torch.device,
    metric: Literal["w1", "mmd"] = "w1",
    predictor = None,
    compute_baseline: bool = False,
    normalize_predicted_latent: bool = True,
    is_source_only: bool = False,
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
        metric: Metric to compute ("w1" or "mmd")
        predictor: Optional predictor to predict target latent from source latent + treat_cond
        compute_baseline: Whether to compute baseline metric (x0 vs x1)
        normalize_predicted_latent: Whether to normalize predicted latent
        is_source_only: Whether the model is source-only (MFM variant)
    
    Returns:
        Dictionary with 'x1_pred', 'model_metric', and optionally 'baseline_metric'
    """
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    baseline_metric = None
    if compute_baseline:
        baseline_metric = compute_metric(x0_tensor, x1_tensor, metric=metric)
    
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    if is_source_only:
        target_latent_tensor = None
    elif predictor is not None:
        # Concatenate source latent with treatment condition for the predictor
        # The predictor was trained with this conditioning
        source_latent_conditioned = np.concatenate([source_latent, treat_cond], axis=1)
        predicted_target_latent = predictor.predict(source_latent_conditioned)
        if normalize_predicted_latent:
            predicted_target_latent = normalize_latent_np(predicted_target_latent)
        target_latent_tensor = torch.tensor(predicted_target_latent, dtype=torch.float32, device=device)
    else:
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
        choices=["w1", "mmd"],
        default="w1",
        help="Metric to compute: 'w1' (Wasserstein-1) or 'mmd' (energy MMD)"
    )
    parser.add_argument(
        "--use_predictor",
        action="store_true",
        help="Use a predictor P to predict target latent from source latent"
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load experiment
    encoder, generator, dataset, cfg = load_experiment(args.experiment_dir, device)
    
    # Check model type
    is_source_only = is_source_only_model(cfg)
    if is_source_only:
        print("Detected SOURCE-ONLY model (no target latent used)")
    
    if is_source_only and args.use_predictor:
        raise ValueError("Source-only models do not use target latent, so predictors are not applicable.")
    
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
        print("TRAINING LINEAR PREDICTOR")
        print("=" * 80)
        
        train_source_latents, train_target_latents, train_treat_conds = compute_latents(
            encoder, train_samples, device, split_name="train", use_cell_cond=use_cell_cond
        )
        
        # Train predictor with source latents conditioned on treatment
        predictor = train_linear_predictor(
            train_source_latents, train_target_latents, train_treat_conds, device=device
        )
    
    # Evaluate
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    if is_source_only:
        print("Mode: SOURCE-ONLY (no target latent used)")
    elif args.use_predictor:
        print("Mode: Using LINEAR predictor as target latent")
    else:
        print("Mode: Using oracle E(x1) as target latent")
    
    metric_name = args.metric.upper()
    print(f"Metric: {metric_name}")
    print("=" * 80)
    
    all_model_metrics = []
    all_baseline_metrics = [] if args.compute_baseline else None
    
    for i, sample in enumerate(test_samples):
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
        
        source_latent = test_source_latents[i:i+1]
        target_latent = test_target_latents[i:i+1]
        sample_treat_cond = test_treat_conds[i:i+1]  # Already extracted first row during compute_latents
        
        print(f"\nSample {i + 1}/{len(test_samples)}:")
        print(f"  Culture: {culture}, Patient: {patient}")
        print(f"  x0 shape: {x0.shape}, x1 shape: {x1.shape}")
        
        # Evaluate
        results = evaluate_sample(
            generator, x0, x1, source_latent, target_latent, sample_treat_cond, device,
            metric=args.metric,
            predictor=predictor,
            compute_baseline=args.compute_baseline,
            normalize_predicted_latent=True,
            is_source_only=is_source_only,
        )
        
        model_metric = results['model_metric']
        all_model_metrics.append(model_metric)
        
        if args.compute_baseline:
            baseline_metric = results['baseline_metric']
            all_baseline_metrics.append(baseline_metric)
            print(f"  {metric_name:<6} Model: {model_metric:>12.6f}  Baseline: {baseline_metric:>12.6f}")
        else:
            print(f"  {metric_name:<6} Model: {model_metric:>12.6f}")
    
    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL RESULTS (mean +/- std)")
    print("=" * 80)
    
    model_mean = np.mean(all_model_metrics)
    model_std = np.std(all_model_metrics)
    model_str = f"{model_mean:.4f} +/- {model_std:.4f}"
    
    if args.compute_baseline:
        baseline_mean = np.mean(all_baseline_metrics)
        baseline_std = np.std(all_baseline_metrics)
        baseline_str = f"{baseline_mean:.4f} +/- {baseline_std:.4f}"
        print(f"{metric_name:<6} Model: {model_str:>20}  Baseline: {baseline_str:>20}")
    else:
        print(f"{metric_name:<6} Model: {model_str:>20}")


if __name__ == "__main__":
    main()
