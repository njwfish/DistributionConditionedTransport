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
from sklearn.linear_model import Ridge

from utils.latents import normalize_latent, normalize_latent_np


# TODO: this predictor model stuff needs to be moved and unified across the codebase.
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


# TODO: metrics should also be moved into a utils file. Maybe find where the losses are usually.
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

def compute_all_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    max_samples_w1: Optional[int] = None,
):
    """Compute W1 distance between two distributions."""
    if max_samples_w1 is not None:
        if pred.shape[0] > max_samples_w1:
            pred = pred[torch.randperm(pred.shape[0])[:max_samples_w1]]
        if target.shape[0] > max_samples_w1:
            target = target[torch.randperm(target.shape[0])[:max_samples_w1]]
    
    return {'W1': wasserstein(pred, target, power=1)}

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
) -> tuple:
    """Compute E(x0) and E(x1) for all samples. Returns (source_latents, target_latents, treat_conds)."""
    print(f"Computing {split_name} latents for {len(samples)} samples...")
    
    source_latents = []
    target_latents = []
    treat_conds = []
    
    encoder.eval()
    with torch.no_grad():
        for i, sample in enumerate(samples):
            culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
            
            x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)
            x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)
            
            source_latents.append(encoder(x0_tensor).cpu().numpy())
            target_latents.append(encoder(x1_tensor).cpu().numpy())
            treat_conds.append(treat_cond[0:1])
            
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(samples)} samples")
    
    source_latents = np.vstack(source_latents)
    target_latents = np.vstack(target_latents)
    treat_conds = np.vstack(treat_conds)
    
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
    
    print(f"  Final MSE: {final_loss:.6f}")
    
    return predictor


# TODO: config loading should also be done in a separate file to unify across evals
def load_experiment(
    experiment_dir: str, 
    device: torch.device, 
    load_train_data: bool = False,
    is_a2a: bool = False,
):
    """
    Load the trained model, config, and instantiate the components.
    
    Returns:
        encoder, generator, test_samples, config, train_samples (or None), is_source_only
    """
    # Load config
    config_path = os.path.join(experiment_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    print(f"Loaded config from {config_path}")
    
    # Load checkpoint
    checkpoint_path = os.path.join(experiment_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Resolve config references
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    resolved_cfg = OmegaConf.create(resolved_cfg)
    
    # Detect source-only model (MFM variant that doesn't use target latent)
    model_source_only = resolved_cfg.get("model", {}).get("source_only", False)
    loss_target = resolved_cfg.get("loss", {}).get("_target_", "")
    is_source_only = model_source_only or "source_only" in loss_target.lower()
    
    if is_source_only:
        print(f"Detected SOURCE-ONLY model (no target latent used)")
    
    # Instantiate datasets
    train_samples = None
    test_samples = None
    

    dataset = hydra.utils.instantiate(resolved_cfg.dataset)
    train_samples = dataset.samples_train 
    test_samples = dataset.samples_test

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
    
    return encoder, generator, test_samples, cfg, train_samples, is_source_only


def evaluate_sample(
    generator: torch.nn.Module,
    x0: np.ndarray,
    x1: np.ndarray,
    source_latent: np.ndarray,
    target_latent: np.ndarray,
    device: torch.device,
    predictor = None,
    compute_baseline: bool = False,
    max_samples_w1: Optional[int] = None,
    normalize_predicted_latent: bool = True,
    is_source_only: bool = False,
):
    """Evaluate the model on a single sample using precomputed latents."""
    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=device)
    x1_tensor = torch.tensor(x1, dtype=torch.float32, device=device)
    
    baseline_metrics = None
    if compute_baseline:
        baseline_metrics = compute_all_metrics(x0_tensor, x1_tensor, max_samples_w1=max_samples_w1)
    
    source_latent_tensor = torch.tensor(source_latent, dtype=torch.float32, device=device)
    
    if is_source_only:
        target_latent_tensor = None
    elif predictor is not None:
        # predictor is either Ridge (sklearn) or MLPPredictor - both have .predict()
        predicted_target_latent = predictor.predict(source_latent)
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
    
    # Compute model metrics
    model_metrics = compute_all_metrics(x1_pred, x1_tensor, max_samples_w1=max_samples_w1)
    
    result = {
        'x1_pred': x1_pred.cpu().numpy(),
        'model': model_metrics,
    }
    
    if compute_baseline:
        result['baseline'] = baseline_metrics
    
    return result

# TODO: this should also be a utils function.
def find_experiment_dir(
    outputs_dir: str = "outputs",
    match_criteria: Optional[dict] = None,
) -> str:
    """
    Search through directories in outputs_dir to find the experiment matching the given criteria.
    Keys use dot notation for nested access (e.g., 'experiment.split_name', 'seed').
    Use 'null' as value to match None.
    """
    print(f"Searching for experiment with: {match_criteria}")
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
            for key_path, target_value in match_criteria.items():
                cfg_value = OmegaConf.select(cfg, key_path)
                # Handle null/none string to match None
                if target_value.lower() in ("null", "none"):
                    if cfg_value is not None:
                        match = False
                        break
                elif str(cfg_value) != target_value:
                    match = False
                    break
            
            if match:
                matching_dirs.append(dir_path)
                print(f"  Found match: {dirname}")
                    
        except Exception as e:
            continue
    
    if len(matching_dirs) == 0:
        raise ValueError(f"No experiment found matching: {match_criteria}")
    
    if len(matching_dirs) > 1:
        print(f"Warning: Multiple matching directories found, using the first one:")
        for d in matching_dirs:
            print(f"  - {d}")
    
    return matching_dirs[0]


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
        choices=["ridge", "mlp"],
        default="ridge",
        help="Type of predictor: 'ridge' or 'mlp'"
    )
    parser.add_argument(
        "--no_normalize_predicted_latent",
        action="store_true",
        help="Disable normalization of predicted latent before passing to generator"
    )
    parser.add_argument(
        "--a2a",
        action="store_true",
        help="Whether this is an a2a-style dataset (has internal train/test split)"
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
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # First load experiment without training data to check if source-only
    encoder, generator, test_samples, cfg, _, is_source_only = load_experiment(
        args.experiment_dir, 
        device, 
        load_train_data=False,
        is_a2a=args.a2a,
    )
    
    if is_source_only and args.use_predictor:
        raise ValueError("Source-only models do not use target latent, so predictors are not applicable.")
    
    # Now load training data if needed for predictor
    train_samples = None
    if args.use_predictor:
        _, _, _, _, train_samples, _ = load_experiment(
            args.experiment_dir, 
            device, 
            load_train_data=True,
            is_a2a=args.a2a,
        )
    
    # Compute and cache latents
    print("\n" + "=" * 80)
    print("COMPUTING/LOADING LATENTS")
    print("=" * 80)
    
    test_cache_path = get_latent_cache_path(args.experiment_dir, "test")
    test_source_latents, test_target_latents, test_treat_conds = compute_and_cache_latents(
        encoder, test_samples, device, test_cache_path, split_name="test"
    )
    
    # Train predictor if requested
    predictor = None
    if args.use_predictor:
        print("\n" + "=" * 80)
        print(f"TRAINING PREDICTOR ({args.predictor_type.upper()})")
        print("=" * 80)
        
        train_cache_path = get_latent_cache_path(args.experiment_dir, "train")
        train_source_latents, train_target_latents, _ = compute_and_cache_latents(
            encoder, train_samples, device, train_cache_path, split_name="train"
        )
        
        if args.predictor_type == "ridge":
            predictor = train_ridge_predictor(train_source_latents, train_target_latents)
        else:  # mlp
            predictor = train_mlp_predictor(train_source_latents, train_target_latents, device=device)
    
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

    max_samples_w1 = args.max_samples_w1 if args.max_samples_w1 > 0 else None
    if max_samples_w1 is not None:
        print(f"W1 subsampling: max {max_samples_w1} samples from source/target")
    print("=" * 80)
    
    metric_names = ['W1']
    
    all_model_metrics = {name: [] for name in metric_names}
    all_baseline_metrics = {name: [] for name in metric_names} if args.compute_baseline else None
    
    for i, sample in enumerate(test_samples):
        culture, x0, x1, cell_cond_source, cell_cond_target, treat_cond, patient = sample
        
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
            compute_baseline=args.compute_baseline,
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
