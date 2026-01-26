"""
Unified evaluation for supervised vs semisupervised distribution transport.

Evaluates both approaches on test distributions across the full support [0, 5],
showing how performance varies with the mean location (mu). This reveals:
- How supervised models (source_only) generalize outside training support [0, 2.5]
- How semisupervised models (any-to-any + ridge predictor) perform across the range

Produces visualizations plotting performance metrics vs mu, with clear comparison
between supervised, semisupervised, and oracle (upper bound) approaches.
"""

import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import hydra
import pickle
import pandas as pd
from tqdm import tqdm
import copy
import gc
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.experiment_utils import get_all_experiments_info, load_checkpoint
from generator.losses import mmd, sliced_wasserstein_distance


# ============================================================================
# Metric functions
# ============================================================================

def compute_energy_distance(x, y):
    """Compute energy distance between two sample sets."""
    n, m = x.shape[0], y.shape[0]
    xy_dist = torch.cdist(x, y, p=2).mean()
    xx_dist = torch.cdist(x, x, p=2).sum() / (n * (n - 1)) if n > 1 else 0
    yy_dist = torch.cdist(y, y, p=2).sum() / (m * (m - 1)) if m > 1 else 0
    return 2 * xy_dist - xx_dist - yy_dist


def compute_metrics(generated, target, device='cpu'):
    """Compute distributional metrics between generated and target samples."""
    gen_t = generated if isinstance(generated, torch.Tensor) else torch.from_numpy(generated).float()
    tgt_t = target if isinstance(target, torch.Tensor) else torch.from_numpy(target).float()
    gen_t, tgt_t = gen_t.to(device), tgt_t.to(device)

    return {
        'mmd': mmd(gen_t, tgt_t).item(),
        'swd': sliced_wasserstein_distance(gen_t, tgt_t, n_projections=100, p=2).item(),
        'energy': compute_energy_distance(gen_t, tgt_t).item(),
    }


# ============================================================================
# Model loading
# ============================================================================

def load_model(cfg, path, device, num_epochs):
    """Load encoder and generator from specific epoch checkpoint."""
    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])

    state = load_checkpoint(path, num_epochs)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])

    enc.eval().to(device)
    gen.eval().to(device)

    return enc, gen


def filter_experiments(configs, experiment_type='mvn', num_epochs=200,
                       n_unique_sets=None, supervised=None, require_checkpoint=True):
    """Filter experiments by criteria.

    Args:
        configs: List of experiment configs
        experiment_type: Type of experiment ('mvn', 'gmm', etc)
        num_epochs: Target epoch for checkpoint loading
        n_unique_sets: Required n_unique_sets value
        supervised: If True, only supervised; if False, only unsupervised; if None, both
        require_checkpoint: If True, filter to experiments that have the checkpoint for num_epochs
    """
    import os
    filtered = []
    for c in configs:
        if experiment_type not in c['name']:
            continue
        if supervised is not None:
            is_supervised = 'supervised' in c['name']
            if is_supervised != supervised:
                continue
        # Check if checkpoint exists for the target epoch
        if require_checkpoint:
            checkpoint_path = os.path.join(c['dir'], f'checkpoint_epoch_{num_epochs}.pt')
            if not os.path.exists(checkpoint_path):
                continue
        if n_unique_sets is not None:
            cfg_n_unique = c['config']['dataset'].get('n_unique_sets')
            if cfg_n_unique != n_unique_sets:
                continue
        # Must be distribution encoder (GNN), not embedding
        if 'Embedding' in c.get('encoder', ''):
            continue
        filtered.append(c)
    return filtered


# ============================================================================
# Test data generation with controlled mu
# ============================================================================

def generate_test_distributions_at_mu(mu_value, n_distributions, data_dim=2,
                                       prior_cov_df=10, prior_cov_scale=1.0,
                                       set_size=100, seed=None):
    """
    Generate test distributions with means centered at a specific mu value.

    For MVN: generates distributions with mean uniformly in [mu - 0.25, mu + 0.25]
    This allows testing at specific points along the [0, 5] range.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate means in a small window around mu_value
    delta = 0.25  # Half-width of window
    mu = np.random.uniform(mu_value - delta, mu_value + delta, (n_distributions, data_dim))

    # Clamp to valid range
    mu = np.clip(mu, 0, 5)

    # Generate covariances from inverse Wishart
    prior_cov_df_adjusted = prior_cov_df * (data_dim - 1)
    cov = np.linalg.inv(sp.stats.wishart.rvs(
        df=prior_cov_df_adjusted,
        scale=prior_cov_scale * np.eye(data_dim),
        size=n_distributions
    ))

    # Sample data
    data = np.zeros((n_distributions, set_size, data_dim))
    for i in range(n_distributions):
        L = np.linalg.cholesky(cov[i])
        z = np.random.randn(set_size, data_dim)
        data[i] = z @ L.T + mu[i]

    return data, mu, cov


def generate_supervised_test_data(source_data, source_mu, n_distributions,
                                   shift=(1.0, 1.0), data_dim=2, seed=None):
    """
    Generate supervised target data: Y = X + shift (shuffled).
    Uses a fixed shift transformation matching training.
    """
    if seed is not None:
        np.random.seed(seed)

    # Fixed shift transformation: A = I, b = shift
    A = np.eye(data_dim)
    b = np.array(shift)

    target_data = np.zeros_like(source_data)
    for i in range(n_distributions):
        target_data[i] = source_data[i] + b
        np.random.shuffle(target_data[i])

    return target_data, A, b


# ============================================================================
# Transport functions
# ============================================================================

def transport_supervised(enc, gen, source_samples, device):
    """
    Transport using supervised model (source_only conditioning).
    Target latent is zeros.

    Args:
        source_samples: (set_size, dim) single distribution samples
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)

        # Add batch dimension for encoder: (1, set_size, dim)
        source_batch = source_t.unsqueeze(0)

        # Get source embedding: (1, latent_dim)
        source_lat = enc(source_batch)

        # Target latent is zeros (source_only): (1, latent_dim)
        target_lat = torch.zeros_like(source_lat)

        set_size, dim = source_t.shape

        # Expand latents for each sample in the set
        source_lat_exp = source_lat.expand(set_size, -1)  # (set_size, latent_dim)
        target_lat_exp = target_lat.expand(set_size, -1)  # (set_size, latent_dim)

        # Transport: source_t is (set_size, dim)
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)

        # Squeeze extra dimension if present (flow matching returns (N, 1, d))
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)  # Handle 1D case

        return transported.cpu().numpy()


def transport_semisupervised(enc, gen, predictor, source_samples, device):
    """
    Transport using semisupervised approach: any-to-any model + ridge predictor.

    The predictor predicts the RESIDUAL: target_lat = source_lat + predictor(source_lat)

    Args:
        source_samples: (set_size, dim) single distribution samples
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)

        # Add batch dimension for encoder: (1, set_size, dim)
        source_batch = source_t.unsqueeze(0)

        # Get source embedding: (1, latent_dim)
        source_lat = enc(source_batch)

        # Predict target latent using residual ridge regression:
        # target_lat = source_lat + predictor(source_lat)
        source_lat_np = source_lat.cpu().numpy()
        residual_np = predictor.predict(source_lat_np)
        target_lat_np = source_lat_np + residual_np
        target_lat = torch.from_numpy(target_lat_np).float().to(device)

        set_size, dim = source_t.shape

        # Expand latents for each sample in the set
        source_lat_exp = source_lat.expand(set_size, -1)  # (set_size, latent_dim)
        target_lat_exp = target_lat.expand(set_size, -1)  # (set_size, latent_dim)

        # Transport: source_t is (set_size, dim)
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)

        # Squeeze extra dimension if present (flow matching returns (N, 1, d))
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)  # Handle 1D case

        return transported.cpu().numpy()


def transport_oracle(enc, gen, source_samples, target_samples, device):
    """
    Transport using oracle (true target embedding). Upper bound on performance.

    Args:
        source_samples: (set_size, dim) single distribution samples
        target_samples: (set_size, dim) single distribution samples
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)
        target_t = torch.from_numpy(target_samples).float().to(device)

        # Add batch dimension for encoder: (1, set_size, dim)
        source_batch = source_t.unsqueeze(0)
        target_batch = target_t.unsqueeze(0)

        # Get embeddings: (1, latent_dim)
        source_lat = enc(source_batch)
        target_lat = enc(target_batch)

        set_size, dim = source_t.shape

        # Expand latents for each sample in the set
        source_lat_exp = source_lat.expand(set_size, -1)  # (set_size, latent_dim)
        target_lat_exp = target_lat.expand(set_size, -1)  # (set_size, latent_dim)

        # Transport: source_t is (set_size, dim)
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)

        # Squeeze extra dimension if present (flow matching returns (N, 1, d))
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)  # Handle 1D case

        return transported.cpu().numpy()


# ============================================================================
# Evaluation at specific mu values
# ============================================================================

def evaluate_at_mu(mu_value, enc, gen, device, method='supervised', predictor=None,
                   n_distributions=100, set_size=100, data_dim=2, seed=None,
                   shift=(1.0, 1.0)):
    """
    Evaluate transport performance at a specific mu value.

    Args:
        mu_value: Center of the mu range to test
        enc, gen: Encoder and generator models
        device: Device to use
        method: 'supervised', 'semisupervised', or 'oracle'
        predictor: Ridge predictor (required for semisupervised)
        n_distributions: Number of test distributions
        set_size: Samples per distribution
        data_dim: Data dimensionality
        seed: Random seed for source data generation (can vary per mu)
        shift: Fixed shift vector for transformation Y = X + shift

    Returns:
        dict with mean and std of each metric
    """
    # Generate test data at this mu
    source_data, source_mu, source_cov = generate_test_distributions_at_mu(
        mu_value, n_distributions, data_dim, set_size=set_size, seed=seed
    )

    # Generate target data using fixed shift transformation
    target_data, A, b = generate_supervised_test_data(
        source_data, source_mu, n_distributions, shift=shift, data_dim=data_dim, seed=seed
    )

    mmd_vals, swd_vals, energy_vals = [], [], []

    for i in range(n_distributions):
        source = source_data[i]
        target = target_data[i]

        if method == 'supervised':
            transported = transport_supervised(enc, gen, source, device)
        elif method == 'semisupervised':
            transported = transport_semisupervised(enc, gen, predictor, source, device)
        elif method == 'oracle':
            transported = transport_oracle(enc, gen, source, target, device)
        else:
            raise ValueError(f"Unknown method: {method}")

        metrics = compute_metrics(transported, target, device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])

    return {
        'mu': mu_value,
        'mmd_mean': np.mean(mmd_vals),
        'mmd_std': np.std(mmd_vals),
        'swd_mean': np.mean(swd_vals),
        'swd_std': np.std(swd_vals),
        'energy_mean': np.mean(energy_vals),
        'energy_std': np.std(energy_vals),
    }


def fit_predictor_from_data(enc, device, n_train=10000, set_size=100,
                             data_dim=2, alpha=1.0, seed=42, predictor_type='ridge',
                             shift=(1.0, 1.0)):
    """
    Fit predictor using supervised data with restricted support [0, 2.5].

    Predicts the RESIDUAL (target_emb - source_emb) rather than target_emb directly.
    This can be easier to learn since the transformation is often roughly identity + small delta.
    """
    print(f"  Generating {n_train} supervised training samples (restricted support)...")
    np.random.seed(seed)

    # Generate source data with restricted support
    source_data, source_mu, source_cov = generate_test_distributions_at_mu(
        1.25, n_train, data_dim, set_size=set_size, seed=seed  # Center of [0, 2.5]
    )
    # Regenerate with full restricted range
    source_mu = np.random.uniform(0, 2.5, (n_train, data_dim))
    prior_cov_df_adjusted = 10 * (data_dim - 1)
    source_cov = np.linalg.inv(sp.stats.wishart.rvs(
        df=prior_cov_df_adjusted, scale=1.0 * np.eye(data_dim), size=n_train
    ))
    source_data = np.zeros((n_train, set_size, data_dim))
    for i in range(n_train):
        L = np.linalg.cholesky(source_cov[i])
        z = np.random.randn(set_size, data_dim)
        source_data[i] = z @ L.T + source_mu[i]

    target_data, A, b = generate_supervised_test_data(
        source_data, source_mu, n_train, shift=shift, data_dim=data_dim, seed=seed
    )

    print("  Embedding training data...")
    source_latents = []
    target_latents = []

    batch_size = 256
    with torch.no_grad():
        for i in tqdm(range(0, n_train, batch_size), desc="  Embedding"):
            batch_end = min(i + batch_size, n_train)
            src_batch = torch.from_numpy(source_data[i:batch_end]).float().to(device)
            tgt_batch = torch.from_numpy(target_data[i:batch_end]).float().to(device)

            source_latents.append(enc(src_batch).cpu().numpy())
            target_latents.append(enc(tgt_batch).cpu().numpy())

    source_latents = np.concatenate(source_latents, axis=0)
    target_latents = np.concatenate(target_latents, axis=0)

    # Compute residuals: delta = target - source
    residuals = target_latents - source_latents

    if predictor_type == 'ridge':
        print(f"  Fitting ridge predictor for RESIDUAL (alpha={alpha})...")
        predictor = Ridge(alpha=alpha)
        predictor.fit(source_latents, residuals)
    elif predictor_type == 'mlp':
        print(f"  Fitting MLP predictor for RESIDUAL...")
        predictor = MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            verbose=False,
            random_state=seed
        )
        predictor.fit(source_latents, residuals)
    else:
        raise ValueError(f"Unknown predictor_type: {predictor_type}")

    # Evaluate: predicted_target = source + predicted_residual
    pred_residuals = predictor.predict(source_latents)
    pred_targets = source_latents + pred_residuals
    train_mse = np.mean((pred_targets - target_latents) ** 2)
    residual_mse = np.mean((pred_residuals - residuals) ** 2)
    print(f"  Residual MSE: {residual_mse:.6f}, Full prediction MSE: {train_mse:.6f}")

    # Also compute baseline: just use mean residual (constant offset)
    mean_residual = np.mean(residuals, axis=0)
    const_pred_targets = source_latents + mean_residual
    const_mse = np.mean((const_pred_targets - target_latents) ** 2)
    print(f"  Constant offset MSE: {const_mse:.6f}")

    return predictor, train_mse


# ============================================================================
# Visualization
# ============================================================================

def plot_performance_vs_mu(results_df, save_path, metric='mmd', title=None):
    """
    Plot performance metric vs mu for different methods.

    Args:
        results_df: DataFrame with columns [mu, method, {metric}_mean, {metric}_std, ...]
        save_path: Path to save figure
        metric: Which metric to plot ('mmd', 'swd', 'energy')
        title: Optional title
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'supervised': '#e41a1c',      # Red
        'semisupervised': '#377eb8',  # Blue
        'oracle': '#4daf4a',          # Green
    }
    labels = {
        'supervised': 'Supervised (source-only)',
        'semisupervised': 'Semisupervised (any-to-any + ridge)',
        'oracle': 'Oracle (true target embedding)',
    }

    # Add shaded region for training support
    ax.axvspan(0, 2.5, alpha=0.15, color='gray', label='Training support [0, 2.5]')
    ax.axvline(x=2.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    for method in ['supervised', 'semisupervised', 'oracle']:
        method_df = results_df[results_df['method'] == method].sort_values('mu')
        if len(method_df) == 0:
            continue

        mu = method_df['mu'].values
        mean = method_df[f'{metric}_mean'].values
        std = method_df[f'{metric}_std'].values

        ax.plot(mu, mean, color=colors[method], linewidth=2, label=labels[method])
        ax.fill_between(mu, mean - std, mean + std, color=colors[method], alpha=0.2)

    ax.set_xlabel('Mean location (μ)', fontsize=12)
    metric_labels = {'mmd': 'MMD', 'swd': 'Sliced Wasserstein Distance', 'energy': 'Energy Distance'}
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_metrics(results_df, save_dir, generator_name):
    """Plot all metrics for a given generator."""
    for metric in ['mmd', 'swd', 'energy']:
        plot_performance_vs_mu(
            results_df,
            f"{save_dir}/{generator_name}_{metric}_vs_mu.png",
            metric=metric,
            title=f"{generator_name.replace('_', ' ').title()}: {metric.upper()} vs μ"
        )


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_model_pair(supervised_exp, unsupervised_exp, device, args, model_seed):
    """
    Evaluate supervised vs semisupervised for a matched pair of models.

    Args:
        supervised_exp: Supervised experiment config (or None)
        unsupervised_exp: Unsupervised (any-to-any) experiment config
        device: Device to use
        args: Command line arguments
        model_seed: The training seed used for this model

    Returns:
        DataFrame with results at each mu value
    """
    results = []

    # Fixed shift transformation (same for all models)
    shift = (1.0, 1.0)
    print(f"Using fixed shift transformation: Y = X + {shift}")

    # Load unsupervised model (used for semisupervised and oracle)
    print("Loading unsupervised (any-to-any) model...")
    enc_unsup, gen_unsup = load_model(unsupervised_exp['config'], unsupervised_exp['dir'], device, args.num_epochs)

    # Fit predictor for semisupervised
    predictor, train_mse = fit_predictor_from_data(
        enc_unsup, device, n_train=args.n_train, set_size=args.set_size,
        data_dim=args.data_dim, alpha=args.ridge_alpha, seed=model_seed,
        predictor_type=args.predictor_type, shift=shift
    )

    # Load supervised model if available
    if supervised_exp is not None:
        print("Loading supervised model...")
        enc_sup, gen_sup = load_model(supervised_exp['config'], supervised_exp['dir'], device, args.num_epochs)
    else:
        enc_sup, gen_sup = None, None

    # Evaluate at each mu value
    mu_values = np.linspace(0.5, 4.5, args.n_mu_points)

    # Use eval_set_size for evaluation (can be much larger than training set_size)
    eval_set_size = args.eval_set_size
    print(f"Using eval_set_size={eval_set_size} for evaluation (predictor trained with set_size={args.set_size})")

    for mu in tqdm(mu_values, desc="Evaluating across μ"):
        # Supervised
        if enc_sup is not None:
            res = evaluate_at_mu(
                mu, enc_sup, gen_sup, device, method='supervised',
                n_distributions=args.n_distributions, set_size=eval_set_size,
                data_dim=args.data_dim, seed=int(mu * 1000),
                shift=shift
            )
            res['method'] = 'supervised'
            results.append(res)

        # Semisupervised
        res = evaluate_at_mu(
            mu, enc_unsup, gen_unsup, device, method='semisupervised',
            predictor=predictor, n_distributions=args.n_distributions,
            set_size=eval_set_size, data_dim=args.data_dim, seed=int(mu * 1000),
            shift=shift
        )
        res['method'] = 'semisupervised'
        results.append(res)

        # Oracle
        res = evaluate_at_mu(
            mu, enc_unsup, gen_unsup, device, method='oracle',
            n_distributions=args.n_distributions, set_size=eval_set_size,
            data_dim=args.data_dim, seed=int(mu * 1000),
            shift=shift
        )
        res['method'] = 'oracle'
        results.append(res)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--experiment', type=str, default='mvn', choices=['mvn', 'gmm'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./supervised_comparison_results')
    parser.add_argument('--n_train', type=int, default=10000, help='Supervised training samples for ridge')
    parser.add_argument('--n_distributions', type=int, default=100, help='Test distributions per mu')
    parser.add_argument('--n_mu_points', type=int, default=20, help='Number of mu values to test')
    parser.add_argument('--set_size', type=int, default=100, help='Samples per distribution for predictor training (should match supervised model)')
    parser.add_argument('--eval_set_size', type=int, default=10000, help='Samples per distribution for evaluation (can be larger)')
    parser.add_argument('--data_dim', type=int, default=2, help='Data dimensionality')
    parser.add_argument('--ridge_alpha', type=float, default=1.0, help='Ridge regularization')
    parser.add_argument('--predictor_type', type=str, default='ridge', choices=['ridge', 'mlp'],
                        help='Type of predictor: ridge or mlp')
    parser.add_argument('--generator', type=str, default=None, help='Specific generator to evaluate')
    parser.add_argument('--seed', type=int, default=None, help='Specific seed to evaluate')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create save directory
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load all experiment configs
    print("Loading experiment configs...")
    configs = get_all_experiments_info(args.output_dir)

    # Find supervised experiments
    supervised_exps = filter_experiments(
        configs, args.experiment, args.num_epochs, supervised=True
    )
    print(f"Found {len(supervised_exps)} supervised {args.experiment} experiments")

    # Find unsupervised (any-to-any) experiments with n_unique_sets=10000
    unsupervised_exps = filter_experiments(
        configs, args.experiment, args.num_epochs, n_unique_sets=10000, supervised=False
    )
    print(f"Found {len(unsupervised_exps)} unsupervised {args.experiment} experiments (n_unique_sets=10000)")

    if not unsupervised_exps:
        print("No unsupervised experiments found!")
        return

    # Helper to extract actual generator type
    def get_generator_type(exp):
        gen_cfg = exp['config']['generator']
        gen_target = gen_cfg.get('_target_', '')
        loss_type = gen_cfg.get('loss_type', '')

        if 'FlowMatching' in gen_target:
            return 'flow_matching'
        elif loss_type:
            return loss_type
        else:
            return gen_target.split('.')[-1]

    # Group by generator and seed
    all_results = []

    for unsup_exp in unsupervised_exps:
        gen_type = get_generator_type(unsup_exp)
        seed = unsup_exp['config'].get('seed', 42)

        # Filter by generator/seed if specified
        if args.generator and gen_type != args.generator:
            continue
        if args.seed is not None and seed != args.seed:
            continue

        print(f"\n{'='*60}")
        print(f"Generator: {gen_type}, Seed: {seed}")
        print(f"{'='*60}")

        # Find matching supervised experiment
        sup_exp = None
        for s in supervised_exps:
            s_gen_type = get_generator_type(s)
            if s_gen_type == gen_type and s['config'].get('seed', 42) == seed:
                # Also check n_unique_sets matches
                sup_n_unique = s['config']['dataset'].get('n_unique_sets', 10000)
                if sup_n_unique == 10000:
                    sup_exp = s
                    break

        if sup_exp:
            print(f"Found matching supervised experiment")
        else:
            print(f"No matching supervised experiment found, will only evaluate semisupervised + oracle")

        try:
            results_df = evaluate_model_pair(sup_exp, unsup_exp, device, args, model_seed=seed)
            results_df['generator'] = gen_type
            results_df['seed'] = seed

            # Generate plots
            plot_all_metrics(results_df, args.save_dir, f"{gen_type}_seed{seed}")

            all_results.append(results_df)

            # Print summary
            print("\nSummary:")
            in_support = results_df[results_df['mu'] <= 2.5]
            out_support = results_df[results_df['mu'] > 2.5]

            for method in results_df['method'].unique():
                in_mmd = in_support[in_support['method'] == method]['mmd_mean'].mean()
                out_mmd = out_support[out_support['method'] == method]['mmd_mean'].mean()
                print(f"  {method:20s}: in-support MMD={in_mmd:.4f}, out-support MMD={out_mmd:.4f}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        torch.cuda.empty_cache()
        gc.collect()

    # Combine all results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df.to_csv(f"{args.save_dir}/all_results.csv", index=False)

        with open(f"{args.save_dir}/all_results.pkl", 'wb') as f:
            pickle.dump(combined_df, f)

        # Generate aggregate plots (mean across seeds)
        print("\nGenerating aggregate plots...")
        for gen_type in combined_df['generator'].unique():
            gen_df = combined_df[combined_df['generator'] == gen_type]
            # Average across seeds
            agg_df = gen_df.groupby(['mu', 'method']).agg({
                'mmd_mean': 'mean', 'mmd_std': 'mean',
                'swd_mean': 'mean', 'swd_std': 'mean',
                'energy_mean': 'mean', 'energy_std': 'mean',
            }).reset_index()
            plot_all_metrics(agg_df, args.save_dir, f"{gen_type}_aggregate")

        # Print final summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY (averaged across seeds)")
        print("="*80)

        summary = []
        for gen_type in combined_df['generator'].unique():
            gen_df = combined_df[combined_df['generator'] == gen_type]
            for method in gen_df['method'].unique():
                method_df = gen_df[gen_df['method'] == method]
                in_support = method_df[method_df['mu'] <= 2.5]
                out_support = method_df[method_df['mu'] > 2.5]

                summary.append({
                    'generator': gen_type,
                    'method': method,
                    'in_support_mmd': in_support['mmd_mean'].mean(),
                    'out_support_mmd': out_support['mmd_mean'].mean(),
                    'in_support_swd': in_support['swd_mean'].mean(),
                    'out_support_swd': out_support['swd_mean'].mean(),
                })

        summary_df = pd.DataFrame(summary)
        print(summary_df.to_string(index=False))
        summary_df.to_csv(f"{args.save_dir}/summary.csv", index=False)


if __name__ == '__main__':
    main()
