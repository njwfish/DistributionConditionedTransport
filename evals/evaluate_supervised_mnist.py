"""
Unified evaluation for supervised vs semisupervised distribution transport on MNIST-Colors.

Evaluates both approaches on test distributions across the full color support [0, 1],
showing how performance varies with the source color. This reveals:
- How supervised models (source_only) generalize outside training support [0.2, 0.8]
- How semisupervised models (any-to-any + ridge predictor) perform across the range

Training uses restricted color range [0.2, 0.8], test uses full [0, 1].

Produces visualizations plotting performance metrics vs color, with clear comparison
between supervised, semisupervised, and oracle (upper bound) approaches.

Metrics computed on flattened images: MMD, SWD, Energy Distance.
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
import gc
import argparse
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from utils.experiment_utils import get_all_experiments_info, load_checkpoint
from generator.losses import mmd, sliced_wasserstein_distance
from datasets.mnist_colors import MNISTColorsDataset


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
    """Compute distributional metrics between generated and target samples (flattened images)."""
    # Flatten images: (batch, C, H, W) -> (batch, C*H*W)
    if generated.dim() == 4:
        gen_flat = generated.reshape(generated.shape[0], -1)
        tgt_flat = target.reshape(target.shape[0], -1)
    else:
        gen_flat = generated
        tgt_flat = target

    gen_t = gen_flat if isinstance(gen_flat, torch.Tensor) else torch.from_numpy(gen_flat).float()
    tgt_t = tgt_flat if isinstance(tgt_flat, torch.Tensor) else torch.from_numpy(tgt_flat).float()
    gen_t, tgt_t = gen_t.to(device), tgt_t.to(device)

    return {
        'mmd': mmd(gen_t, tgt_t).item(),
        'swd': sliced_wasserstein_distance(gen_t, tgt_t, n_projections=100, p=2).item(),
        'energy': compute_energy_distance(gen_t, tgt_t).item(),
    }


def compute_color_mse(generated, target):
    """Compute MSE between mean colors of generated and target images."""
    if generated.dim() == 4:
        gen_mean_color = generated.mean(dim=(2, 3))  # (batch, 3)
        tgt_mean_color = target.mean(dim=(2, 3))
    else:
        # Already flattened - reshape to get color channels
        batch_size = generated.shape[0]
        gen_reshaped = generated.reshape(batch_size, 3, -1)
        tgt_reshaped = target.reshape(batch_size, 3, -1)
        gen_mean_color = gen_reshaped.mean(dim=2)
        tgt_mean_color = tgt_reshaped.mean(dim=2)
    return ((gen_mean_color - tgt_mean_color) ** 2).mean().item()


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


def filter_experiments(configs, experiment_type='mnist_colors', num_epochs=200,
                       n_unique_sets=None, supervised=None, require_checkpoint=True):
    """Filter experiments by criteria.

    Args:
        configs: List of experiment configs
        experiment_type: Type of experiment ('mnist_colors', etc)
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
# Test data generation for MNIST with controlled color
# ============================================================================

def generate_mnist_test_data(color_value, n_distributions, set_size=64,
                              color_shift=(0.1, 0.1, 0.1), seed=None):
    """
    Generate MNIST test distributions at a specific source color value.

    Args:
        color_value: Base color value (applied to all RGB channels)
        n_distributions: Number of test distributions
        set_size: Images per distribution
        color_shift: RGB shift for target
        seed: Random seed

    Returns:
        source_data: (n_distributions, set_size, 3, 28, 28)
        target_data: (n_distributions, set_size, 3, 28, 28)
        source_colors: (n_distributions, 3)
        target_colors: (n_distributions, 3)
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Create a temporary dataset to sample from
    dataset = MNISTColorsDataset(
        n_sets=n_distributions,
        set_size=set_size,
        seed=seed,
    )

    source_data = []
    target_data = []
    source_colors = []
    target_colors = []

    color_shift_t = torch.tensor(color_shift)

    for i in range(n_distributions):
        # Generate source color in a small window around color_value
        delta = 0.05
        src_color = torch.tensor([
            np.random.uniform(max(0.0, color_value - delta), min(1.0, color_value + delta))
            for _ in range(3)
        ]).float()

        # Target color with shift (wrap using modulo)
        tgt_color = torch.fmod(src_color + color_shift_t, 1.0)

        # Sample images from a random digit class
        digit_class = np.random.randint(0, 10)
        source_images = dataset._sample_class_images(digit_class)
        target_images = dataset._sample_class_images(digit_class)

        # Apply colors
        source_samples = dataset._apply_color_transform(source_images, src_color)
        target_samples = dataset._apply_color_transform(target_images, tgt_color)

        # Shuffle target samples
        perm = torch.randperm(target_samples.size(0))
        target_samples = target_samples[perm]

        source_data.append(source_samples)
        target_data.append(target_samples)
        source_colors.append(src_color)
        target_colors.append(tgt_color)

    source_data = torch.stack(source_data)  # (n_dist, set_size, 3, 28, 28)
    target_data = torch.stack(target_data)
    source_colors = torch.stack(source_colors)
    target_colors = torch.stack(target_colors)

    return source_data, target_data, source_colors, target_colors


# ============================================================================
# Transport functions
# ============================================================================

def transport_supervised(enc, gen, source_samples, device):
    """
    Transport using supervised model (source_only conditioning).
    Target latent is zeros.

    Args:
        source_samples: (set_size, 3, H, W) tensor
    """
    with torch.no_grad():
        source_t = source_samples.float().to(device)

        # Add batch dimension for encoder: (1, set_size, 3, H, W)
        source_batch = source_t.unsqueeze(0)

        # Get source embedding: (1, latent_dim)
        source_lat = enc(source_batch)

        # Target latent is zeros (source_only): (1, latent_dim)
        target_lat = torch.zeros_like(source_lat)

        set_size = source_t.shape[0]

        # Expand latents for each sample in the set
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        # Transport
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)

        # Squeeze extra dimension if present
        transported = transported.squeeze()

        return transported.cpu()


def transport_semisupervised(enc, gen, predictor, source_samples, device):
    """
    Transport using semisupervised approach: any-to-any model + ridge predictor.
    The predictor predicts the RESIDUAL: target_lat = source_lat + predictor(source_lat)
    """
    with torch.no_grad():
        source_t = source_samples.float().to(device)

        # Add batch dimension for encoder
        source_batch = source_t.unsqueeze(0)

        # Get source embedding
        source_lat = enc(source_batch)

        # Predict target latent using residual ridge regression
        source_lat_np = source_lat.cpu().numpy()
        residual_np = predictor.predict(source_lat_np)
        target_lat_np = source_lat_np + residual_np
        target_lat = torch.from_numpy(target_lat_np).float().to(device)

        set_size = source_t.shape[0]

        # Expand latents
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        # Transport
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()

        return transported.cpu()


def transport_oracle(enc, gen, source_samples, target_samples, device):
    """
    Transport using oracle (true target embedding). Upper bound on performance.
    """
    with torch.no_grad():
        source_t = source_samples.float().to(device)
        target_t = target_samples.float().to(device)

        # Add batch dimension for encoder
        source_batch = source_t.unsqueeze(0)
        target_batch = target_t.unsqueeze(0)

        # Get embeddings
        source_lat = enc(source_batch)
        target_lat = enc(target_batch)

        set_size = source_t.shape[0]

        # Expand latents
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        # Transport
        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()

        return transported.cpu()


# ============================================================================
# Evaluation at specific color values
# ============================================================================

def evaluate_at_color(color_value, enc, gen, device, method='supervised', predictor=None,
                      n_distributions=50, set_size=64, color_shift=(0.1, 0.1, 0.1), seed=None):
    """
    Evaluate transport performance at a specific color value.

    Returns:
        dict with mean and std of each metric
    """
    # Generate test data at this color
    source_data, target_data, source_colors, target_colors = generate_mnist_test_data(
        color_value, n_distributions, set_size=set_size, color_shift=color_shift, seed=seed
    )

    mmd_vals, swd_vals, energy_vals, color_mse_vals = [], [], [], []

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

        # Also compute color MSE
        color_mse = compute_color_mse(transported, target)
        color_mse_vals.append(color_mse)

    return {
        'color': color_value,
        'mmd_mean': np.mean(mmd_vals),
        'mmd_std': np.std(mmd_vals),
        'swd_mean': np.mean(swd_vals),
        'swd_std': np.std(swd_vals),
        'energy_mean': np.mean(energy_vals),
        'energy_std': np.std(energy_vals),
        'color_mse_mean': np.mean(color_mse_vals),
        'color_mse_std': np.std(color_mse_vals),
    }


def fit_predictor_from_mnist_data(enc, device, n_train=5000, set_size=64,
                                   color_shift=(0.1, 0.1, 0.1), alpha=1.0, seed=42,
                                   predictor_type='ridge'):
    """
    Fit predictor using supervised MNIST data with restricted color support [0.2, 0.8].
    Predicts the RESIDUAL (target_emb - source_emb).
    """
    print(f"  Generating {n_train} supervised MNIST training samples...")
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataset for sampling images
    dataset = MNISTColorsDataset(
        n_sets=n_train,
        set_size=set_size,
        seed=seed,
    )

    color_shift_t = torch.tensor(color_shift)
    source_latents = []
    target_latents = []

    batch_size = 64
    print("  Embedding training data...")

    with torch.no_grad():
        for batch_start in tqdm(range(0, n_train, batch_size), desc="  Embedding"):
            batch_end = min(batch_start + batch_size, n_train)
            batch_sources = []
            batch_targets = []

            for i in range(batch_start, batch_end):
                # Generate source color in restricted support [0.2, 0.8]
                src_color = torch.tensor([
                    np.random.uniform(0.2, 0.8) for _ in range(3)
                ]).float()

                # Target color with shift
                tgt_color = torch.fmod(src_color + color_shift_t, 1.0)

                # Sample images
                digit_class = np.random.randint(0, 10)
                source_images = dataset._sample_class_images(digit_class)
                target_images = dataset._sample_class_images(digit_class)

                # Apply colors
                source_samples = dataset._apply_color_transform(source_images, src_color)
                target_samples = dataset._apply_color_transform(target_images, tgt_color)

                batch_sources.append(source_samples)
                batch_targets.append(target_samples)

            # Stack and encode
            src_batch = torch.stack(batch_sources).to(device)
            tgt_batch = torch.stack(batch_targets).to(device)

            source_latents.append(enc(src_batch).cpu().numpy())
            target_latents.append(enc(tgt_batch).cpu().numpy())

    source_latents = np.concatenate(source_latents, axis=0)
    target_latents = np.concatenate(target_latents, axis=0)

    # Compute residuals
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

    # Evaluate
    pred_residuals = predictor.predict(source_latents)
    pred_targets = source_latents + pred_residuals
    train_mse = np.mean((pred_targets - target_latents) ** 2)
    residual_mse = np.mean((pred_residuals - residuals) ** 2)
    print(f"  Residual MSE: {residual_mse:.6f}, Full prediction MSE: {train_mse:.6f}")

    return predictor, train_mse


# ============================================================================
# Visualization
# ============================================================================

def plot_performance_vs_color(results_df, save_path, metric='mmd', title=None):
    """Plot performance metric vs color for different methods."""
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
    ax.axvspan(0.2, 0.8, alpha=0.15, color='gray', label='Training support [0.2, 0.8]')
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    for method in ['supervised', 'semisupervised', 'oracle']:
        method_df = results_df[results_df['method'] == method].sort_values('color')
        if len(method_df) == 0:
            continue

        color_vals = method_df['color'].values
        mean = method_df[f'{metric}_mean'].values
        std = method_df[f'{metric}_std'].values

        ax.plot(color_vals, mean, color=colors[method], linewidth=2, label=labels[method])
        ax.fill_between(color_vals, mean - std, mean + std, color=colors[method], alpha=0.2)

    ax.set_xlabel('Source Color Value', fontsize=12)
    metric_labels = {
        'mmd': 'MMD',
        'swd': 'Sliced Wasserstein Distance',
        'energy': 'Energy Distance',
        'color_mse': 'Color MSE'
    }
    ax.set_ylabel(metric_labels.get(metric, metric), fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)

    ax.legend(loc='upper left', fontsize=10)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_metrics(results_df, save_dir, generator_name):
    """Plot all metrics for a given generator."""
    for metric in ['mmd', 'swd', 'energy', 'color_mse']:
        plot_performance_vs_color(
            results_df,
            f"{save_dir}/{generator_name}_{metric}_vs_color.png",
            metric=metric,
            title=f"{generator_name.replace('_', ' ').title()}: {metric.upper()} vs Color"
        )


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_model_pair(supervised_exp, unsupervised_exp, device, args, model_seed):
    """Evaluate supervised vs semisupervised for a matched pair of MNIST models."""
    results = []

    color_shift = (0.1, 0.1, 0.1)
    print(f"Using color shift: {color_shift}")

    # Load unsupervised model
    print("Loading unsupervised (any-to-any) model...")
    enc_unsup, gen_unsup = load_model(unsupervised_exp['config'], unsupervised_exp['dir'], device, args.num_epochs)

    # Fit predictor for semisupervised
    predictor, train_mse = fit_predictor_from_mnist_data(
        enc_unsup, device, n_train=args.n_train, set_size=args.set_size,
        color_shift=color_shift, alpha=args.ridge_alpha, seed=model_seed,
        predictor_type=args.predictor_type
    )

    # Load supervised model if available
    if supervised_exp is not None:
        print("Loading supervised model...")
        enc_sup, gen_sup = load_model(supervised_exp['config'], supervised_exp['dir'], device, args.num_epochs)
    else:
        enc_sup, gen_sup = None, None

    # Evaluate at each color value across full [0, 1] support
    color_values = np.linspace(0.05, 0.95, args.n_color_points)

    print(f"Using set_size={args.eval_set_size} for evaluation")

    for color_val in tqdm(color_values, desc="Evaluating across colors"):
        # Supervised
        if enc_sup is not None:
            res = evaluate_at_color(
                color_val, enc_sup, gen_sup, device, method='supervised',
                n_distributions=args.n_distributions, set_size=args.eval_set_size,
                color_shift=color_shift, seed=int(color_val * 10000)
            )
            res['method'] = 'supervised'
            results.append(res)

        # Semisupervised
        res = evaluate_at_color(
            color_val, enc_unsup, gen_unsup, device, method='semisupervised',
            predictor=predictor, n_distributions=args.n_distributions,
            set_size=args.eval_set_size, color_shift=color_shift,
            seed=int(color_val * 10000)
        )
        res['method'] = 'semisupervised'
        results.append(res)

        # Oracle
        res = evaluate_at_color(
            color_val, enc_unsup, gen_unsup, device, method='oracle',
            n_distributions=args.n_distributions, set_size=args.eval_set_size,
            color_shift=color_shift, seed=int(color_val * 10000)
        )
        res['method'] = 'oracle'
        results.append(res)

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./supervised_comparison_mnist')
    parser.add_argument('--n_train', type=int, default=5000, help='Training samples for ridge')
    parser.add_argument('--n_distributions', type=int, default=50, help='Test distributions per color')
    parser.add_argument('--n_color_points', type=int, default=15, help='Number of color values to test')
    parser.add_argument('--set_size', type=int, default=64, help='Images per distribution for predictor training')
    parser.add_argument('--eval_set_size', type=int, default=64, help='Images per distribution for evaluation')
    parser.add_argument('--ridge_alpha', type=float, default=1.0, help='Ridge regularization')
    parser.add_argument('--predictor_type', type=str, default='ridge', choices=['ridge', 'mlp'])
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

    # Find supervised MNIST experiments
    supervised_exps = filter_experiments(
        configs, 'mnist_colors', args.num_epochs, supervised=True
    )
    print(f"Found {len(supervised_exps)} supervised mnist_colors experiments")

    # Find unsupervised (any-to-any) MNIST experiments
    unsupervised_exps = filter_experiments(
        configs, 'mnist_colors', args.num_epochs, n_unique_sets=10000, supervised=False
    )
    print(f"Found {len(unsupervised_exps)} unsupervised mnist_colors experiments (n_unique_sets=10000)")

    if not unsupervised_exps:
        print("No unsupervised experiments found!")
        return

    # Helper to extract generator type
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

    # Evaluate each model pair
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
                sup_n_unique = s['config']['dataset'].get('n_unique_sets', 10000)
                if sup_n_unique == 10000:
                    sup_exp = s
                    break

        if sup_exp:
            print(f"Found matching supervised experiment")
        else:
            print(f"No matching supervised experiment, will only evaluate semisupervised + oracle")

        try:
            results_df = evaluate_model_pair(sup_exp, unsup_exp, device, args, model_seed=seed)
            results_df['generator'] = gen_type
            results_df['seed'] = seed

            # Generate plots
            plot_all_metrics(results_df, args.save_dir, f"{gen_type}_seed{seed}")

            all_results.append(results_df)

            # Print summary
            print("\nSummary:")
            in_support = results_df[(results_df['color'] >= 0.2) & (results_df['color'] <= 0.8)]
            out_support = results_df[(results_df['color'] < 0.2) | (results_df['color'] > 0.8)]

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

        # Generate aggregate plots
        print("\nGenerating aggregate plots...")
        for gen_type in combined_df['generator'].unique():
            gen_df = combined_df[combined_df['generator'] == gen_type]
            agg_df = gen_df.groupby(['color', 'method']).agg({
                'mmd_mean': 'mean', 'mmd_std': 'mean',
                'swd_mean': 'mean', 'swd_std': 'mean',
                'energy_mean': 'mean', 'energy_std': 'mean',
                'color_mse_mean': 'mean', 'color_mse_std': 'mean',
            }).reset_index()
            plot_all_metrics(agg_df, args.save_dir, f"{gen_type}_aggregate")

        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY (averaged across seeds)")
        print("="*80)

        summary = []
        for gen_type in combined_df['generator'].unique():
            gen_df = combined_df[combined_df['generator'] == gen_type]
            for method in gen_df['method'].unique():
                method_df = gen_df[gen_df['method'] == method]
                in_support = method_df[(method_df['color'] >= 0.2) & (method_df['color'] <= 0.8)]
                out_support = method_df[(method_df['color'] < 0.2) | (method_df['color'] > 0.8)]

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
