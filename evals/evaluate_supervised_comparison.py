"""
Unified evaluation for supervised vs semisupervised distribution transport.

Evaluates both approaches on a 2D grid of test distributions across [0, 5] x [0, 5],
showing how performance varies with the mean location (mu_x, mu_y). This reveals:
- How supervised models (source_only) generalize outside training support [0, 2.5]^2
- How semisupervised models (any-to-any + ridge predictor) perform across the range

Produces results that can be visualized as heatmaps or contour plots.
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
from sklearn.linear_model import RidgeCV
import scipy as sp
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
    enc.load_state_dict(state['encoder_state_dict'], strict=False)
    gen.load_state_dict(state['generator_state_dict'], strict=False)

    enc.eval().to(device)
    gen.eval().to(device)

    return enc, gen


def get_encoder_type(exp):
    """Classify encoder type from experiment config."""
    enc_target = exp['config'].get('encoder', {}).get('_target_', exp.get('encoder', ''))
    enc_str = str(enc_target)
    if 'KME' in enc_str or 'kernel_mean' in enc_str:
        return 'kme'
    elif 'DistributionEncoderTx' in enc_str or 'EncoderTx' in enc_str:
        return 'tx'
    elif 'Embedding' in enc_str:
        return 'embedding'
    else:
        return 'gnn'


def filter_experiments(configs, experiment_type='mvn', num_epochs=200,
                       n_unique_sets=None, supervised=None, require_checkpoint=True,
                       encoder_type=None, checkpoint_epoch=None):
    """Filter experiments by criteria."""
    import os
    ckpt_epoch = checkpoint_epoch if checkpoint_epoch is not None else num_epochs
    filtered = []
    for c in configs:
        if experiment_type not in c['name']:
            continue
        if supervised is not None:
            is_supervised = 'supervised' in c['name']
            if is_supervised != supervised:
                continue
        if require_checkpoint:
            checkpoint_path = os.path.join(c['dir'], f'checkpoint_epoch_{ckpt_epoch}.pt')
            if not os.path.exists(checkpoint_path):
                continue
        if n_unique_sets is not None:
            cfg_n_unique = c['config']['dataset'].get('n_unique_sets')
            if cfg_n_unique != n_unique_sets:
                continue
        if 'Embedding' in c.get('encoder', ''):
            continue
        if encoder_type is not None and get_encoder_type(c) != encoder_type:
            continue
        filtered.append(c)
    return filtered


# ============================================================================
# Test data generation on 2D grid
# ============================================================================

def generate_distribution_at_mu(mu_x, mu_y, data_dim=2, prior_cov_df=10,
                                 prior_cov_scale=1.0, set_size=100, seed=None):
    """
    Generate a single test distribution with mean at (mu_x, mu_y).

    Returns:
        data: (set_size, data_dim) samples
        mu: (data_dim,) mean vector
        cov: (data_dim, data_dim) covariance matrix
    """
    if seed is not None:
        np.random.seed(seed)

    mu = np.array([mu_x, mu_y])

    # Generate covariance from inverse Wishart
    prior_cov_df_adjusted = prior_cov_df * (data_dim - 1)
    cov = np.linalg.inv(sp.stats.wishart.rvs(
        df=prior_cov_df_adjusted,
        scale=prior_cov_scale * np.eye(data_dim)
    ))

    # Sample data
    L = np.linalg.cholesky(cov)
    z = np.random.randn(set_size, data_dim)
    data = z @ L.T + mu

    return data, mu, cov


def generate_mvn_target(source_data, shift=(1.0, 1.0), seed=None):
    """
    Generate MVN target data: Y = X + shift (then shuffled).
    Matches SupervisedMVNDataset.
    """
    if seed is not None:
        np.random.seed(seed)

    b = np.array(shift)
    target_data = source_data + b
    np.random.shuffle(target_data)

    return target_data


def generate_gmm_target(source_data, shift=(0.5, 0.5), off_axis_shift=None, seed=None):
    """
    Generate GMM target data with off-axis shift.
    Matches SupervisedGMMDataset.

    Creates two shifted versions:
    - Left: X + shift + off_axis_shift
    - Right: X + shift - off_axis_shift
    Then stacks and samples half.

    Args:
        source_data: Source samples
        shift: Base shift vector
        off_axis_shift: Off-axis shift vector (e.g., [-0.1, 0.1] for GMM)
        seed: Random seed
    """
    if seed is not None:
        np.random.seed(seed)

    b = np.array(shift)
    if off_axis_shift is None:
        off_axis_shift = np.array([-0.1, 0.1])  # Default
    else:
        off_axis_shift = np.array(off_axis_shift)

    target_left = source_data + b + off_axis_shift
    target_right = source_data + b - off_axis_shift
    target_stacked = np.vstack([target_left, target_right])

    np.random.shuffle(target_stacked)
    target_data = target_stacked[:source_data.shape[0]]

    return target_data


# ============================================================================
# Transport functions
# ============================================================================

def transport_supervised(enc, gen, source_samples, device):
    """
    Transport using supervised model (source_only conditioning).
    Target latent is zeros.
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)
        source_batch = source_t.unsqueeze(0)
        source_lat = enc(source_batch)
        target_lat = torch.zeros_like(source_lat)

        set_size = source_t.shape[0]
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)

        return transported.cpu().numpy()


def transport_semisupervised(enc, gen, predictor, source_samples, device):
    """
    Transport using semisupervised approach: any-to-any model + ridge predictor.
    The predictor predicts the RESIDUAL: target_lat = source_lat + predictor(source_lat)
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)
        source_batch = source_t.unsqueeze(0)
        source_lat = enc(source_batch)

        source_lat_np = source_lat.cpu().numpy()
        residual_np = predictor.predict(source_lat_np)
        target_lat_np = source_lat_np + residual_np
        target_lat = torch.from_numpy(target_lat_np).float().to(device)

        set_size = source_t.shape[0]
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)

        return transported.cpu().numpy()


def transport_oracle(enc, gen, source_samples, target_samples, device):
    """
    Transport using oracle (true target embedding). Upper bound on performance.
    """
    with torch.no_grad():
        source_t = torch.from_numpy(source_samples).float().to(device)
        target_t = torch.from_numpy(target_samples).float().to(device)

        source_batch = source_t.unsqueeze(0)
        target_batch = target_t.unsqueeze(0)

        source_lat = enc(source_batch)
        target_lat = enc(target_batch)

        set_size = source_t.shape[0]
        source_lat_exp = source_lat.expand(set_size, -1)
        target_lat_exp = target_lat.expand(set_size, -1)

        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()
        if transported.dim() == 1:
            transported = transported.unsqueeze(-1)

        return transported.cpu().numpy()


# ============================================================================
# Evaluation on 2D grid
# ============================================================================

def is_in_distribution(mu_x, mu_y, threshold=2.5):
    """Check if a point is in the training support [0, threshold]^2."""
    return mu_x <= threshold and mu_y <= threshold


def evaluate_at_grid_point(mu_x, mu_y, enc, gen, device, method='supervised',
                           predictor=None, set_size=100, data_dim=2, seed=None,
                           experiment_type='mvn', shift=None, off_axis_shift=None):
    """
    Evaluate transport performance at a single grid point (mu_x, mu_y).

    Args:
        mu_x, mu_y: Grid point coordinates (mean location)
        enc, gen: Encoder and generator models
        device: Device to use
        method: 'supervised', 'semisupervised', or 'oracle'
        predictor: Ridge predictor (required for semisupervised)
        set_size: Samples per distribution
        data_dim: Data dimensionality
        seed: Random seed
        experiment_type: 'mvn' or 'gmm'
        shift: Shift vector (from dataset)
        off_axis_shift: Off-axis shift vector for GMM (from dataset)

    Returns:
        dict with metrics
    """
    # Generate source distribution at this grid point
    source_data, source_mu, source_cov = generate_distribution_at_mu(
        mu_x, mu_y, data_dim=data_dim, set_size=set_size, seed=seed
    )

    # Generate target data based on experiment type
    # Note: shift should always come from the dataset config, not hardcoded defaults
    if shift is None:
        raise ValueError("shift must be provided from the supervised dataset config")

    if experiment_type == 'mvn':
        target_data = generate_mvn_target(source_data, shift=shift, seed=seed)
    else:  # gmm
        target_data = generate_gmm_target(source_data, shift=shift, off_axis_shift=off_axis_shift, seed=seed)

    # Transport
    if method == 'supervised':
        transported = transport_supervised(enc, gen, source_data, device)
    elif method == 'semisupervised':
        transported = transport_semisupervised(enc, gen, predictor, source_data, device)
    elif method == 'oracle':
        transported = transport_oracle(enc, gen, source_data, target_data, device)
    else:
        raise ValueError(f"Unknown method: {method}")

    metrics = compute_metrics(transported, target_data, device)

    return {
        'mu_x': mu_x,
        'mu_y': mu_y,
        'in_distribution': is_in_distribution(mu_x, mu_y),
        **metrics
    }


def fit_predictor_from_dataset(enc, device, supervised_exp_config, n_train=10000):
    """
    Fit predictor using the exact same dataset as the supervised model training.

    Instantiates the dataset from the experiment config and uses it for training.
    Predicts the RESIDUAL (target_emb - source_emb) rather than target_emb directly.
    """
    print(f"  Loading supervised dataset from config...")

    # Instantiate the dataset exactly as training would
    dataset = hydra.utils.instantiate(supervised_exp_config['dataset'])

    # Limit to n_train samples
    n_train = min(n_train, len(dataset))
    print(f"  Using {n_train} samples from dataset (total: {len(dataset)})")

    # Collect source and target data
    source_data_list = []
    target_data_list = []

    for i in tqdm(range(n_train), desc="  Loading data"):
        sample = dataset[i]
        source_data_list.append(sample['source_samples'].numpy())
        target_data_list.append(sample['target_samples'].numpy())

    source_data = np.array(source_data_list)
    target_data = np.array(target_data_list)

    print(f"  Embedding {n_train} training distributions...")
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

    # Fit RidgeCV
    alphas = np.logspace(-6, 6, 50)
    print(f"  Fitting RidgeCV predictor for RESIDUAL (searching {len(alphas)} alphas via CV)...")
    predictor = RidgeCV(alphas=alphas, cv=5)
    predictor.fit(source_latents, residuals)
    print(f"  Selected alpha: {predictor.alpha_:.6g}")

    # Evaluate
    pred_residuals = predictor.predict(source_latents)
    pred_targets = source_latents + pred_residuals
    train_mse = np.mean((pred_targets - target_latents) ** 2)
    print(f"  Training MSE: {train_mse:.6f}")

    # Also get the shift/transformation info from dataset for evaluation
    dataset_info = {
        'shift': getattr(dataset, 'b', None),
        'off_axis_shift': getattr(dataset, 'off_axis_shift', None),
    }

    return predictor, train_mse, dataset_info


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_model_pair(supervised_exp, unsupervised_exp, device, args, model_seed):
    """
    Evaluate supervised vs semisupervised on a 2D grid.
    """
    results = []

    sup_ckpt = args.supervised_checkpoint_epoch or args.checkpoint_epoch
    unsup_ckpt = args.unsupervised_checkpoint_epoch or args.checkpoint_epoch

    # Load unsupervised model (used for semisupervised and oracle)
    print(f"Loading unsupervised (any-to-any) model at epoch {unsup_ckpt}...")
    enc_unsup, gen_unsup = load_model(unsupervised_exp['config'], unsupervised_exp['dir'], device, unsup_ckpt)

    # Load supervised model if available
    if supervised_exp is not None:
        print(f"Loading supervised model at epoch {sup_ckpt}...")
        enc_sup, gen_sup = load_model(supervised_exp['config'], supervised_exp['dir'], device, sup_ckpt)

        # Fit predictor using the exact same dataset as supervised training
        predictor, train_mse, dataset_info = fit_predictor_from_dataset(
            enc_unsup, device, supervised_exp['config'], n_train=args.n_train
        )

        # Get shift/transformation from the dataset
        if dataset_info['shift'] is None:
            raise ValueError("Could not extract shift from supervised dataset. Check dataset.b attribute.")
        shift = tuple(dataset_info['shift'])
        off_axis_shift = dataset_info['off_axis_shift']  # May be None for MVN

        # Get data_dim from config
        data_shape = supervised_exp['config']['dataset'].get('data_shape', [2])
        # Handle both list and ListConfig from hydra
        if hasattr(data_shape, '__iter__') and not isinstance(data_shape, str):
            data_dim = int(data_shape[0])
        else:
            data_dim = int(data_shape)
    else:
        enc_sup, gen_sup = None, None
        predictor = None
        # Without a supervised experiment, we cannot properly evaluate
        # as we don't know the exact transformation used
        raise ValueError(
            "No supervised experiment found. A supervised experiment is required "
            "to determine the correct transformation (shift) for evaluation."
        )

    print(f"Experiment type: {args.experiment}")
    print(f"Data dim: {data_dim}")
    print(f"Shift: {shift}")
    if off_axis_shift is not None:
        print(f"Off-axis shift: {off_axis_shift}")

    # Create 2D evaluation grid
    mu_x_values = np.linspace(args.mu_min, args.mu_max, args.n_grid_points)
    mu_y_values = np.linspace(args.mu_min, args.mu_max, args.n_grid_points)

    print(f"\nEvaluating on {args.n_grid_points}x{args.n_grid_points} grid from [{args.mu_min}, {args.mu_max}]^2")
    print(f"Using eval_set_size={args.eval_set_size}")

    total_points = args.n_grid_points ** 2
    pbar = tqdm(total=total_points * 3, desc="Evaluating grid")  # 3 methods

    for mu_x in mu_x_values:
        for mu_y in mu_y_values:
            # Deterministic seed based on grid position
            grid_seed = int((mu_x * 1000 + mu_y * 10 + model_seed) % (2**31))

            # Supervised
            if enc_sup is not None:
                res = evaluate_at_grid_point(
                    mu_x, mu_y, enc_sup, gen_sup, device, method='supervised',
                    set_size=args.eval_set_size, data_dim=data_dim, seed=grid_seed,
                    experiment_type=args.experiment, shift=shift, off_axis_shift=off_axis_shift
                )
                res['method'] = 'supervised'
                results.append(res)
            pbar.update(1)

            # Semisupervised
            if predictor is not None:
                res = evaluate_at_grid_point(
                    mu_x, mu_y, enc_unsup, gen_unsup, device, method='semisupervised',
                    predictor=predictor, set_size=args.eval_set_size, data_dim=data_dim,
                    seed=grid_seed, experiment_type=args.experiment, shift=shift, off_axis_shift=off_axis_shift
                )
                res['method'] = 'semisupervised'
                results.append(res)
            pbar.update(1)

            # Oracle
            res = evaluate_at_grid_point(
                mu_x, mu_y, enc_unsup, gen_unsup, device, method='oracle',
                set_size=args.eval_set_size, data_dim=data_dim, seed=grid_seed,
                experiment_type=args.experiment, shift=shift, off_axis_shift=off_axis_shift
            )
            res['method'] = 'oracle'
            results.append(res)
            pbar.update(1)

    pbar.close()

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--experiment', type=str, default='mvn', choices=['mvn', 'gmm'])
    parser.add_argument('--num_epochs', type=int, default=200, help='Filter experiments that trained to this many epochs')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='Load checkpoint at this epoch (defaults to num_epochs)')
    parser.add_argument('--supervised_checkpoint_epoch', type=int, default=None, help='Load supervised checkpoint at this epoch (overrides checkpoint_epoch for supervised)')
    parser.add_argument('--unsupervised_checkpoint_epoch', type=int, default=None, help='Load unsupervised checkpoint at this epoch (overrides checkpoint_epoch for unsupervised)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='./supervised_comparison_results')
    parser.add_argument('--n_train', type=int, default=10000, help='Number of training samples for predictor (from supervised dataset)')
    parser.add_argument('--n_grid_points', type=int, default=21, help='Evaluation grid points per dimension')
    parser.add_argument('--mu_min', type=float, default=0.25, help='Minimum mu value')
    parser.add_argument('--mu_max', type=float, default=4.75, help='Maximum mu value')
    parser.add_argument('--eval_set_size', type=int, default=10000, help='Samples per distribution for evaluation')
    parser.add_argument('--data_dim', type=int, default=2, help='Data dimensionality')
    parser.add_argument('--generator', type=str, default=None, help='Specific generator to evaluate')
    parser.add_argument('--seed', type=int, default=None, help='Specific seed to evaluate')
    args = parser.parse_args()

    # Default checkpoint_epoch to num_epochs if not specified
    if args.checkpoint_epoch is None:
        args.checkpoint_epoch = args.num_epochs

    # Resolve per-model checkpoint epochs
    sup_ckpt = args.supervised_checkpoint_epoch or args.checkpoint_epoch
    unsup_ckpt = args.unsupervised_checkpoint_epoch or args.checkpoint_epoch

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Loading supervised checkpoint at epoch {sup_ckpt}, unsupervised at epoch {unsup_ckpt}")

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # Load all experiment configs
    print("Loading experiment configs...")
    configs = get_all_experiments_info(args.output_dir)

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

    all_results = []

    # Iterate over encoder types
    for enc_type in ['gnn', 'kme', 'tx']:
        # Find supervised experiments for this encoder
        supervised_exps = filter_experiments(
            configs, args.experiment, args.num_epochs, supervised=True, encoder_type=enc_type,
            checkpoint_epoch=sup_ckpt
        )
        print(f"\nFound {len(supervised_exps)} supervised {args.experiment} experiments (encoder={enc_type}, ckpt={sup_ckpt})")

        # Find unsupervised (any-to-any) experiments with n_unique_sets=10000
        unsupervised_exps = filter_experiments(
            configs, args.experiment, args.num_epochs, n_unique_sets=10000, supervised=False, encoder_type=enc_type,
            checkpoint_epoch=unsup_ckpt
        )
        print(f"Found {len(unsupervised_exps)} unsupervised {args.experiment} experiments (encoder={enc_type}, n_unique_sets=10000, ckpt={unsup_ckpt})")

        if not unsupervised_exps:
            print(f"No unsupervised experiments found for encoder={enc_type}, skipping")
            continue

        for unsup_exp in unsupervised_exps:
            gen_type = get_generator_type(unsup_exp)
            seed = unsup_exp['config'].get('seed', 42)

            if args.generator and gen_type != args.generator:
                continue
            if args.seed is not None and seed != args.seed:
                continue

            print(f"\n{'='*60}")
            print(f"Encoder: {enc_type}, Generator: {gen_type}, Seed: {seed}")
            print(f"{'='*60}")

            # Find matching supervised experiment (same encoder, generator, seed)
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
                print(f"No matching supervised experiment found")

            try:
                results_df = evaluate_model_pair(sup_exp, unsup_exp, device, args, model_seed=seed)
                results_df['generator'] = gen_type
                results_df['seed'] = seed
                results_df['encoder'] = enc_type

                all_results.append(results_df)

                # Print summary
                print("\nSummary (mean metrics):")
                id_mask = results_df['in_distribution']
                ood_mask = ~results_df['in_distribution']

                for method in results_df['method'].unique():
                    method_df = results_df[results_df['method'] == method]
                    id_mmd = method_df[id_mask[method_df.index]]['mmd'].mean()
                    ood_mmd = method_df[ood_mask[method_df.index]]['mmd'].mean()
                    print(f"  {method:20s}: ID MMD={id_mmd:.4f}, OOD MMD={ood_mmd:.4f}")

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

        print(f"\nSaved results to {args.save_dir}/all_results.csv")

        # Print final summary
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)

        for enc_type in combined_df['encoder'].unique():
            for gen_type in combined_df['generator'].unique():
                subset = combined_df[(combined_df['encoder'] == enc_type) & (combined_df['generator'] == gen_type)]
                if len(subset) == 0:
                    continue
                print(f"\n{enc_type} / {gen_type}:")

                for method in ['supervised', 'semisupervised', 'oracle']:
                    method_df = subset[subset['method'] == method]
                    if len(method_df) == 0:
                        continue
                    id_mmd = method_df[method_df['in_distribution']]['mmd'].mean()
                    ood_mmd = method_df[~method_df['in_distribution']]['mmd'].mean()
                    id_swd = method_df[method_df['in_distribution']]['swd'].mean()
                    ood_swd = method_df[~method_df['in_distribution']]['swd'].mean()
                    print(f"  {method:20s}: ID MMD={id_mmd:.4f}, OOD MMD={ood_mmd:.4f}, ID SWD={id_swd:.4f}, OOD SWD={ood_swd:.4f}")


if __name__ == '__main__':
    main()
