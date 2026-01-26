"""
Evaluation script for MVN and GMM experiments.

Evaluates all trained models (DistributionEncoder vs EmbeddingEncoder) across:
- Different generators (flow_matching, mmd, sinkhorn, wasserstein, energy)
- Different n_unique_sets (10, 100, 1000, 10000)
- Different seeds (40, 41, 42)

Metrics computed:
- Analytic Wasserstein (Bures squared distance)
- MMD
- Sliced Wasserstein Distance
- Energy Distance
"""

import sys
from pathlib import Path

# Add parent dir to path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import hydra
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import copy
import gc
import scipy as sp

from utils.experiment_utils import get_all_experiments_info, load_best_model, load_checkpoint
from utils.experiment_utils import get_experiment_info as load_experiment_info
from generator.losses import mmd, sliced_wasserstein_distance


def get_filtered_experiments_fast(output_dir, experiment_type='gmm', checkpoint_epoch=None):
    """
    Fast experiment loading that pre-filters by directory name.

    Much faster than loading all configs when there are many experiments.
    """
    import os
    from omegaconf import OmegaConf

    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return []

    # Get all directories
    all_dirs = [d for d in os.listdir(output_dir)
                if os.path.isdir(os.path.join(output_dir, d))]

    # Pre-filter by experiment type in directory name
    # GMM experiments: gmm_exp_*, gmm_emb_exp_*
    # MVN experiments: mvn_exp_*, mvn_emb_exp_*
    if experiment_type == 'gmm':
        prefixes = ['gmm_exp', 'gmm_emb_exp']
    else:
        prefixes = ['mvn_exp', 'mvn_emb_exp']

    filtered_dirs = [d for d in all_dirs if any(d.startswith(p) for p in prefixes)]
    print(f"  Pre-filtered {len(filtered_dirs)}/{len(all_dirs)} directories by name ({experiment_type})")

    # Further filter by checkpoint existence if needed
    if checkpoint_epoch is not None:
        ckpt_name = f"checkpoint_epoch_{checkpoint_epoch}.pt"
        filtered_dirs = [d for d in filtered_dirs
                        if os.path.exists(os.path.join(output_dir, d, ckpt_name))]
        print(f"  Filtered to {len(filtered_dirs)} directories with {ckpt_name}")

    # Load configs for filtered directories with progress
    experiments = []
    for i, exp_name in enumerate(filtered_dirs):
        if (i + 1) % 20 == 0 or i == len(filtered_dirs) - 1:
            print(f"  Loading configs: {i+1}/{len(filtered_dirs)}", end='\r')

        exp_dir = os.path.join(output_dir, exp_name)
        try:
            info = load_experiment_info(exp_dir, load_checkpoints=False, force_no_checkpoints=True)
            experiments.append(info)
        except Exception as e:
            pass  # Skip failed loads

    print()  # New line after progress
    return experiments


# ============================================================================
# GMM distance functions
# ============================================================================

def _solve_ot_batch(args):
    """Solve a batch of OT problems. Helper for parallel processing."""
    import ot
    start_i, weights1_batch, weights2, all_bures_batch = args
    n2 = len(weights2)
    results = []
    for i, bures_row in enumerate(all_bures_batch):
        for j in range(n2):
            val = ot.emd2(weights1_batch[i], weights2[j], bures_row[j])
            results.append((start_i + i, j, val))
    return results


def pairwise_gmm_bures_distance(weights1, mu1, cov1, weights2, mu2, cov2, verbose=False, n_jobs=8):
    """
    Compute pairwise GMM-OT (Wasserstein-type) distances between two sets of GMMs.

    Uses the GMM optimal transport distance from Delon & Desolneux (2020).
    Optimized with vectorized Bures computation and parallel OT solving.

    Args:
        weights1: (n1, k) mixture weights for first set of GMMs
        mu1: (n1, k, d) component means for first set
        cov1: (n1, k, d, d) component covariances for first set
        weights2: (n2, k) mixture weights for second set of GMMs
        mu2: (n2, k, d) component means for second set
        cov2: (n2, k, d, d) component covariances for second set
        verbose: Whether to show progress
        n_jobs: Number of parallel workers for OT solving (default: 8)

    Returns:
        dist: (n1, n2) matrix of GMM-OT distances
    """
    from ot.gmm import dist_bures_squared
    from concurrent.futures import ProcessPoolExecutor
    import ot

    n1, k, d = mu1.shape
    n2 = mu2.shape[0]

    if verbose:
        print(f"      Computing vectorized Bures distances ({n1}x{k} vs {n2}x{k} components)...")

    # Step 1: Vectorized Bures computation for ALL component pairs
    mu1_flat = mu1.reshape(n1 * k, d)
    mu2_flat = mu2.reshape(n2 * k, d)
    cov1_flat = cov1.reshape(n1 * k, d, d)
    cov2_flat = cov2.reshape(n2 * k, d, d)

    # Compute all pairwise Bures distances between components: (n1*k, n2*k)
    all_bures = dist_bures_squared(mu1_flat, mu2_flat, cov1_flat, cov2_flat)

    # Reshape to (n1, k, n2, k) then transpose to (n1, n2, k, k)
    all_bures = all_bures.reshape(n1, k, n2, k).transpose(0, 2, 1, 3)

    if verbose:
        print(f"      Solving {n1}x{n2} OT problems with {n_jobs} workers...")

    # Step 2: Solve OT problems in parallel using ProcessPoolExecutor
    n_problems = n1 * n2
    if n_problems < 1000 or n_jobs <= 1:
        # For small problems, use simple loop (avoid multiprocessing overhead)
        result = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                result[i, j] = ot.emd2(weights1[i], weights2[j], all_bures[i, j])
    else:
        # For large problems, use parallel processing
        batch_size = max(1, n1 // n_jobs)
        tasks = []
        for w in range(n_jobs):
            start_i = w * batch_size
            end_i = start_i + batch_size if w < n_jobs - 1 else n1
            if start_i < n1:
                tasks.append((start_i, weights1[start_i:end_i], weights2, all_bures[start_i:end_i]))

        result = np.zeros((n1, n2))
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for batch_results in executor.map(_solve_ot_batch, tasks):
                for i, j, val in batch_results:
                    result[i, j] = val

    return result


# ============================================================================
# Metric functions
# ============================================================================

def batched_cov(x):
    """Compute batched covariance matrices."""
    x_centered = x - x.mean(axis=1, keepdims=True)
    cov = x_centered.transpose(0, 2, 1) @ x_centered
    cov = cov / (x.shape[1] - 1)
    return cov


def dist_bures_squared_diag(m_s, m_t, C_s, C_t):
    """
    Compute squared Bures distance between corresponding Gaussians.
    Works for MVN (exact) and GMM (approximation treating as single Gaussian).
    """
    try:
        from ot.backend import get_backend
        nx = get_backend(m_s, C_s, m_t, C_t)
    except ImportError:
        # Fallback to numpy implementation
        return _dist_bures_squared_diag_numpy(m_s, m_t, C_s, C_t)
    
    # Mean part: ||m_s[i] - m_t[i]||^2
    diff = m_s - m_t
    D_means = nx.einsum("id,id->i", diff, diff)
    
    # Covariance part
    Cs12 = nx.sqrtm(C_s)
    C2 = nx.einsum("ikl,ilm,imn->ikn", Cs12, C_t, Cs12)
    C = nx.sqrtm(C2)
    
    trace_C_s = nx.einsum("ikk->i", C_s)
    trace_C_t = nx.einsum("ikk->i", C_t)
    trace_C = nx.einsum("ikk->i", C)
    
    D_covs = trace_C_s + trace_C_t - 2 * trace_C
    
    return nx.maximum(D_means + D_covs, 0)


def _dist_bures_squared_diag_numpy(m_s, m_t, C_s, C_t):
    """Numpy fallback for Bures distance."""
    from scipy.linalg import sqrtm
    
    n = m_s.shape[0]
    result = np.zeros(n)
    
    for i in range(n):
        # Mean part
        diff = m_s[i] - m_t[i]
        D_mean = np.dot(diff, diff)
        
        # Covariance part
        Cs12 = sqrtm(C_s[i])
        if np.iscomplexobj(Cs12):
            Cs12 = Cs12.real
        C2 = Cs12 @ C_t[i] @ Cs12
        C = sqrtm(C2)
        if np.iscomplexobj(C):
            C = C.real
        
        D_cov = np.trace(C_s[i]) + np.trace(C_t[i]) - 2 * np.trace(C)
        result[i] = max(D_mean + D_cov, 0)
    
    return result


def compute_energy_distance(x, y):
    """Compute energy distance between two sample sets."""
    # E[||X - Y||] - 0.5 * E[||X - X'||] - 0.5 * E[||Y - Y'||]
    n, m = x.shape[0], y.shape[0]
    
    # Cross term
    xy_dist = torch.cdist(x, y, p=2).mean()
    
    # Self terms
    xx_dist = torch.cdist(x, x, p=2).sum() / (n * (n - 1)) if n > 1 else 0
    yy_dist = torch.cdist(y, y, p=2).sum() / (m * (m - 1)) if m > 1 else 0
    
    return 2 * xy_dist - xx_dist - yy_dist


def compute_all_metrics(generated, target, device='cpu'):
    """Compute all distributional metrics between generated and target samples."""
    gen_t = torch.from_numpy(generated).float().to(device)
    tgt_t = torch.from_numpy(target).float().to(device)
    
    metrics = {}
    
    # MMD
    metrics['mmd'] = mmd(gen_t, tgt_t).item()
    
    # Sliced Wasserstein
    metrics['swd'] = sliced_wasserstein_distance(gen_t, tgt_t, n_projections=100, p=2).item()
    
    # Energy distance
    metrics['energy'] = compute_energy_distance(gen_t, tgt_t).item()
    
    return metrics


# ============================================================================
# Training distribution parameter generation
# ============================================================================

def get_training_distribution_params(exp_config, n_unique_sets, is_gmm=False):
    """Generate the distribution parameters that were used during training.

    CRITICAL: For embedding encoder, the embedding indices correspond to specific
    distributions generated with the training seed. We must use the same seed to
    get the correct parameters.

    Args:
        exp_config: Experiment configuration
        n_unique_sets: Number of unique distributions to return
        is_gmm: If True, return GMM parameters (weights, mu, cov)

    Returns:
        For MVN: (mu, cov) where mu is (n, d) and cov is (n, d, d)
        For GMM: (weights, mu, cov) where weights is (n, k), mu is (n, k, d), cov is (n, k, d, d)
    """
    # Get training config
    dataset_cfg = exp_config['dataset']
    seed = exp_config.get('seed', 42)
    data_shape = dataset_cfg.get('data_shape', [2])

    # Handle both list and ListConfig
    if hasattr(data_shape, '__iter__') and not isinstance(data_shape, str):
        data_dim = int(list(data_shape)[0])
    else:
        data_dim = int(data_shape)

    prior_mu = list(dataset_cfg.get('prior_mu', [0, 5]))
    prior_cov_df = int(dataset_cfg.get('prior_cov_df', 10))
    prior_cov_scale = float(dataset_cfg.get('prior_cov_scale', 1.0))
    n_sets = int(dataset_cfg.get('n_sets', 50000))

    # Regenerate parameters with training seed
    np.random.seed(seed)

    if is_gmm:
        # GMM parameters
        n_components = int(dataset_cfg.get('n_components', 3))
        dirichlet_alpha = float(dataset_cfg.get('dirichlet_alpha', 1.0))

        # Sample mixture weights from Dirichlet distribution
        weights = np.random.dirichlet(
            alpha=[dirichlet_alpha] * n_components,
            size=n_sets
        )  # shape: (n_sets, n_components)

        # Sample component means
        mu = np.random.uniform(
            prior_mu[0], prior_mu[1],
            (n_sets, n_components, data_dim)
        )  # shape: (n_sets, n_components, data_dim)

        # Sample component covariances from inverse Wishart
        cov = np.zeros((n_sets, n_components, data_dim, data_dim))
        for i in range(n_sets):
            for j in range(n_components):
                cov[i, j] = np.linalg.inv(sp.stats.wishart.rvs(
                    df=prior_cov_df,
                    scale=prior_cov_scale * np.eye(data_dim)
                ))

        return weights[:n_unique_sets], mu[:n_unique_sets], cov[:n_unique_sets]
    else:
        # MVN parameters
        prior_cov_df_adjusted = prior_cov_df * (data_dim - 1)

        mu = np.random.uniform(prior_mu[0], prior_mu[1], (n_sets, data_dim))
        cov = np.linalg.inv(sp.stats.wishart.rvs(
            df=prior_cov_df_adjusted,
            scale=prior_cov_scale * np.eye(data_dim),
            size=n_sets
        ))

        return mu[:n_unique_sets], cov[:n_unique_sets]


def sample_from_distribution(mu, cov, n_samples):
    """Sample from a multivariate normal distribution."""
    L = np.linalg.cholesky(cov)
    z = np.random.randn(n_samples, len(mu))
    return z @ L.T + mu


def sample_from_gmm(weights, mu, cov, n_samples):
    """Sample from a Gaussian Mixture Model.

    Args:
        weights: (k,) mixture weights
        mu: (k, d) component means
        cov: (k, d, d) component covariances
        n_samples: number of samples to draw

    Returns:
        samples: (n_samples, d) samples from the GMM
    """
    n_components = len(weights)
    d = mu.shape[1]

    # Sample component assignments
    assignments = np.random.choice(n_components, size=n_samples, p=weights)

    # Sample from each component
    samples = np.zeros((n_samples, d))
    for k in range(n_components):
        mask = assignments == k
        n_k = mask.sum()
        if n_k > 0:
            L = np.linalg.cholesky(cov[k])
            z = np.random.randn(n_k, d)
            samples[mask] = z @ L.T + mu[k]

    return samples


# ============================================================================
# Transport functions
# ============================================================================

def emb_transport(enc, gen, training_mu, training_cov, target_mu, target_cov,
                  source_idx, target_indices, device, set_size=1000, batch_size=500,
                  source_samples=None, nearest_idx=None):
    """Transport using EmbeddingEncoder.

    For embedding encoder, we sample from TRAINING distributions and use
    training distribution parameters for nearest neighbor lookup.

    Args:
        enc: Embedding encoder
        gen: Generator
        training_mu: (n_train, d) training distribution means
        training_cov: (n_train, d, d) training distribution covariances
        target_mu: (n_target, d) target distribution means
        target_cov: (n_target, d, d) target distribution covariances
        source_idx: Index of source distribution in training set
        target_indices: Indices of target distributions
        device: Torch device
        set_size: Number of samples per distribution
        batch_size: Batch size for processing
        source_samples: Optional pre-computed source samples (for caching)
        nearest_idx: Optional pre-computed nearest neighbor indices (for caching)

    Returns:
        generated: (n_targets, set_size, d) generated samples
    """
    from ot.gmm import dist_bures_squared

    # Use cached source samples or generate new ones
    if source_samples is None:
        src_mu, src_cov = training_mu[source_idx], training_cov[source_idx]
        source_samples = sample_from_distribution(src_mu, src_cov, set_size)

    source_samples_t = torch.from_numpy(source_samples).float().to(device)

    # Use cached nearest indices or compute
    if nearest_idx is None:
        w2 = dist_bures_squared(training_mu, target_mu, training_cov, target_cov)
        nearest_idx = np.argmin(w2, axis=0)

    # Get source embedding
    source_lat = enc(torch.tensor([source_idx]).long().to(device))

    # Process in batches
    n_targets = len(target_indices)
    all_generated = []

    for i in range(0, n_targets, batch_size):
        batch_end = min(i + batch_size, n_targets)
        batch_nearest = nearest_idx[i:batch_end]

        target_lat_batch = enc(torch.from_numpy(batch_nearest).long().to(device))

        batch_size_actual = batch_end - i
        source_expanded = source_samples_t.repeat(batch_size_actual, 1)
        source_lat_expanded = source_lat.repeat(batch_size_actual, 1)

        with torch.no_grad():
            generated_batch = gen.sample(source_expanded, source_lat_expanded, target_lat_batch)
            all_generated.append(generated_batch.cpu().numpy())

        torch.cuda.empty_cache()

    # Reshape from (n_targets * set_size, d) to (n_targets, set_size, d)
    flat_generated = np.concatenate(all_generated, axis=0)
    return flat_generated.reshape(n_targets, set_size, -1)


def gmm_emb_transport(enc, gen, training_weights, training_mu, training_cov,
                      target_weights, target_mu, target_cov,
                      source_idx, target_indices, device, set_size=1000, batch_size=500,
                      source_samples=None, nearest_idx=None):
    """Transport using EmbeddingEncoder for GMM distributions.

    For embedding encoder, we sample from TRAINING distributions and use
    GMM-OT distance for nearest neighbor lookup.

    Args:
        enc: Embedding encoder
        gen: Generator
        training_weights: (n_train, k) training GMM weights
        training_mu: (n_train, k, d) training GMM means
        training_cov: (n_train, k, d, d) training GMM covariances
        target_weights: (n_target, k) target GMM weights
        target_mu: (n_target, k, d) target GMM means
        target_cov: (n_target, k, d, d) target GMM covariances
        source_idx: Index of source distribution in training set
        target_indices: Indices of target distributions
        device: Torch device
        set_size: Number of samples per distribution
        batch_size: Batch size for processing
        source_samples: Optional pre-computed source samples (for caching)
        nearest_idx: Optional pre-computed nearest neighbor indices (for caching)

    Returns:
        generated: (n_targets, set_size, d) generated samples
    """
    # Use cached source samples or generate new ones
    if source_samples is None:
        src_weights = training_weights[source_idx]
        src_mu = training_mu[source_idx]
        src_cov = training_cov[source_idx]
        source_samples = sample_from_gmm(src_weights, src_mu, src_cov, set_size)

    source_samples_t = torch.from_numpy(source_samples).float().to(device)

    # Use cached nearest indices or compute
    if nearest_idx is None:
        print("    Computing pairwise GMM-OT distances for nearest neighbor lookup...")
        gmm_dist = pairwise_gmm_bures_distance(
            training_weights, training_mu, training_cov,
            target_weights, target_mu, target_cov
        )
        nearest_idx = np.argmin(gmm_dist, axis=0)

    # Get source embedding
    source_lat = enc(torch.tensor([source_idx]).long().to(device))

    # Process in batches
    n_targets = len(target_indices)
    all_generated = []

    for i in range(0, n_targets, batch_size):
        batch_end = min(i + batch_size, n_targets)
        batch_nearest = nearest_idx[i:batch_end]

        target_lat_batch = enc(torch.from_numpy(batch_nearest).long().to(device))

        batch_size_actual = batch_end - i
        source_expanded = source_samples_t.repeat(batch_size_actual, 1)
        source_lat_expanded = source_lat.repeat(batch_size_actual, 1)

        with torch.no_grad():
            generated_batch = gen.sample(source_expanded, source_lat_expanded, target_lat_batch)
            all_generated.append(generated_batch.cpu().numpy())

        torch.cuda.empty_cache()

    # Reshape from (n_targets * set_size, d) to (n_targets, set_size, d)
    flat_generated = np.concatenate(all_generated, axis=0)
    return flat_generated.reshape(n_targets, set_size, -1)


def dist_transport(enc, gen, ds, source_idx, target_idx, device, batch_size=500):
    """Transport using DistributionEncoder."""
    source_samples = torch.from_numpy(ds.data[source_idx]).float().to(device).unsqueeze(0)
    
    with torch.no_grad():
        # Get source embedding once
        source_lat = enc(source_samples)
        
        # Process targets in batches to avoid OOM
        n_targets = len(target_idx)
        all_generated = []
        
        for i in range(0, n_targets, batch_size):
            batch_end = min(i + batch_size, n_targets)
            target_idx_batch = target_idx[i:batch_end]
            
            # Load target batch
            target_samples_batch = torch.from_numpy(ds.data[target_idx_batch]).float().to(device)
            target_lat_batch = enc(target_samples_batch)
            
            # Expand source for this batch
            batch_size_actual = batch_end - i
            source_expanded = source_samples.reshape(-1, source_samples.shape[-1]).repeat(batch_size_actual, 1)
            source_lat_expanded = source_lat.repeat(batch_size_actual, 1)
            
            # Generate
            generated_batch = gen.sample(source_expanded, source_lat_expanded, target_lat_batch)
            all_generated.append(generated_batch.cpu().numpy())

            # Clear cache
            torch.cuda.empty_cache()

    # Reshape from (n_targets * set_size, d) to (n_targets, set_size, d)
    flat_generated = np.concatenate(all_generated, axis=0)
    set_size = ds.data.shape[1]
    return flat_generated.reshape(n_targets, set_size, -1)


# ============================================================================
# Model loading
# ============================================================================

def load_model(cfg, path, device, epoch=None):
    """Load encoder and generator from checkpoint.

    Args:
        cfg: Model configuration
        path: Experiment directory path
        device: Device to load model to
        epoch: If specified, load checkpoint_epoch_{epoch}.pt instead of best_model.pt
    """
    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])

    if epoch is not None:
        state = load_checkpoint(path, epoch)
    else:
        state = load_best_model(path)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])

    enc.eval().to(device)
    gen.eval().to(device)

    return enc, gen


def prepare_dataset(dataset_cfg):
    """Instantiate dataset."""
    return hydra.utils.instantiate(dataset_cfg)


def filter_experiments(configs, experiment_type='mvn', num_epochs=200, checkpoint_epoch=None,
                       generators=None):
    """Filter experiments by type, training epochs, and generator type.

    Args:
        configs: List of experiment configs
        experiment_type: 'mvn' or 'gmm'
        num_epochs: Required num_epochs in config
        checkpoint_epoch: If specified, only include experiments that have this checkpoint file
        generators: List of generator types to include. Options:
                   'flow_matching', 'mmd', 'swd', 'sinkhorn', 'energy', 'wasserstein'
                   If None, includes all generators.
    """
    import os
    filtered = []
    for c in configs:
        if experiment_type not in c['name']:
            continue
        # Handle configs that might not have 'training' key
        try:
            config_epochs = c['config']['training']['num_epochs']
        except (KeyError, TypeError):
            continue
        if config_epochs != num_epochs:
            continue
        if 'n_unique_sets' not in c['config'].get('dataset', {}):
            continue
        # Check if checkpoint file exists when specific epoch requested
        if checkpoint_epoch is not None:
            ckpt_path = os.path.join(c['dir'], f"checkpoint_epoch_{checkpoint_epoch}.pt")
            if not os.path.exists(ckpt_path):
                continue
        # Filter by generator type
        if generators is not None:
            gen_cfg = c['config'].get('generator', {})
            gen_target = gen_cfg.get('_target_', '')

            # Determine generator type
            if 'FlowMatchingGenerator' in gen_target:
                gen_type = 'flow_matching'
            elif 'EnergyGenerator' in gen_target:
                gen_type = 'energy'
            elif 'DirectGenerator' in gen_target:
                # For DirectGenerator, use loss_type
                gen_type = gen_cfg.get('loss_type', 'unknown')
            else:
                gen_type = 'unknown'

            if gen_type not in generators:
                continue

        filtered.append(c)
    return filtered


def get_experiment_info(cfg):
    """Extract key info from experiment config."""
    return {
        'encoder_type': 'embedding' if 'Embedding' in cfg['encoder'] else 'distribution',
        'generator_type': cfg['generator'],
        'n_unique_sets': cfg['config']['dataset']['n_unique_sets'],
        'seed': cfg['config'].get('seed', 42),
    }


# ============================================================================
# Main evaluation
# ============================================================================

def evaluate_embedding_model(enc, gen, training_mu, training_cov, ood_mu, ood_cov,
                             n_eval=None, source_idx=0, device='cuda',
                             set_size=1000, batch_size=500,
                             cached_training_samples=None, cached_ood_samples=None):
    """Evaluate embedding encoder model for MVN distributions.

    Args:
        enc: Embedding encoder
        gen: Generator
        training_mu: (n_train, d) training distribution means
        training_cov: (n_train, d, d) training distribution covariances
        ood_mu: (n_ood, d) OOD distribution means
        ood_cov: (n_ood, d, d) OOD distribution covariances
        n_eval: Number of in-dist and out-dist targets to evaluate (randomly subsampled).
                If None, uses all available.
        source_idx: Index of source distribution
        device: Torch device
        set_size: Number of samples per distribution
        batch_size: Batch size for processing
        cached_training_samples: Optional pre-computed training samples (n_train, set_size, d)
        cached_ood_samples: Optional pre-computed OOD samples (n_ood, set_size, d)

    Returns:
        Dictionary of evaluation metrics
    """
    from ot.gmm import dist_bures_squared

    results = {}
    n_unique_sets = len(training_mu)
    n_ood_total = len(ood_mu)

    # Pre-compute source samples (reused for all transport)
    src_mu, src_cov = training_mu[source_idx], training_cov[source_idx]
    source_samples = sample_from_distribution(src_mu, src_cov, set_size)

    # ========== IN-DISTRIBUTION EVALUATION ==========
    # Determine how many in-dist targets to evaluate
    n_in_dist = min(n_eval, n_unique_sets) if n_eval is not None else n_unique_sets
    if n_in_dist < n_unique_sets:
        in_dist_idx = np.random.choice(n_unique_sets, n_in_dist, replace=False)
    else:
        in_dist_idx = np.arange(n_unique_sets)

    print(f"  Evaluating in_dist ({len(in_dist_idx)} targets)...")

    # For in-dist, nearest neighbor is itself (no distance computation needed)
    in_dist_nearest = in_dist_idx

    generated = emb_transport(
        enc, gen, training_mu, training_cov,
        training_mu[in_dist_idx], training_cov[in_dist_idx],
        source_idx, in_dist_idx, device, set_size, batch_size,
        source_samples=source_samples, nearest_idx=in_dist_nearest
    )

    # Compare to training distributions (analytic)
    gen_mu = generated.mean(axis=1)
    gen_cov = batched_cov(generated)

    w2_squared = _dist_bures_squared_diag_numpy(
        gen_mu, training_mu[in_dist_idx], gen_cov, training_cov[in_dist_idx]
    )
    results['in_dist_w2_squared'] = w2_squared.mean()

    # Sample-based metrics
    print("    Computing sample-based metrics for in_dist...")
    n_sample_eval = min(100, len(in_dist_idx))
    sample_idx = np.random.choice(len(in_dist_idx), n_sample_eval, replace=False)

    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        idx = in_dist_idx[i]
        # Use cached samples or generate
        if cached_training_samples is not None:
            target_samples = cached_training_samples[idx]
        else:
            target_samples = sample_from_distribution(
                training_mu[idx], training_cov[idx], set_size
            )
        metrics = compute_all_metrics(generated[i], target_samples, device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])

    results['in_dist_mmd'] = np.mean(mmd_vals)
    results['in_dist_swd'] = np.mean(swd_vals)
    results['in_dist_energy'] = np.mean(energy_vals)

    # ========== OUT-OF-DISTRIBUTION EVALUATION ==========
    # Determine how many OOD targets to evaluate
    n_ood_eval = min(n_eval, n_ood_total) if n_eval is not None else n_ood_total
    if n_ood_eval < n_ood_total:
        ood_eval_idx = np.random.choice(n_ood_total, n_ood_eval, replace=False)
    else:
        ood_eval_idx = np.arange(n_ood_total)

    print(f"  Evaluating out_dist ({n_ood_eval} targets)...")

    # Subset OOD distributions
    ood_mu_eval = ood_mu[ood_eval_idx]
    ood_cov_eval = ood_cov[ood_eval_idx]

    # Pre-compute nearest neighbors for OOD
    ood_w2 = dist_bures_squared(training_mu, ood_mu_eval, training_cov, ood_cov_eval)
    ood_nearest = np.argmin(ood_w2, axis=0)

    generated = emb_transport(
        enc, gen, training_mu, training_cov,
        ood_mu_eval, ood_cov_eval,
        source_idx, np.arange(n_ood_eval), device, set_size, batch_size,
        source_samples=source_samples, nearest_idx=ood_nearest
    )

    # Compare to actual OOD targets (analytic)
    gen_mu = generated.mean(axis=1)
    gen_cov = batched_cov(generated)

    w2_squared = _dist_bures_squared_diag_numpy(gen_mu, ood_mu_eval, gen_cov, ood_cov_eval)
    results['out_dist_w2_squared'] = w2_squared.mean()

    # Sample-based metrics
    print("    Computing sample-based metrics for out_dist...")
    n_sample_eval = min(100, n_ood_eval)
    sample_idx = np.random.choice(n_ood_eval, n_sample_eval, replace=False)

    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        # Use cached samples or generate
        if cached_ood_samples is not None and ood_eval_idx[i] < len(cached_ood_samples):
            target_samples = cached_ood_samples[ood_eval_idx[i]]
        else:
            target_samples = sample_from_distribution(ood_mu_eval[i], ood_cov_eval[i], set_size)
        metrics = compute_all_metrics(generated[i], target_samples, device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])

    results['out_dist_mmd'] = np.mean(mmd_vals)
    results['out_dist_swd'] = np.mean(swd_vals)
    results['out_dist_energy'] = np.mean(energy_vals)

    # Store counts for tracking
    results['n_in_dist_eval'] = len(in_dist_idx)
    results['n_out_dist_eval'] = n_ood_eval

    return results


def evaluate_gmm_embedding_model(enc, gen, training_weights, training_mu, training_cov,
                                  ood_weights, ood_mu, ood_cov,
                                  n_eval=None, source_idx=0, device='cuda',
                                  set_size=1000, batch_size=500,
                                  cached_training_samples=None, cached_ood_samples=None):
    """Evaluate embedding encoder model for GMM distributions.

    Args:
        enc: Embedding encoder
        gen: Generator
        training_weights: (n_train, k) training GMM weights
        training_mu: (n_train, k, d) training GMM means
        training_cov: (n_train, k, d, d) training GMM covariances
        ood_weights: (n_ood, k) OOD GMM weights
        ood_mu: (n_ood, k, d) OOD GMM means
        ood_cov: (n_ood, k, d, d) OOD GMM covariances
        n_eval: Number of in-dist and out-dist targets to evaluate (randomly subsampled).
                If None, uses all available.
        source_idx: Index of source distribution
        device: Torch device
        set_size: Number of samples per distribution
        batch_size: Batch size for processing
        cached_training_samples: Optional pre-computed training samples (n_train, set_size, d)
        cached_ood_samples: Optional pre-computed OOD samples (n_ood, set_size, d)

    Returns:
        Dictionary of evaluation metrics
    """
    from ot.gmm import gmm_ot_loss

    results = {}
    n_unique_sets = len(training_mu)
    n_ood_total = len(ood_mu)

    # Pre-compute source samples (reused for all transport)
    src_weights = training_weights[source_idx]
    src_mu = training_mu[source_idx]
    src_cov = training_cov[source_idx]
    source_samples = sample_from_gmm(src_weights, src_mu, src_cov, set_size)

    # ========== IN-DISTRIBUTION EVALUATION ==========
    # Determine how many in-dist targets to evaluate
    n_in_dist = min(n_eval, n_unique_sets) if n_eval is not None else n_unique_sets
    if n_in_dist < n_unique_sets:
        in_dist_idx = np.random.choice(n_unique_sets, n_in_dist, replace=False)
    else:
        in_dist_idx = np.arange(n_unique_sets)

    print(f"  Evaluating in_dist ({len(in_dist_idx)} targets)...")

    # For in-dist, nearest neighbor is itself (no GMM-OT computation needed)
    in_dist_nearest = in_dist_idx

    generated = gmm_emb_transport(
        enc, gen, training_weights, training_mu, training_cov,
        training_weights[in_dist_idx], training_mu[in_dist_idx], training_cov[in_dist_idx],
        source_idx, in_dist_idx, device, set_size, batch_size,
        source_samples=source_samples, nearest_idx=in_dist_nearest
    )

    # Use cached samples or generate for metric computation
    print("    Computing metrics for in_dist...")
    n_sample_eval = min(100, len(in_dist_idx))
    sample_idx = np.random.choice(len(in_dist_idx), n_sample_eval, replace=False)

    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        idx = in_dist_idx[i]
        # Use cached samples or generate
        if cached_training_samples is not None:
            target_samples = cached_training_samples[idx]
        else:
            target_samples = sample_from_gmm(
                training_weights[idx], training_mu[idx], training_cov[idx], set_size
            )

        metrics = compute_all_metrics(generated[i], target_samples, device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])

    # No analytic W2 for GMM - use sample-based metrics only
    results['in_dist_w2_squared'] = np.nan
    results['in_dist_mmd'] = np.mean(mmd_vals)
    results['in_dist_swd'] = np.mean(swd_vals)
    results['in_dist_energy'] = np.mean(energy_vals)

    # ========== OUT-OF-DISTRIBUTION EVALUATION ==========
    # Determine how many OOD targets to evaluate
    n_ood_eval = min(n_eval, n_ood_total) if n_eval is not None else n_ood_total
    if n_ood_eval < n_ood_total:
        ood_eval_idx = np.random.choice(n_ood_total, n_ood_eval, replace=False)
    else:
        ood_eval_idx = np.arange(n_ood_total)

    print(f"  Evaluating out_dist ({n_ood_eval} targets)...")

    # Subset OOD distributions
    ood_weights_eval = ood_weights[ood_eval_idx]
    ood_mu_eval = ood_mu[ood_eval_idx]
    ood_cov_eval = ood_cov[ood_eval_idx]

    # Pre-compute nearest neighbors for OOD (training -> OOD)
    print("    Computing pairwise GMM-OT distances for OOD NN lookup...")
    ood_gmm_dist = pairwise_gmm_bures_distance(
        training_weights, training_mu, training_cov,
        ood_weights_eval, ood_mu_eval, ood_cov_eval
    )
    ood_nearest = np.argmin(ood_gmm_dist, axis=0)

    generated = gmm_emb_transport(
        enc, gen, training_weights, training_mu, training_cov,
        ood_weights_eval, ood_mu_eval, ood_cov_eval,
        source_idx, np.arange(n_ood_eval), device, set_size, batch_size,
        source_samples=source_samples, nearest_idx=ood_nearest
    )

    # Compute metrics
    print("    Computing metrics for out_dist...")
    n_sample_eval = min(100, n_ood_eval)
    sample_idx = np.random.choice(n_ood_eval, n_sample_eval, replace=False)

    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        # Use cached samples or generate
        if cached_ood_samples is not None and ood_eval_idx[i] < len(cached_ood_samples):
            target_samples = cached_ood_samples[ood_eval_idx[i]]
        else:
            target_samples = sample_from_gmm(ood_weights_eval[i], ood_mu_eval[i], ood_cov_eval[i], set_size)

        metrics = compute_all_metrics(generated[i], target_samples, device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])

    # No analytic W2 for GMM - use sample-based metrics only
    results['out_dist_w2_squared'] = np.nan
    results['out_dist_mmd'] = np.mean(mmd_vals)
    results['out_dist_swd'] = np.mean(swd_vals)
    results['out_dist_energy'] = np.mean(energy_vals)

    # Store counts for tracking
    results['n_in_dist_eval'] = len(in_dist_idx)
    results['n_out_dist_eval'] = n_ood_eval

    return results


def evaluate_distribution_model(enc, gen, ds, n_unique_sets,
                                n_eval=None, n_out_dist=10000, source_idx=0,
                                device='cuda', batch_size=500, is_gmm=False):
    """Evaluate distribution encoder model.

    Args:
        n_eval: Number of in-dist and out-dist targets to evaluate (randomly subsampled).
                If None, uses n_unique_sets for in-dist and n_out_dist for out-dist.
    """
    results = {}

    # In-distribution evaluation
    n_in_dist = min(n_eval, n_unique_sets) if n_eval is not None else n_unique_sets
    if n_in_dist < n_unique_sets:
        in_dist_idx = np.random.choice(n_unique_sets, n_in_dist, replace=False)
    else:
        in_dist_idx = np.arange(n_unique_sets)

    print(f"  Evaluating in_dist ({len(in_dist_idx)} targets)...")
    
    generated = dist_transport(enc, gen, ds, source_idx, in_dist_idx, device, batch_size)
    
    # For GMM, skip analytic Bures (mu has extra component dimension)
    # For MVN, compute analytic Bures distance
    if not is_gmm and hasattr(ds, 'mu') and ds.mu.ndim == 2:
        target_mu = ds.mu[in_dist_idx]
        target_cov = ds.cov[in_dist_idx]
        gen_mu = generated.mean(axis=1)
        gen_cov = batched_cov(generated)
        
        w2_squared = dist_bures_squared_diag(gen_mu, target_mu, gen_cov, target_cov)
        results['in_dist_w2_squared'] = w2_squared.mean()
    else:
        results['in_dist_w2_squared'] = np.nan  # Not applicable for GMM
    
    n_sample_eval = min(100, len(in_dist_idx))
    sample_idx = np.random.choice(len(in_dist_idx), n_sample_eval, replace=False)
    
    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        metrics = compute_all_metrics(generated[i], ds.data[in_dist_idx[i]], device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])
    
    results['in_dist_mmd'] = np.mean(mmd_vals)
    results['in_dist_swd'] = np.mean(swd_vals)
    results['in_dist_energy'] = np.mean(energy_vals)
    
    # Out-of-distribution evaluation
    n_ood_eval = min(n_eval, n_out_dist) if n_eval is not None else n_out_dist
    if n_ood_eval < n_out_dist:
        ood_subset = np.random.choice(n_out_dist, n_ood_eval, replace=False)
        out_dist_idx = n_unique_sets + ood_subset
    else:
        out_dist_idx = np.arange(n_unique_sets, n_unique_sets + n_out_dist)

    print(f"  Evaluating out_dist ({len(out_dist_idx)} targets)...")
    
    generated = dist_transport(enc, gen, ds, source_idx, out_dist_idx, device, batch_size)
    
    if not is_gmm and hasattr(ds, 'mu') and ds.mu.ndim == 2:
        target_mu = ds.mu[out_dist_idx]
        target_cov = ds.cov[out_dist_idx]
        gen_mu = generated.mean(axis=1)
        gen_cov = batched_cov(generated)
        
        w2_squared = dist_bures_squared_diag(gen_mu, target_mu, gen_cov, target_cov)
        results['out_dist_w2_squared'] = w2_squared.mean()
    else:
        results['out_dist_w2_squared'] = np.nan
    
    n_sample_eval = min(100, len(out_dist_idx))
    sample_idx = np.random.choice(len(out_dist_idx), n_sample_eval, replace=False)
    
    mmd_vals, swd_vals, energy_vals = [], [], []
    for i in sample_idx:
        metrics = compute_all_metrics(generated[i], ds.data[out_dist_idx[i]], device=device)
        mmd_vals.append(metrics['mmd'])
        swd_vals.append(metrics['swd'])
        energy_vals.append(metrics['energy'])
    
    results['out_dist_mmd'] = np.mean(mmd_vals)
    results['out_dist_swd'] = np.mean(swd_vals)
    results['out_dist_energy'] = np.mean(energy_vals)

    # Store counts for tracking
    results['n_in_dist_eval'] = len(in_dist_idx)
    results['n_out_dist_eval'] = len(out_dist_idx)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../outputs')
    parser.add_argument('--experiment', type=str, default='mvn', choices=['mvn', 'gmm'])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_path', type=str, default='eval_results.pkl')
    parser.add_argument('--batch_size', type=int, default=500, help='Batch size for transport')
    parser.add_argument('--n_out_dist', type=int, default=10000, help='Number of OOD distributions to generate')
    parser.add_argument('--set_size', type=int, default=1000, help='Samples per distribution')
    parser.add_argument('--checkpoint_epoch', type=int, default=None, help='Load specific epoch checkpoint instead of best_model (e.g., 1000)')
    parser.add_argument('--n_eval', type=int, default=None, help='Number of in-dist and out-dist targets to evaluate (randomly subsampled). If None, uses all available.')
    parser.add_argument('--generators', type=str, nargs='+', default=['flow_matching', 'mmd', 'swd'],
                        help='Generator types to evaluate. Options: flow_matching, mmd, swd, sinkhorn, energy, wasserstein. Default: flow_matching mmd swd')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if args.checkpoint_epoch is not None:
        print(f"Loading checkpoint_epoch_{args.checkpoint_epoch}.pt instead of best_model.pt")
    if args.n_eval is not None:
        print(f"Evaluating on {args.n_eval} in-dist and {args.n_eval} out-dist targets (randomly subsampled)")
    else:
        print(f"Evaluating on all in-dist targets and {args.n_out_dist} out-dist targets")
    print(f"Filtering generators: {args.generators}")

    # Load experiment configs (fast pre-filtered loading)
    print("Loading experiment configs...")
    configs = get_filtered_experiments_fast(args.output_dir, args.experiment, args.checkpoint_epoch)
    experiments = filter_experiments(configs, args.experiment, args.num_epochs, args.checkpoint_epoch,
                                     generators=args.generators)
    print(f"Found {len(experiments)} {args.experiment} experiments")
    
    if not experiments:
        print("No experiments found!")
        return
    
    # Prepare dataset for distribution encoder evaluation
    print("Preparing evaluation dataset...")
    dataset_cfg = copy.deepcopy(experiments[0]['config']['dataset'])
    max_n_unique = max(exp['config']['dataset']['n_unique_sets'] for exp in experiments)
    dataset_cfg['n_sets'] = max_n_unique + args.n_out_dist
    dataset_cfg['n_unique_sets'] = max_n_unique + args.n_out_dist
    dataset_cfg['set_size'] = args.set_size
    dataset_cfg['seed'] = 999  # Different seed for OOD
    ds = prepare_dataset(dataset_cfg)
    print(f"Dataset prepared: {len(ds)} sets")
    
    # Generate OOD distributions for embedding encoder evaluation
    print("Generating OOD distributions...")
    np.random.seed(888)
    data_shape = list(dataset_cfg.get('data_shape', [2]))
    data_dim = data_shape[0]
    prior_mu = list(dataset_cfg.get('prior_mu', [0, 5]))
    prior_cov_df = int(dataset_cfg.get('prior_cov_df', 10))
    prior_cov_scale = float(dataset_cfg.get('prior_cov_scale', 1.0))

    is_gmm = args.experiment == 'gmm'

    if is_gmm:
        # Generate GMM OOD parameters
        n_components = int(dataset_cfg.get('n_components', 3))
        dirichlet_alpha = float(dataset_cfg.get('dirichlet_alpha', 1.0))

        ood_weights = np.random.dirichlet(
            alpha=[dirichlet_alpha] * n_components,
            size=args.n_out_dist
        )
        ood_mu = np.random.uniform(
            prior_mu[0], prior_mu[1],
            (args.n_out_dist, n_components, data_dim)
        )
        ood_cov = np.zeros((args.n_out_dist, n_components, data_dim, data_dim))
        for i in range(args.n_out_dist):
            for j in range(n_components):
                ood_cov[i, j] = np.linalg.inv(sp.stats.wishart.rvs(
                    df=prior_cov_df,
                    scale=prior_cov_scale * np.eye(data_dim)
                ))
        print(f"  Generated {args.n_out_dist} OOD GMMs with {n_components} components")
    else:
        # Generate MVN OOD parameters
        prior_cov_df_adjusted = prior_cov_df * (data_dim - 1)
        ood_mu = np.random.uniform(prior_mu[0], prior_mu[1], (args.n_out_dist, data_dim))
        ood_cov = np.linalg.inv(sp.stats.wishart.rvs(
            df=prior_cov_df_adjusted, scale=prior_cov_scale * np.eye(data_dim), size=args.n_out_dist
        ))
        ood_weights = None  # Not used for MVN
        print(f"  Generated {args.n_out_dist} OOD MVN distributions")
    
    # Evaluate all models
    all_results = []
    
    for exp in tqdm(experiments, desc="Evaluating models"):
        info = get_experiment_info(exp)

        print(f"\nEvaluating: {info}")

        try:
            enc, gen = load_model(exp['config'], exp['dir'], device, epoch=args.checkpoint_epoch)

            if info['encoder_type'] == 'embedding':
                # Get training distribution parameters
                print("  Loading training distribution parameters...")

                if is_gmm:
                    # GMM embedding evaluation
                    training_weights, training_mu, training_cov = get_training_distribution_params(
                        exp['config'], info['n_unique_sets'], is_gmm=True
                    )

                    results = evaluate_gmm_embedding_model(
                        enc, gen,
                        training_weights, training_mu, training_cov,
                        ood_weights, ood_mu, ood_cov,
                        n_eval=args.n_eval,
                        device=device,
                        set_size=args.set_size,
                        batch_size=args.batch_size
                    )
                else:
                    # MVN embedding evaluation
                    training_mu, training_cov = get_training_distribution_params(
                        exp['config'], info['n_unique_sets'], is_gmm=False
                    )

                    results = evaluate_embedding_model(
                        enc, gen, training_mu, training_cov, ood_mu, ood_cov,
                        n_eval=args.n_eval,
                        device=device,
                        set_size=args.set_size,
                        batch_size=args.batch_size
                    )
            else:
                results = evaluate_distribution_model(
                    enc, gen, ds, info['n_unique_sets'],
                    n_eval=args.n_eval,
                    n_out_dist=args.n_out_dist,
                    device=device,
                    batch_size=args.batch_size,
                    is_gmm=is_gmm
                )
            
            results.update(info)
            results['experiment_dir'] = exp['dir']
            all_results.append(results)
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    
    # Save results
    print(f"\nSaving results to {args.save_path}...")
    with open(args.save_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Also save as DataFrame for easy analysis
    df = pd.DataFrame(all_results)
    csv_path = args.save_path.replace('.pkl', '.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # For GMM, show sample-based metrics (no analytic W2)
    # For MVN, show analytic W2
    if is_gmm:
        summary_cols = ['in_dist_mmd', 'in_dist_swd', 'out_dist_mmd', 'out_dist_swd']
    else:
        summary_cols = ['in_dist_w2_squared', 'out_dist_w2_squared', 'in_dist_mmd', 'out_dist_mmd']

    print(df.groupby(['encoder_type', 'generator_type', 'n_unique_sets'])[
        summary_cols
    ].mean().round(4))


if __name__ == '__main__':
    main()
