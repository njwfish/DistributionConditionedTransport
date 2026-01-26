"""
Debug script to understand why oracle performance is so bad.
"""
import sys
from pathlib import Path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

import torch
import numpy as np
import hydra
import scipy as sp

from utils.experiment_utils import get_all_experiments_info, load_best_model
from generator.losses import mmd, sliced_wasserstein_distance


def load_model(cfg, path, device):
    """Load encoder and generator from checkpoint."""
    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])

    state = load_best_model(path)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])

    enc.eval().to(device)
    gen.eval().to(device)

    return enc, gen


def generate_test_data(n_samples=1000, data_dim=2, seed=42):
    """Generate source and target distributions."""
    np.random.seed(seed)

    # Source distribution: N(mu, Sigma)
    source_mu = np.array([1.0, 1.0])
    source_cov = np.array([[1.0, 0.3], [0.3, 1.0]])

    source_samples = np.random.multivariate_normal(source_mu, source_cov, n_samples)

    # Transformation Y = Ax + b
    np.random.seed(seed + 1000)  # Match training
    H = np.random.randn(data_dim, data_dim)
    Q, R = np.linalg.qr(H)
    A = Q * np.sign(np.diag(R))
    b = np.random.randn(data_dim)

    # Target samples
    target_samples = source_samples @ A.T + b
    np.random.shuffle(target_samples)

    # Target distribution parameters (for reference)
    target_mu = A @ source_mu + b
    target_cov = A @ source_cov @ A.T

    return source_samples, target_samples, A, b, source_mu, source_cov, target_mu, target_cov


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load configs
    configs = get_all_experiments_info('outputs')

    # Find any-to-any MVN sinkhorn model (seed 42, n_unique_sets=10000, 1000 epochs)
    unsup_exp = None
    for c in configs:
        if 'mvn' not in c['name']:
            continue
        if 'supervised' in c['name']:
            continue
        if c['config']['training']['num_epochs'] != 1000:
            continue
        if c['config']['dataset'].get('n_unique_sets') != 10000:
            continue
        if c['config'].get('seed', 42) != 42:
            continue
        gen_cfg = c['config']['generator']
        if gen_cfg.get('loss_type', '') == 'sinkhorn':
            unsup_exp = c
            break

    if unsup_exp is None:
        print("Could not find unsupervised experiment!")
        return

    print(f"Found: {unsup_exp['name']}")
    print(f"Dir: {unsup_exp['dir']}")

    # Load model
    enc, gen = load_model(unsup_exp['config'], unsup_exp['dir'], device)

    # Generate test data
    source_samples, target_samples, A, b, source_mu, source_cov, target_mu, target_cov = generate_test_data(
        n_samples=1000, seed=42
    )

    print(f"\nSource mu: {source_mu}")
    print(f"Target mu: {target_mu}")
    print(f"A:\n{A}")
    print(f"b: {b}")

    # Convert to tensors
    source_t = torch.from_numpy(source_samples).float().to(device)
    target_t = torch.from_numpy(target_samples).float().to(device)

    # Get embeddings
    with torch.no_grad():
        source_lat = enc(source_t.unsqueeze(0))  # (1, latent_dim)
        target_lat = enc(target_t.unsqueeze(0))  # (1, latent_dim)

    print(f"\nSource latent shape: {source_lat.shape}")
    print(f"Source latent mean: {source_lat.mean().item():.4f}, std: {source_lat.std().item():.4f}")
    print(f"Target latent mean: {target_lat.mean().item():.4f}, std: {target_lat.std().item():.4f}")
    print(f"Latent distance: {(source_lat - target_lat).norm().item():.4f}")

    # Transport using oracle (true target latent)
    with torch.no_grad():
        n_samples = source_t.shape[0]
        source_lat_exp = source_lat.expand(n_samples, -1)
        target_lat_exp = target_lat.expand(n_samples, -1)

        transported = gen.sample(source_t, source_lat_exp, target_lat_exp)
        transported = transported.squeeze()

    print(f"\nTransported shape: {transported.shape}")
    print(f"Transported mean: {transported.mean(dim=0).cpu().numpy()}")
    print(f"Expected target mean: {target_mu}")

    # Compute metrics
    mmd_val = mmd(transported, target_t).item()
    swd_val = sliced_wasserstein_distance(transported, target_t, n_projections=100).item()

    print(f"\n=== Oracle Transport Metrics ===")
    print(f"MMD: {mmd_val:.6f}")
    print(f"SWD: {swd_val:.6f}")

    # Also test: what if we transport to a DIFFERENT random distribution (what the model was trained on)
    print("\n=== Testing transport to independent target (what model was trained on) ===")
    np.random.seed(999)
    indep_target_mu = np.array([3.0, 2.0])
    indep_target_cov = np.array([[0.8, -0.2], [-0.2, 1.2]])
    indep_target_samples = np.random.multivariate_normal(indep_target_mu, indep_target_cov, 1000)
    indep_target_t = torch.from_numpy(indep_target_samples).float().to(device)

    with torch.no_grad():
        indep_target_lat = enc(indep_target_t.unsqueeze(0))
        indep_target_lat_exp = indep_target_lat.expand(n_samples, -1)

        transported_indep = gen.sample(source_t, source_lat_exp, indep_target_lat_exp)
        transported_indep = transported_indep.squeeze()

    mmd_indep = mmd(transported_indep, indep_target_t).item()
    swd_indep = sliced_wasserstein_distance(transported_indep, indep_target_t, n_projections=100).item()

    print(f"Independent target mean: {indep_target_mu}")
    print(f"Transported mean: {transported_indep.mean(dim=0).cpu().numpy()}")
    print(f"MMD: {mmd_indep:.6f}")
    print(f"SWD: {swd_indep:.6f}")

    # Baseline: MMD between two samples from same distribution
    print("\n=== Baseline: same distribution ===")
    target_samples2 = np.random.multivariate_normal(target_mu, target_cov, 1000)
    target_t2 = torch.from_numpy(target_samples2).float().to(device)
    mmd_baseline = mmd(target_t, target_t2).item()
    print(f"MMD between two samples from target dist: {mmd_baseline:.6f}")

    # Key test: Is the encoder embedding the Y=Ax+b target correctly?
    print("\n=== Encoder embedding check ===")
    # Fresh sample from the target distribution (not transformed, but same parameters)
    fresh_target_samples = np.random.multivariate_normal(target_mu, target_cov, 1000)
    fresh_target_t = torch.from_numpy(fresh_target_samples).float().to(device)

    with torch.no_grad():
        fresh_target_lat = enc(fresh_target_t.unsqueeze(0))

    print(f"Y=Ax+b target latent mean: {target_lat.mean().item():.4f}, std: {target_lat.std().item():.4f}")
    print(f"Fresh target latent mean: {fresh_target_lat.mean().item():.4f}, std: {fresh_target_lat.std().item():.4f}")
    print(f"Distance between embeddings: {(target_lat - fresh_target_lat).norm().item():.4f}")

    # Transport to fresh target (should work if encoder is consistent)
    with torch.no_grad():
        fresh_target_lat_exp = fresh_target_lat.expand(n_samples, -1)
        transported_fresh = gen.sample(source_t, source_lat_exp, fresh_target_lat_exp)
        transported_fresh = transported_fresh.squeeze()

    mmd_fresh = mmd(transported_fresh, fresh_target_t).item()
    print(f"Transport to fresh target MMD: {mmd_fresh:.6f}")

    # What about: transport to Y=Ax+b target using fresh target embedding?
    print("\n=== Cross-check: use fresh embedding to transport to original Y=Ax+b target ===")
    mmd_cross = mmd(transported_fresh, target_t).item()
    print(f"MMD(transported_fresh, original_target): {mmd_cross:.6f}")

    # And the original oracle with original target embedding
    mmd_orig = mmd(transported, target_t).item()
    print(f"MMD(transported_orig, original_target): {mmd_orig:.6f}")

    # Check actual sample statistics
    print("\n=== Sample statistics ===")
    print(f"Original target samples mean: {target_samples.mean(axis=0)}")
    print(f"Original target samples cov:\n{np.cov(target_samples.T)}")
    print(f"Fresh target samples mean: {fresh_target_samples.mean(axis=0)}")
    print(f"Fresh target samples cov:\n{np.cov(fresh_target_samples.T)}")
    print(f"Theoretical target cov:\n{target_cov}")


    # KEY TEST: The Y=Ax+b target has mean outside [0,5] support!
    print("\n=== KEY INSIGHT: Target mean is OUTSIDE training support ===")
    print(f"Training support: [0, 5]^2")
    print(f"Y=Ax+b target mean: {target_mu}")
    print(f"Target mean[0] = {target_mu[0]:.2f} < 0 = OUTSIDE SUPPORT!")

    # Test transport to a target IN the support with similar covariance
    print("\n=== Test: transport to target IN support ===")
    in_support_mu = np.array([2.5, 2.0])  # Well inside [0, 5]
    in_support_cov = target_cov  # Same covariance
    in_support_samples = np.random.multivariate_normal(in_support_mu, in_support_cov, 1000)
    in_support_t = torch.from_numpy(in_support_samples).float().to(device)

    with torch.no_grad():
        in_support_lat = enc(in_support_t.unsqueeze(0))
        in_support_lat_exp = in_support_lat.expand(n_samples, -1)
        transported_in_support = gen.sample(source_t, source_lat_exp, in_support_lat_exp)
        transported_in_support = transported_in_support.squeeze()

    print(f"In-support target mean: {in_support_mu}")
    print(f"Transported mean: {transported_in_support.mean(dim=0).cpu().numpy()}")
    mmd_in_support = mmd(transported_in_support, in_support_t).item()
    swd_in_support = sliced_wasserstein_distance(transported_in_support, in_support_t, n_projections=100).item()
    print(f"MMD: {mmd_in_support:.6f}")
    print(f"SWD: {swd_in_support:.6f}")


if __name__ == '__main__':
    main()
