"""
Test utilities for GMM distance functions.

Run with: python test_gmm_distance.py
"""

import sys
from pathlib import Path

# Add parent dir to path
parent_dir = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, parent_dir)

import numpy as np
import time
from ot.gmm import gmm_ot_loss


def pairwise_gmm_bures_distance_reference(weights1, mu1, cov1, weights2, mu2, cov2):
    """
    Reference implementation using gmm_ot_loss for each pair.
    Used for correctness testing.

    Args:
        weights1: (n1, k) mixture weights
        mu1: (n1, k, d) component means
        cov1: (n1, k, d, d) component covariances
        weights2: (n2, k) mixture weights
        mu2: (n2, k, d) component means
        cov2: (n2, k, d, d) component covariances
    """
    n1 = len(weights1)
    n2 = len(weights2)
    result = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            # gmm_ot_loss signature: (m_s, m_t, C_s, C_t, w_s, w_t)
            result[i, j] = gmm_ot_loss(
                mu1[i], mu2[j],
                cov1[i], cov2[j],
                weights1[i], weights2[j]
            )
    return result


def test_pairwise_gmm_bures_distance():
    """Test that optimized implementation matches reference."""
    from evaluate_mvn_gmm import pairwise_gmm_bures_distance

    print("Testing pairwise_gmm_bures_distance...")
    print("="*60)

    # Generate test data
    np.random.seed(42)
    n1, n2, k, d = 10, 15, 3, 2

    weights1 = np.random.dirichlet([1.0] * k, size=n1)
    weights2 = np.random.dirichlet([1.0] * k, size=n2)
    mu1 = np.random.randn(n1, k, d)
    mu2 = np.random.randn(n2, k, d)

    # Generate positive definite covariances
    cov1 = np.zeros((n1, k, d, d))
    cov2 = np.zeros((n2, k, d, d))
    for i in range(n1):
        for j in range(k):
            A = np.random.randn(d, d)
            cov1[i, j] = A @ A.T + 0.1 * np.eye(d)
    for i in range(n2):
        for j in range(k):
            A = np.random.randn(d, d)
            cov2[i, j] = A @ A.T + 0.1 * np.eye(d)

    # Test correctness (serial mode for small problems)
    print("\n1. Testing correctness...")
    ref_result = pairwise_gmm_bures_distance_reference(weights1, mu1, cov1, weights2, mu2, cov2)
    opt_result = pairwise_gmm_bures_distance(weights1, mu1, cov1, weights2, mu2, cov2, n_jobs=1)

    max_diff = np.abs(ref_result - opt_result).max()
    mean_diff = np.abs(ref_result - opt_result).mean()

    print(f"   Max difference: {max_diff:.2e}")
    print(f"   Mean difference: {mean_diff:.2e}")

    passed = max_diff < 1e-6
    print(f"   Result: {'PASS' if passed else 'FAIL'}")

    # Test speed with larger data
    print("\n2. Testing speed (n1=1000, n2=100)...")
    n1_speed, n2_speed = 1000, 100

    weights1_large = np.random.dirichlet([1.0] * k, size=n1_speed)
    weights2_large = np.random.dirichlet([1.0] * k, size=n2_speed)
    mu1_large = np.random.randn(n1_speed, k, d)
    mu2_large = np.random.randn(n2_speed, k, d)
    cov1_large = np.zeros((n1_speed, k, d, d))
    cov2_large = np.zeros((n2_speed, k, d, d))
    for i in range(n1_speed):
        for j in range(k):
            A = np.random.randn(d, d)
            cov1_large[i, j] = A @ A.T + 0.1 * np.eye(d)
    for i in range(n2_speed):
        for j in range(k):
            A = np.random.randn(d, d)
            cov2_large[i, j] = A @ A.T + 0.1 * np.eye(d)

    # Reference timing
    start = time.time()
    ref_large = pairwise_gmm_bures_distance_reference(weights1_large, mu1_large, cov1_large,
                                                       weights2_large, mu2_large, cov2_large)
    ref_time = time.time() - start
    print(f"   Reference (gmm_ot_loss loop): {ref_time:.1f}s")

    # Serial optimized timing
    start = time.time()
    opt_serial = pairwise_gmm_bures_distance(weights1_large, mu1_large, cov1_large,
                                              weights2_large, mu2_large, cov2_large, n_jobs=1)
    serial_time = time.time() - start
    print(f"   Optimized (serial): {serial_time:.1f}s, speedup: {ref_time/serial_time:.1f}x")

    # Parallel optimized timing
    start = time.time()
    opt_parallel = pairwise_gmm_bures_distance(weights1_large, mu1_large, cov1_large,
                                                weights2_large, mu2_large, cov2_large, n_jobs=8)
    parallel_time = time.time() - start
    print(f"   Optimized (8 workers): {parallel_time:.1f}s, speedup: {ref_time/parallel_time:.1f}x")

    # Verify parallel results match serial
    max_diff_parallel = np.abs(opt_serial - opt_parallel).max()
    print(f"   Parallel vs serial max diff: {max_diff_parallel:.2e}")

    print("\n" + "="*60)
    return passed and max_diff_parallel < 1e-6


if __name__ == '__main__':
    passed = test_pairwise_gmm_bures_distance()
    sys.exit(0 if passed else 1)
