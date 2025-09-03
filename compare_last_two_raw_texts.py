import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

try:
    from Bio import pairwise2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Biopython is required. Please install with: pip install biopython"
    ) from exc


def alignment_distance(seq1: str, seq2: str) -> float:
    """Compute 1 - identity via global alignment (globalxx) as described.

    Identity is normalized by max(len(seq1), len(seq2)).
    """
    if not seq1 and not seq2:
        return 0.0
    score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
    length = max(len(seq1), len(seq2))
    identity = float(score) / float(length) if length > 0 else 0.0
    return 1.0 - identity


def load_last_two_raw_texts(dataset_path: Path) -> Tuple[List[str], List[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    data = torch.load(str(dataset_path), map_location="cpu")
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Expected a list with at least two elements in the dataset file.")

    last = data[-1]
    second_last = data[-2]

    try:
        list_a = list(second_last["raw_texts"])  # type: ignore[index]
        list_b = list(last["raw_texts"])  # type: ignore[index]
    except Exception as exc:
        raise KeyError(
            "Could not find 'raw_texts' lists in the last two dataset elements."
        ) from exc

    # Ensure items are strings
    list_a = [str(x) for x in list_a]
    list_b = [str(x) for x in list_b]
    return list_a, list_b


def compute_distance_matrix(
    seqs_a: List[str], seqs_b: List[str], show_progress: bool = True
) -> np.ndarray:
    n = len(seqs_a)
    m = len(seqs_b)
    distances = np.zeros((n, m), dtype=np.float64)

    total = n * m
    idx = 0
    start_time = time.time()
    for i, sa in enumerate(seqs_a):
        for j, sb in enumerate(seqs_b):
            d = alignment_distance(sa, sb)
            distances[i, j] = d
            idx += 1
            if show_progress and (idx % 10 == 0 or idx == total):
                elapsed = time.time() - start_time
                rate = idx / max(elapsed, 1e-9)
                print(
                    f"Computed {idx}/{total} distances (current ~{rate:.2f} pairs/sec)",
                    end="\r",
                    file=sys.stderr,
                )
    if show_progress:
        print(file=sys.stderr)
    return distances


def optimal_pairing_scipy(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError("SciPy not available") from exc

    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    return row_ind.astype(int), col_ind.astype(int), total


def optimal_pairing_dp(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Exact optimal pairing via DP over column subsets.

    Works when number of columns is reasonably small (<= 22 recommended).
    Returns row indices [0..n-1] and chosen column indices, plus total cost.
    If rows > cols, the input cost is expected to be transposed by caller.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return np.array([], dtype=int), np.array([], dtype=int), 0.0

    if n > m:
        # Transpose to ensure n <= m, then map back
        col_idx, row_idx, total = optimal_pairing_dp(cost.T)
        return row_idx, col_idx, total

    if m > 22:
        raise RuntimeError(
            "DP fallback requires <= 22 columns. Install SciPy for larger problems."
        )

    # dp maps mask -> best cost after assigning k rows, where k == popcount(mask)
    dp = {0: 0.0}
    parents: List[dict] = []  # per row: mask -> (prev_mask, chosen_col)

    for row in range(n):
        next_dp = {}
        parent_stage = {}
        for mask, curr_cost in dp.items():
            # Try assigning row to any available column j
            for j in range(m):
                if (mask >> j) & 1:
                    continue
                new_mask = mask | (1 << j)
                new_cost = curr_cost + float(cost[row, j])
                prev_best = next_dp.get(new_mask)
                if prev_best is None or new_cost < prev_best:
                    next_dp[new_mask] = new_cost
                    parent_stage[new_mask] = (mask, j)
        dp = next_dp
        parents.append(parent_stage)

    # Among masks with exactly n bits set, choose minimal cost
    best_mask = None
    best_cost = float("inf")
    for mask, val in dp.items():
        if mask.bit_count() == n and val < best_cost:
            best_cost = val
            best_mask = mask

    if best_mask is None:
        raise RuntimeError("No assignment found in DP fallback.")

    # Reconstruct assignment
    assigned_cols = [0] * n
    mask = best_mask
    for row in range(n - 1, -1, -1):
        prev_mask, col = parents[row][mask]
        assigned_cols[row] = col
        mask = prev_mask

    row_ind = np.arange(n, dtype=int)
    col_ind = np.array(assigned_cols, dtype=int)
    return row_ind, col_ind, float(best_cost)


def compute_optimal_pairing(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    # Prefer SciPy if available
    try:
        return optimal_pairing_scipy(cost)
    except Exception:
        return optimal_pairing_dp(cost)


def summarize_matrix(dist: np.ndarray) -> None:
    n, m = dist.shape
    flat = dist.ravel()
    print("\n=== Distance Matrix Summary ===")
    print(f"Shape: {n} x {m}")
    print(f"Mean: {float(np.mean(flat)):.6f}")
    print(f"Median: {float(np.median(flat)):.6f}")
    print(f"Min: {float(np.min(flat)):.6f}")
    print(f"Max: {float(np.max(flat)):.6f}")
    for p in (10, 25, 75, 90):
        val = float(np.percentile(flat, p))
        print(f"P{p}: {val:.6f}")

    # Row/column nearest-neighbor summaries
    row_min = np.min(dist, axis=1) if n > 0 else np.array([])
    col_min = np.min(dist, axis=0) if m > 0 else np.array([])
    if row_min.size:
        print(
            f"Row-wise min distances: mean={float(np.mean(row_min)):.6f}, "
            f"median={float(np.median(row_min)):.6f}, min={float(np.min(row_min)):.6f}, max={float(np.max(row_min)):.6f}"
        )
    if col_min.size:
        print(
            f"Col-wise min distances: mean={float(np.mean(col_min)):.6f}, "
            f"median={float(np.median(col_min)):.6f}, min={float(np.min(col_min)):.6f}, max={float(np.max(col_min)):.6f}"
        )

    # Fractions below some thresholds (intuition: lower distance -> higher similarity)
    for thr in (0.05, 0.10, 0.20, 0.30):
        frac = float(np.mean(flat <= thr))
        print(f"Fraction <= {thr:.2f}: {frac:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Load dataset, extract last two raw_texts lists, compute NxM alignment "
            "distance matrix and summary metrics, including optimal pairing metrics."
        )
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=(
            "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/"
            "data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100_filtered_test.pt"
        ),
        help="Absolute path to the .pt dataset file to load.",
    )
    parser.add_argument(
        "--print_matrix",
        action="store_true",
        help="Print the full NxM matrix (careful if large).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    seqs_a, seqs_b = load_last_two_raw_texts(dataset_path)
    n, m = len(seqs_a), len(seqs_b)
    print(f"Loaded last two raw_texts lists with sizes: N={n}, M={m}")

    # Compute NxM distance matrix
    start = time.time()
    dist = compute_distance_matrix(seqs_a, seqs_b, show_progress=True)
    elapsed = time.time() - start
    print(f"Computed distance matrix in {elapsed:.2f}s")

    # Optionally print matrix
    if args.print_matrix:
        with np.printoptions(precision=3, suppress=True, linewidth=120):
            print("\nDistance matrix (N x M):")
            print(dist)

    # Summary stats over all elements
    summarize_matrix(dist)

    # Optimal pairing metrics (min sum of distances over min(N, M) pairings)
    print("\n=== Optimal Pairing (Hungarian/DP) ===")
    try:
        row_ind, col_ind, total_cost = compute_optimal_pairing(dist)
        matched_costs = dist[row_ind, col_ind]
        mean_matched = float(np.mean(matched_costs)) if matched_costs.size else 0.0
        print(f"Number of pairs: {len(row_ind)}")
        print(f"Total distance (optimal pairing): {total_cost:.6f}")
        print(f"Mean distance over optimal pairing: {mean_matched:.6f}")
        print(
            f"Matched distances: min={float(np.min(matched_costs)):.6f}, "
            f"median={float(np.median(matched_costs)):.6f}, "
            f"max={float(np.max(matched_costs)):.6f}"
        )
    except Exception as exc:
        print(
            "Could not compute optimal pairing without SciPy and problem too large for DP.\n"
            "Please install SciPy: pip install scipy",
            file=sys.stderr,
        )
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()


