"""
Edit distance-based OT coupling for protein sequences.
Uses rapidfuzz for fast parallel edit distance computation.
"""

import numpy as np
from typing import Optional, List
import multiprocessing

from rapidfuzz import process
from rapidfuzz.distance import Levenshtein as Lev

from .ot import BaseOTCollate


def compute_edit_distance_matrix(
    sequences_a: List[str],
    sequences_b: List[str],
    normalize: bool = True,
    workers: int = -1
) -> np.ndarray:
    """
    Compute pairwise edit distance matrix between two lists of sequences.

    Uses rapidfuzz for fast parallel computation.

    Args:
        sequences_a: List of source sequences
        sequences_b: List of target sequences
        normalize: If True, normalize distances by max sequence length
        workers: Number of parallel workers (-1 for all CPUs)

    Returns:
        np.ndarray of shape (len(sequences_a), len(sequences_b)) with distances
    """
    if workers == -1:
        workers = multiprocessing.cpu_count()

    n_a = len(sequences_a)
    n_b = len(sequences_b)

    # Compute pairwise Levenshtein distances using rapidfuzz cdist
    distances = process.cdist(
        sequences_a,
        sequences_b,
        scorer=Lev.distance,
        workers=workers
    )

    # Convert to float64 first to allow normalization
    distances = np.array(distances, dtype=np.float64)

    if normalize:
        # Normalize by maximum length of each pair
        for i in range(n_a):
            for j in range(n_b):
                max_len = max(len(sequences_a[i]), len(sequences_b[j]))
                if max_len > 0:
                    distances[i, j] = distances[i, j] / max_len

    return distances


class EditDistanceOTCollate(BaseOTCollate):
    """
    OT coupling based on edit distance for protein sequences.

    This coupling computes pairwise edit distances between source and target
    protein sequences, then uses optimal transport to find an optimal pairing.
    """

    def __init__(
        self,
        ot_type: str = 'sinkhorn',
        ot_params: Optional[dict] = None,
        normalize_distance: bool = True,
        workers: int = -1,
        replace: bool = False,
        sequence_key: str = 'raw_texts',
        debug: bool = False
    ):
        """
        Args:
            ot_type: Type of OT solver ('sinkhorn', 'emd', 'bregman')
            ot_params: Additional parameters for OT solver
            normalize_distance: Whether to normalize edit distances by sequence length
            workers: Number of parallel workers for edit distance computation (-1 for all CPUs)
            replace: Whether to sample with replacement from OT plan
            sequence_key: Key in sample dict containing the raw text sequences
            debug: Whether to print debug information about pairings
        """
        super(EditDistanceOTCollate, self).__init__(ot_type, ot_params, replace)
        self.normalize_distance = normalize_distance
        self.workers = workers
        self.sequence_key = sequence_key
        self.debug = debug
        self._call_count = 0

    def compute_cost_matrix(self, source_samples, target_samples) -> np.ndarray:
        """Compute cost matrix using edit distance on raw text sequences."""
        # Extract raw text sequences
        if isinstance(source_samples, dict) and self.sequence_key in source_samples:
            source_seqs = source_samples[self.sequence_key]
            target_seqs = target_samples[self.sequence_key]
        else:
            raise ValueError(
                f"Cannot find sequence key '{self.sequence_key}' in samples. "
                "Make sure your dataset provides raw text sequences."
            )

        cost = compute_edit_distance_matrix(
            source_seqs,
            target_seqs,
            normalize=self.normalize_distance,
            workers=self.workers
        )

        # Store for debug output in solve_ot
        self._current_source_seqs = source_seqs
        self._current_target_seqs = target_seqs
        self._current_cost = cost

        return cost

    def solve_ot(self, cost, set_size, ot_type, ot_params):
        """Override to add debug output."""
        source_idx, target_idx = super().solve_ot(cost, set_size, ot_type, ot_params)

        if self.debug and self._call_count < 3:  # Only print first 3 batches
            self._call_count += 1
            import sys

            print(f"\n{'='*60}", file=sys.stderr)
            print(f"Edit Distance OT Coupling - Batch {self._call_count}", file=sys.stderr)
            print(f"{'='*60}", file=sys.stderr)

            # Show a few example pairings
            n_show = min(5, len(source_idx))
            print(f"\nShowing {n_show} example pairings:", file=sys.stderr)
            print("-" * 60, file=sys.stderr)

            for i in range(n_show):
                src_i = source_idx[i]
                tgt_i = target_idx[i]
                src_seq = self._current_source_seqs[src_i]
                tgt_seq = self._current_target_seqs[tgt_i]
                dist = self._current_cost[src_i, tgt_i]

                # Compute raw edit distance for display
                raw_dist = Lev.distance(src_seq, tgt_seq)

                print(f"\nPair {i+1}:", file=sys.stderr)
                print(f"  Source[{src_i}]: {src_seq[:40]}{'...' if len(src_seq) > 40 else ''}", file=sys.stderr)
                print(f"  Target[{tgt_i}]: {tgt_seq[:40]}{'...' if len(tgt_seq) > 40 else ''}", file=sys.stderr)
                print(f"  Edit distance: {raw_dist} (normalized: {dist:.3f})", file=sys.stderr)

            # Show cost matrix statistics
            print(f"\nCost matrix statistics:", file=sys.stderr)
            print(f"  Min: {self._current_cost.min():.3f}", file=sys.stderr)
            print(f"  Max: {self._current_cost.max():.3f}", file=sys.stderr)
            print(f"  Mean: {self._current_cost.mean():.3f}", file=sys.stderr)

            # Show distribution of paired distances vs random
            paired_dists = [self._current_cost[source_idx[i], target_idx[i]] for i in range(len(source_idx))]
            print(f"\nPaired distances: min={min(paired_dists):.3f}, max={max(paired_dists):.3f}, mean={np.mean(paired_dists):.3f}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr, flush=True)

        return source_idx, target_idx
