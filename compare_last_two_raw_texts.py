import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from bisect import bisect_left
from collections import defaultdict

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

try:
    from Bio import pairwise2  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Biopython is required. Please install with: pip install biopython"
    ) from exc


# Lightweight slow-path logger
_SLOW_LOG_PATH = \
    "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/slow_logger.log"

def slow_log(msg: str) -> None:
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(_SLOW_LOG_PATH, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        # Never crash on logging
        pass



# Fast Levenshtein distance helpers (use C-accelerated libs when available)
try:  # rapidfuzz preferred for thresholded distance
    from rapidfuzz.distance import Levenshtein as _rf_lev  # type: ignore

    def _levenshtein_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
        if max_distance is None:
            return int(_rf_lev.distance(a, b))
        # score_cutoff means: return > cutoff if distance exceeds cutoff
        d = _rf_lev.distance(a, b, score_cutoff=max_distance)
        # rapidfuzz returns max_distance + 1 when above cutoff
        return int(d)
except Exception:
    try:
        from Levenshtein import distance as _lev_c  # type: ignore

        def _levenshtein_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
            # python-Levenshtein doesn't support threshold; compute full distance
            return int(_lev_c(a, b))
    except Exception:
        def _levenshtein_distance(a: str, b: str, max_distance: Optional[int] = None) -> int:
            # Lightweight Python DP with early-abort on threshold
            if a == b:
                return 0
            la, lb = len(a), len(b)
            if la == 0:
                return lb
            if lb == 0:
                return la
            # Ensure a is the shorter to reduce memory
            if la > lb:
                a, b = b, a
                la, lb = lb, la

            prev = list(range(lb + 1))
            # If we have a threshold, we can early abort when all cells in a row exceed it
            for i in range(1, la + 1):
                ca = a[i - 1]
                cur = [i] + [0] * lb
                # Standard DP recurrence
                row_min = cur[0]
                for j in range(1, lb + 1):
                    cost = 0 if ca == b[j - 1] else 1
                    deletion = prev[j] + 1
                    insertion = cur[j - 1] + 1
                    substitution = prev[j - 1] + cost
                    cur[j] = deletion if deletion < insertion else insertion
                    if substitution < cur[j]:
                        cur[j] = substitution
                    if cur[j] < row_min:
                        row_min = cur[j]
                if max_distance is not None and row_min > max_distance:
                    # Exceeded threshold everywhere in this row
                    return max_distance + 1
                prev = cur
            return prev[lb]


def compute_membership_and_min_levenshtein(
    seqs_a: List[str], seqs_b: List[str]
) -> Tuple[int, Optional[int]]:
    """Compute how many elements of A appear in B and the min Levenshtein distance.

    - Membership count is per-element (duplicates in A counted multiple times).
    - Min distance is the minimum Levenshtein distance over all pairs (a in A, b in B).
      If either set is empty, returns None for the distance.
    - Early exit with distance 0 if any common element exists.
    - Uses length-based pruning and an optional threshold to avoid unnecessary work.
    """
    if not seqs_a or not seqs_b:
        present_count = 0 if not seqs_a else sum(1 for s in seqs_a if False)
        return present_count, None

    set_b = set(seqs_b)
    present_count = sum(1 for s in seqs_a if s in set_b)
    if present_count > 0:
        # If there is any exact intersection, min distance is 0
        return present_count, 0

    # Deduplicate B for distance search; duplicates don't help the minimum
    unique_b = list(set_b)

    # Group B by length for quick lower-bound pruning
    length_to_b: Dict[int, List[str]] = defaultdict(list)
    for sb in unique_b:
        length_to_b[len(sb)].append(sb)
    b_lengths = sorted(length_to_b.keys())

    best: Optional[int] = None  # current best distance

    for sa in seqs_a:
        la = len(sa)
        # Lower bound by closest length bin; if it's already >= best, skip
        # Find nearest length(s) in B
        idx = bisect_left(b_lengths, la)
        def closest_len_diff() -> int:
            if not b_lengths:
                return 10**9
            if idx == 0:
                return abs(b_lengths[0] - la)
            if idx == len(b_lengths):
                return abs(la - b_lengths[-1])
            left = abs(la - b_lengths[idx - 1])
            right = abs(b_lengths[idx] - la)
            return left if left < right else right

        lb_len = closest_len_diff()
        if best is not None and lb_len >= best:
            continue

        # Consider length buckets in order of increasing |len diff|
        # This helps find a good (small) best early to tighten thresholds
        # Build candidate lengths sorted by abs difference
        sorted_lengths = sorted(b_lengths, key=lambda L: abs(L - la))
        for L in sorted_lengths:
            # Prune by length difference lower bound
            len_diff = abs(L - la)
            if best is not None and len_diff >= best:
                # Further lengths will only be worse since sorted by diff
                break
            candidates = length_to_b[L]
            # Threshold one less than current best so we can early-abort > best
            threshold = best - 1 if best is not None else None
            for sb in candidates:
                d = _levenshtein_distance(sa, sb, threshold)
                if best is None or d < best:
                    best = d
                    if best <= 1:
                        # Can't be 0 because we have no exact matches; 1 is optimal now
                        return present_count, best

    return present_count, best

# Custom scoring to handle ambiguous 'X' residues during alignment
# - Identical non-'X' matches score +1.0
# - Non-'X' mismatches score -0.5
# - Any pair involving 'X' (including 'X' vs 'X') scores 0.0 so it does not
#   contribute positively or negatively to the alignment score
_AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
_SUBST_MATRIX: Dict[Tuple[str, str], float] = {}
for _aa in _AA_ALPHABET + ["X"]:
    for _bb in _AA_ALPHABET + ["X"]:
        if _aa == _bb and _aa != "X":
            _score = 1.0
        elif _aa != "X" and _bb != "X" and _aa != _bb:
            _score = -0.5
        else:
            # Any pair with 'X' gets 0.0 (neutral)
            _score = 0.0
        _SUBST_MATRIX[(_aa, _bb)] = _score

_GAP_OPEN = -1.0
_GAP_EXTEND = -0.1

# Global counter for alignment_distance calls
_alignment_call_count = 0

def alignment_distance(seq1: str, seq2: str) -> float:
    """Compute distance = 1 - identity with special handling for 'X'.

    - Uses Biopython global alignment with a custom substitution matrix where
      any pair involving 'X' scores 0.0, identical non-'X' pairs score +1.0,
      and non-'X' mismatches score -0.5. Mild gap penalties discourage
      over-alignment through gaps.
    - Identity is computed only over aligned positions where both residues are
      not gaps and not 'X'. Positions with 'X' are excluded from both the
      numerator and denominator.
    """
    global _alignment_call_count
    _alignment_call_count += 1
    
    # Log timing every 100th call
    if _alignment_call_count % 100 == 0:
        alignment_start = time.time()
        
    if not seq1 and not seq2:
        if _alignment_call_count % 100 == 0:
            slow_log(f"alignment_distance call #{_alignment_call_count} took {time.time() - alignment_start:.4f}s (empty seqs)")
        return 0.0

    a = seq1.upper()
    b = seq2.upper()

    # TODO: figure out what exactly this globalds does.
    # Perform one best global alignment using the custom scoring.
    alignments = pairwise2.align.globalds(
        a,
        b,
        _SUBST_MATRIX,
        _GAP_OPEN,
        _GAP_EXTEND,
        one_alignment_only=True,
        penalize_end_gaps=False,
    )
    if not alignments:
        if _alignment_call_count % 100 == 0:
            slow_log(f"alignment_distance call #{_alignment_call_count} took {time.time() - alignment_start:.4f}s (no alignments)")
        return 0.0

    aligned_a, aligned_b, _score, _start, _end = alignments[0]

    matches = 0
    considered = 0
    for ca, cb in zip(aligned_a, aligned_b):
        # Exclude positions involving 'X' from both numerator and denominator
        if ca == 'X' or cb == 'X':
            continue

        # If both are gaps (rare), ignore
        if ca == '-' and cb == '-':
            continue

        considered += 1
        if ca == '-' or cb == '-':
            # Count indels as mismatches when the counterpart is a known residue
            continue
        if ca == cb:
            matches += 1

    if considered == 0:
        # No comparable positions (all gaps/'X'): treat as zero distance
        if _alignment_call_count % 100 == 0:
            slow_log(f"alignment_distance call #{_alignment_call_count} took {time.time() - alignment_start:.4f}s (no comparable positions)")
        return 0.0

    identity = float(matches) / float(considered)
    result = 1.0 - identity
    
    # Log timing every 100th call
    if _alignment_call_count % 100 == 0:
        slow_log(f"alignment_distance call #{_alignment_call_count} took {time.time() - alignment_start:.4f}s")
    
    return result


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


def load_last_two_token_tensors(dataset_path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")
    data = torch.load(str(dataset_path), map_location="cpu")
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Expected a list with at least two elements in the dataset file.")
    last = data[-1]
    second_last = data[-2]
    # Expect progen ids/masks to be present for generator prompts; if not, raise
    def _extract(x: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        s = x.get("samples", {})
        if not isinstance(s, dict):
            raise KeyError("'samples' missing or not a dict in dataset entries")
        keys = ["progen_input_ids", "progen_attention_mask", "esm_input_ids", "esm_attention_mask"]
        out: Dict[str, torch.Tensor] = {}
        for k in keys:
            if k in s and isinstance(s[k], torch.Tensor):
                out[k] = s[k]
        if "progen_input_ids" not in out or "progen_attention_mask" not in out:
            raise KeyError("Expected 'progen_input_ids' and 'progen_attention_mask' in samples")
        if "esm_input_ids" not in out or "esm_attention_mask" not in out:
            raise KeyError("Expected 'esm_input_ids' and 'esm_attention_mask' in samples for encoder")
        return out
    return _extract(second_last), _extract(last)


def sequences_equivalent_with_x(seq1: str, seq2: str) -> bool:
    """Check if two sequences are equivalent considering X as wildcard.
    
    Two sequences are considered equivalent if they match at every position,
    where X in either sequence matches any amino acid (including X).
    Uses alignment to handle length differences.
    """
    if seq1 == seq2:
        return True
    
    # Align with the same scoring that treats 'X' as neutral
    alignments = pairwise2.align.globalds(
        seq1.upper(),
        seq2.upper(),
        _SUBST_MATRIX,
        _GAP_OPEN,
        _GAP_EXTEND,
        one_alignment_only=True,
        penalize_end_gaps=False,
    )
    if not alignments:
        return False
    
    aligned_seq1, aligned_seq2, score, start, end = alignments[0]
    
    # Check if sequences match considering X as wildcard
    for c1, c2 in zip(aligned_seq1, aligned_seq2):
        # Skip gaps for now - we'll consider sequences with gaps as different
        if c1 == '-' or c2 == '-':
            return False
        
        # If neither is X, they must match exactly
        if c1 != 'X' and c2 != 'X' and c1 != c2:
            return False
        
        # If one is X, that's fine (X matches anything)
    
    return True


def count_unique_sequences_with_x_wildcard(all_sequences: List[str]) -> int:
    """Count unique sequences treating X as wildcard that matches any amino acid."""
    if not all_sequences:
        return 0
    
    unique_seqs = []
    total = len(all_sequences)
    
    print(f"Analyzing {total} sequences for uniqueness (treating X as wildcard)...", file=sys.stderr)
    
    for i, seq in enumerate(all_sequences):
        is_unique = True
        
        # Check against all previously found unique sequences
        for unique_seq in unique_seqs:
            if sequences_equivalent_with_x(seq, unique_seq):
                is_unique = False
                break
        
        if is_unique:
            unique_seqs.append(seq)
        
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"Processed {i+1}/{total} sequences, found {len(unique_seqs)} unique so far", 
                  end="\r", file=sys.stderr)
    
    print(file=sys.stderr)
    return len(unique_seqs)


def analyze_mismatch_patterns(seqs_a: List[str], seqs_b: List[str]) -> Dict[str, int]:
    """Analyze which amino acids in seqs_a cause the most mismatches vs seqs_b.
    
    Returns a dict mapping amino acid -> mismatch count for generated sequences.
    """
    mismatch_counts: Dict[str, int] = {}
    
    print("Analyzing mismatch patterns...", file=sys.stderr)
    total_pairs = len(seqs_a) * len(seqs_b)
    processed = 0
    
    for sa in seqs_a:
        for sb in seqs_b:
            # Get alignment
            alignments = pairwise2.align.globalxx(sa, sb)
            if not alignments:
                continue
            
            aligned_a, aligned_b, score, start, end = alignments[0]
            
            # Count mismatches for each amino acid in seqs_a
            for c_a, c_b in zip(aligned_a, aligned_b):
                # Skip gaps and X's as before
                if c_a != 'X' and c_b != 'X' and c_a != '-' and c_b != '-':
                    if c_a != c_b:  # Mismatch
                        if c_a not in mismatch_counts:
                            mismatch_counts[c_a] = 0
                        mismatch_counts[c_a] += 1
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_pairs} alignment pairs for mismatch analysis", 
                      end="\r", file=sys.stderr)
    
    print(file=sys.stderr)
    return mismatch_counts


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
            "data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100_filtered.pt"
        ),
        help="Absolute path to the .pt dataset file to load.",
    )
    parser.add_argument(
        "--print_matrix",
        action="store_true",
        help="Print the full NxM matrix (careful if large). Requires --compute_matrix.",
    )
    parser.add_argument(
        "--compute_matrix",
        action="store_true",
        help=(
            "Compute the NxM alignment distance matrix and related summaries. "
            "Disabled by default in favor of fast set-overlap and Levenshtein-min reporting."
        ),
    )
    parser.add_argument(
        "--compare_mode",
        type=str,
        choices=["second_last", "generated"],
        default="second_last",
        help="Compare last vs second_last (default) or last vs generated sequences.",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "Absolute path to hashed outputs subdirectory containing best_model.pt and config.yaml. "
            "Required if --compare_mode=generated."
        ),
    )
    parser.add_argument(
        "--num_generate",
        type=int,
        default=None,
        help=(
            "Number of sequences to generate when --compare_mode=generated. "
            "If not specified, generates one sequence for each sequence in second_last dataset."
        ),
    )
    parser.add_argument(
        "--use_baseline",
        action="store_true",
        help=(
            "Use baseline ESM encoder and generator (no Hydra config/checkpoint). "
            "Instantiates baseline models with default args and uses second_last ESM features as latents."
        ),
    )
    parser.add_argument(
        "--debug_print_sequences",
        action="store_true",
        help="Print all sequences from second_last and last (can be very verbose).",
    )
    parser.add_argument(
        "--do_unique_x_analysis",
        action="store_true",
        help="Run unique-sequence analysis treating X as wildcard (slow).",
    )
    parser.add_argument(
        "--do_mismatch_analysis",
        action="store_true",
        help="Run mismatch pattern analysis for generated sequences (slow).",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    dist: Optional[np.ndarray] = None
    if args.compare_mode == "second_last":
        seqs_a, seqs_b = load_last_two_raw_texts(dataset_path)
        
        if args.debug_print_sequences:
            print("SEQUENCES CHECKING")
            for i, seq in enumerate(seqs_a):
                print(f"Sequence {i}: {seq}")
            for i, seq in enumerate(seqs_b):
                print(f"Sequence {i}: {seq}")
            print("SEQUENCES CHECKING")

        n, m = len(seqs_a), len(seqs_b)
        print(f"Length of second_last raw_texts: {n}")
        print(f"Length of last raw_texts: {m}")
        print(f"Loaded last two raw_texts lists with sizes: N={n} (second_last), M={m} (last)")

        # Fast overlap + min-Levenshtein reporting
        present_count, min_lev = compute_membership_and_min_levenshtein(seqs_a, seqs_b)
        print("\n=== Set Overlap and Levenshtein-Min ===")
        print(f"Exact overlaps (elements of second_last in last): {present_count} / {n}")
        if min_lev is None:
            print("Minimum Levenshtein distance between sets: N/A (one set is empty)")
        else:
            print(f"Minimum Levenshtein distance between sets: {min_lev}")

        # Optionally compute distance matrix
        if args.compute_matrix:
            start = time.time()
            dist = compute_distance_matrix(seqs_a, seqs_b, show_progress=True)
            elapsed = time.time() - start
            print(f"Computed distance matrix in {elapsed:.2f}s")
    else:
        # Generated mode
        if args.use_baseline:
            # Baseline path: instantiate baseline encoder/generator with defaults and generate using second_last ESM features
            slow_log("Starting baseline generation path")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            slow_log(f"Using device: {device}")

            # Load tokens and texts from dataset
            slow_log("Loading tokens and texts from dataset")
            start_load = time.time()
            second_last_tokens, last_tokens = load_last_two_token_tensors(dataset_path)
            second_last_texts, last_texts = load_last_two_raw_texts(dataset_path)
            slow_log(f"Dataset loading took {time.time() - start_load:.2f}s")

            print(f"Length of second_last raw_texts: {len(second_last_texts)}")
            print(f"Length of last raw_texts: {len(last_texts)}")

            # Imports are local to avoid heavy deps when not needed
            slow_log("Importing baseline models")
            from encoder.protein_encoders_ESM_baseline import ProteinSetEncoder as BaselineProteinSetEncoder
            from generator.ESM_baseline import ESM2_Baseline_Generator

            slow_log("Instantiating baseline encoder")
            start_enc = time.time()
            encoder = BaselineProteinSetEncoder().to(device)
            slow_log(f"Encoder instantiation took {time.time() - start_enc:.2f}s")
            
            slow_log("Instantiating baseline generator")
            start_gen = time.time()
            generator = ESM2_Baseline_Generator().to(device)
            slow_log(f"Generator instantiation took {time.time() - start_gen:.2f}s")
            
            encoder.eval()
            generator.eval()

            # Prepare tensors from second_last (source) set
            src_esm_ids_all = second_last_tokens["esm_input_ids"]  # [N, L]
            src_esm_mask_all = second_last_tokens["esm_attention_mask"]  # [N, L]

            num_src = int(src_esm_ids_all.shape[0])

            # Determine how many sequences to actually generate
            if args.num_generate is not None:
                num_to_generate = min(args.num_generate, num_src)
                print(f"Generating {num_to_generate} sequences (requested: {args.num_generate}, available: {num_src})")
            else:
                num_to_generate = num_src
                print(f"Generating {num_to_generate} sequences (one for each sequence in second_last)")

            generated_texts: List[str] = []

            # Simple batching for efficiency/memory
            batch_size = max(1, min(32, num_to_generate))
            slow_log(f"Starting generation loop: {num_to_generate} sequences, batch_size={batch_size}")
            
            total_batches = (num_to_generate + batch_size - 1) // batch_size
            for batch_idx, start_idx in enumerate(range(0, num_to_generate, batch_size)):
                batch_start_time = time.time()
                end_idx = min(start_idx + batch_size, num_to_generate)
                current_batch_size = end_idx - start_idx
                
                slow_log(f"Processing batch {batch_idx + 1}/{total_batches} (sequences {start_idx}-{end_idx-1})")
                
                # Prepare batch data
                prep_start = time.time()
                cur_src_esm_ids = src_esm_ids_all[start_idx:end_idx].to(device)
                cur_src_esm_mask = src_esm_mask_all[start_idx:end_idx].to(device)
                slow_log(f"  Batch data prep took {time.time() - prep_start:.3f}s")

                with torch.no_grad():
                    # Get per-token hidden states (B, L, H)
                    enc_start = time.time()
                    latent_target = encoder.esm_extractor(cur_src_esm_ids, cur_src_esm_mask)
                    slow_log(f"  Encoder forward took {time.time() - enc_start:.3f}s")

                    # Sample one sequence per input using baseline generator
                    gen_start = time.time()
                    out_ids = generator.sample(latent_target, num_samples=1, return_texts=False)
                    slow_log(f"  Generator sampling took {time.time() - gen_start:.3f}s")

                # Decode ids to strings using generator tokenizer
                decode_start = time.time()
                # out_ids: [B, 1, L]
                out_ids_cpu = out_ids.detach().cpu()
                B = out_ids_cpu.shape[0]
                for b in range(B):
                    ids_b = out_ids_cpu[b, 0].tolist()
                    text = generator.tokenizer.decode(ids_b, skip_special_tokens=True).replace(" ", "")
                    generated_texts.append(text)
                slow_log(f"  Decoding took {time.time() - decode_start:.3f}s")
                
                slow_log(f"  Total batch time: {time.time() - batch_start_time:.3f}s")
            
            slow_log(f"Generation completed: {len(generated_texts)} sequences generated")

            # Check that all generated sequences have the same length
            if generated_texts:
                seq_lengths = [len(seq) for seq in generated_texts]
                unique_lengths = set(seq_lengths)
                assert len(unique_lengths) == 1, f"Generated sequences have different lengths: {unique_lengths}"
                seq_length = seq_lengths[0]
                print(f"All generated sequences have length: {seq_length}")
                slow_log(f"All generated sequences have length: {seq_length}")
            else:
                print("No sequences generated")
                slow_log("No sequences generated")

            # Now compare last_texts vs generated_texts
            seqs_a = generated_texts
            seqs_b = last_texts
            print(f"Length of generated sequences: {len(seqs_a)}")
            print(f"Length of last raw_texts: {len(seqs_b)}")
            print(f"Generated {len(seqs_a)} sequences. Comparing to last ({len(seqs_b)} sequences)")

            # Fast overlap + min-Levenshtein reporting
            present_count, min_lev = compute_membership_and_min_levenshtein(seqs_a, seqs_b)
            print("\n=== Set Overlap and Levenshtein-Min ===")
            print(f"Exact overlaps (elements of generated in last): {present_count} / {len(seqs_a)}")
            if min_lev is None:
                print("Minimum Levenshtein distance between sets: N/A (one set is empty)")
            else:
                print(f"Minimum Levenshtein distance between sets: {min_lev}")

            # Optionally compute distance matrix
            if args.compute_matrix:
                start = time.time()
                dist = compute_distance_matrix(seqs_a, seqs_b, show_progress=True)
                elapsed = time.time() - start
                print(f"Computed distance matrix in {elapsed:.2f}s")

            # Optional slow mismatch analysis
            if args.do_mismatch_analysis:
                print("\n=== Mismatch Pattern Analysis ===")
                mismatch_counts = analyze_mismatch_patterns(seqs_a, seqs_b)
                
                if mismatch_counts:
                    sorted_mismatches = sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)
                    top_5 = sorted_mismatches[:5]
                    print("Top 5 amino acids in generated sequences causing most mismatches:")
                    for i, (aa, count) in enumerate(top_5, 1):
                        print(f"  {i}. {aa}: {count} mismatches")
                    total_mismatches = sum(mismatch_counts.values())
                    print(f"\nTotal mismatches analyzed: {total_mismatches}")
                    if total_mismatches > 0:
                        print("Percentage contribution of top 5:")
                        for i, (aa, count) in enumerate(top_5, 1):
                            pct = (count / total_mismatches) * 100
                            print(f"  {i}. {aa}: {pct:.1f}%")
                else:
                    print("No mismatches found in comparable positions.")
        else:
            # Original path: Hydra-instantiated models and checkpoint loading
            if not args.run_dir:
                raise ValueError("--run_dir is required for --compare_mode=generated")
            run_dir = Path(args.run_dir)
            cfg_path = run_dir / "config.yaml"
            ckpt_path = run_dir / "best_model.pt"
            if not cfg_path.exists():
                raise FileNotFoundError(f"config.yaml not found in run dir: {run_dir}")
            if not ckpt_path.exists():
                raise FileNotFoundError(f"best_model.pt not found in run dir: {run_dir}")

            # Load tokens and texts from dataset
            second_last_tokens, last_tokens = load_last_two_token_tensors(dataset_path)
            second_last_texts, last_texts = load_last_two_raw_texts(dataset_path)
            
            # Print lengths for both datasets
            print(f"Length of second_last raw_texts: {len(second_last_texts)}")
            print(f"Length of last raw_texts: {len(last_texts)}")

            # Instantiate models from saved cfg
            cfg = OmegaConf.load(str(cfg_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            encoder = hydra.utils.instantiate(cfg.encoder)
            generator = hydra.utils.instantiate(cfg.generator)
            encoder = encoder.to(device)
            generator = generator.to(device)
            encoder.eval()
            generator.eval()

            # Load weights from checkpoint
            try:
                state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
            except TypeError:
                state = torch.load(str(ckpt_path), map_location=device)
            if 'encoder_state_dict' not in state or 'generator_state_dict' not in state:
                raise KeyError("Checkpoint must contain 'encoder_state_dict' and 'generator_state_dict'")
            encoder.load_state_dict(state['encoder_state_dict'])
            generator.load_state_dict(state['generator_state_dict'])

            # Determine set_size from cfg
            try:
                set_size = int(cfg.experiment.set_size)
            except Exception as exc:
                raise KeyError("cfg.experiment.set_size not found in config.yaml") from exc

            # Prepare tensors
            src_ids_all = second_last_tokens["progen_input_ids"]  # [set_size_like, L]
            src_mask_all = second_last_tokens["progen_attention_mask"]
            src_esm_ids_all = second_last_tokens["esm_input_ids"]
            src_esm_mask_all = second_last_tokens["esm_attention_mask"]

            tgt_ids_all = last_tokens["progen_input_ids"]
            tgt_mask_all = last_tokens["progen_attention_mask"]
            tgt_esm_ids_all = last_tokens["esm_input_ids"]
            tgt_esm_mask_all = last_tokens["esm_attention_mask"]

            num_src = int(src_esm_ids_all.shape[0])
            seq_len = int(src_ids_all.shape[1])
            if hasattr(cfg.generator, 'seq_length'):
                expected_L = int(cfg.generator.seq_length)
                if expected_L != seq_len:
                    print(f"Warning: generator.seq_length={expected_L} but dataset L={seq_len}")

            # Determine how many sequences to actually generate
            if args.num_generate is not None:
                num_to_generate = min(args.num_generate, num_src)
                print(f"Generating {num_to_generate} sequences (requested: {args.num_generate}, available: {num_src})")
            else:
                num_to_generate = num_src
                print(f"Generating {num_to_generate} sequences (one for each sequence in second_last)")

            # Partition into batches of set_size
            full_batches = num_to_generate // set_size
            remainder = num_to_generate % set_size

            # Helper to encode a set of size set_size: expects tensors [set_size, L] -> batch dict with [1, set_size, L]
            def encode_set(esm_ids: torch.Tensor, esm_mask: torch.Tensor) -> torch.Tensor:
                batch = {
                    'esm_input_ids': esm_ids.unsqueeze(0).to(device),
                    'esm_attention_mask': esm_mask.unsqueeze(0).to(device),
                }
                with torch.no_grad():
                    lat = encoder(batch)
                if lat.dim() == 2 and lat.size(0) == 1:
                    lat = lat
                elif lat.dim() == 1:
                    lat = lat.unsqueeze(0)
                return lat  # shape [1, latent_dim]

            generated_texts: List[str] = []

            # For each full batch
            for b in range(full_batches):
                start_idx = b * set_size
                end_idx = start_idx + set_size
                cur_src_ids = src_ids_all[start_idx:end_idx]
                cur_src_mask = src_mask_all[start_idx:end_idx]
                cur_src_esm_ids = src_esm_ids_all[start_idx:end_idx]
                cur_src_esm_mask = src_esm_mask_all[start_idx:end_idx]

                # Random target subset of size set_size from last
                perm_tgt = torch.randperm(tgt_esm_ids_all.shape[0])[:set_size]
                cur_tgt_esm_ids = tgt_esm_ids_all[perm_tgt]
                cur_tgt_esm_mask = tgt_esm_mask_all[perm_tgt]

                # Encode latents
                lat_src = encode_set(cur_src_esm_ids, cur_src_esm_mask)  # [1, d]
                lat_tgt = encode_set(cur_tgt_esm_ids, cur_tgt_esm_mask)  # [1, d]

                # Build x_source dict as expected by generator.sample with batch size 1 and set_size
                x_source = {
                    'esm_input_ids': cur_src_esm_ids.unsqueeze(0).to(device),
                    'esm_attention_mask': cur_src_esm_mask.unsqueeze(0).to(device),
                    'progen_input_ids': cur_src_ids.unsqueeze(0).to(device),
                    'progen_attention_mask': cur_src_mask.unsqueeze(0).to(device),
                }

                out_ids, out_texts = generator.sample(x_source, lat_src, lat_tgt, num_samples=cur_src_esm_ids.shape[0], return_texts=True)
                print(type(out_texts), len(out_texts), len(out_texts[0]), type(out_texts[0]))
                generated_texts.extend(out_texts[0])

            # Handle remainder by padding with repeats up to set_size
            if remainder > 0:
                start_idx = full_batches * set_size
                end_idx = start_idx + remainder
                cur_src_ids = src_ids_all[start_idx:end_idx]
                cur_src_mask = src_mask_all[start_idx:end_idx]
                cur_src_esm_ids = src_esm_ids_all[start_idx:end_idx]
                cur_src_esm_mask = src_esm_mask_all[start_idx:end_idx]

                pad_needed = set_size - remainder
                pad_idx = torch.randint(low=0, high=num_src, size=(pad_needed,))
                cur_src_ids_padded = torch.cat([cur_src_ids, src_ids_all[pad_idx]], dim=0)
                cur_src_mask_padded = torch.cat([cur_src_mask, src_mask_all[pad_idx]], dim=0)
                cur_src_esm_ids_padded = torch.cat([cur_src_esm_ids, src_esm_ids_all[pad_idx]], dim=0)
                cur_src_esm_mask_padded = torch.cat([cur_src_esm_mask, src_esm_mask_all[pad_idx]], dim=0)

                perm_tgt = torch.randperm(tgt_esm_ids_all.shape[0])[:set_size]
                cur_tgt_esm_ids = tgt_esm_ids_all[perm_tgt]
                cur_tgt_esm_mask = tgt_esm_mask_all[perm_tgt]

                lat_src = encode_set(cur_src_esm_ids_padded, cur_src_esm_mask_padded)
                lat_tgt = encode_set(cur_tgt_esm_ids, cur_tgt_esm_mask)

                x_source = {
                    'esm_input_ids': cur_src_esm_ids.unsqueeze(0).to(device),
                    'esm_attention_mask': cur_src_esm_mask.unsqueeze(0).to(device),
                    'progen_input_ids': cur_src_ids.unsqueeze(0).to(device),
                    'progen_attention_mask': cur_src_mask.unsqueeze(0).to(device),
                }

                out_ids, out_texts = generator.sample(x_source, lat_src, lat_tgt, num_samples=cur_src_esm_ids.shape[0], return_texts=True)
                print(type(out_texts), len(out_texts), len(out_texts[0]), type(out_texts[0]))
                print(out_texts)
                generated_texts.extend(out_texts[0])
           

            # Now compare last_texts vs generated_texts
            seqs_a = generated_texts
            seqs_b = last_texts
            print(f"Length of generated sequences: {len(seqs_a)}")
            print(f"Length of last raw_texts: {len(seqs_b)}")
            print(f"Generated {len(seqs_a)} sequences. Comparing to last ({len(seqs_b)} sequences)")

            # Fast overlap + min-Levenshtein reporting
            present_count, min_lev = compute_membership_and_min_levenshtein(seqs_a, seqs_b)
            print("\n=== Set Overlap and Levenshtein-Min ===")
            print(f"Exact overlaps (elements of generated in last): {present_count} / {len(seqs_a)}")
            if min_lev is None:
                print("Minimum Levenshtein distance between sets: N/A (one set is empty)")
            else:
                print(f"Minimum Levenshtein distance between sets: {min_lev}")

            # Optionally compute distance matrix
            if args.compute_matrix:
                start = time.time()
                dist = compute_distance_matrix(seqs_a, seqs_b, show_progress=True)
                elapsed = time.time() - start
                print(f"Computed distance matrix in {elapsed:.2f}s")

            # Optional slow mismatch analysis
            if args.do_mismatch_analysis:
                print("\n=== Mismatch Pattern Analysis ===")
                mismatch_counts = analyze_mismatch_patterns(seqs_a, seqs_b)
                
                if mismatch_counts:
                    sorted_mismatches = sorted(mismatch_counts.items(), key=lambda x: x[1], reverse=True)
                    top_5 = sorted_mismatches[:5]
                    print("Top 5 amino acids in generated sequences causing most mismatches:")
                    for i, (aa, count) in enumerate(top_5, 1):
                        print(f"  {i}. {aa}: {count} mismatches")
                    total_mismatches = sum(mismatch_counts.values())
                    print(f"\nTotal mismatches analyzed: {total_mismatches}")
                    if total_mismatches > 0:
                        print("Percentage contribution of top 5:")
                        for i, (aa, count) in enumerate(top_5, 1):
                            pct = (count / total_mismatches) * 100
                            print(f"  {i}. {aa}: {pct:.1f}%")
                else:
                    print("No mismatches found in comparable positions.")

    # Optionally print matrix and summaries if computed
    if dist is not None:
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


