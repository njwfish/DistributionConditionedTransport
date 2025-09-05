import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

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
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    print(f"Loading dataset from: {dataset_path}")
    if args.compare_mode == "second_last":
        seqs_a, seqs_b = load_last_two_raw_texts(dataset_path)
        n, m = len(seqs_a), len(seqs_b)
        print(f"Length of second_last raw_texts: {n}")
        print(f"Length of last raw_texts: {m}")
        print(f"Loaded last two raw_texts lists with sizes: N={n} (second_last), M={m} (last)")

        # Compute NxM distance matrix
        start = time.time()
        dist = compute_distance_matrix(seqs_a, seqs_b, show_progress=True)
        elapsed = time.time() - start
        print(f"Computed distance matrix in {elapsed:.2f}s")
    else:
        # Generated mode
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
        # PyTorch 2.6 defaults weights_only=True, which can fail for configs in checkpoints.
        # Load with weights_only=False (trusted local checkpoint). Fallback for older PyTorch.
        try:
            state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        except TypeError:
            # Older PyTorch without weights_only kwarg
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

        # Partition into batches of set_size
        full_batches = num_src // set_size
        remainder = num_src % set_size

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

            # Sample 1 sequence per source element by generating one set and decoding each set element prompt
            # The generator's sample returns [batch, num_samples, seq_len] token ids of target (with BOS)
            out_ids = generator.sample(x_source, lat_src, lat_tgt, num_samples=1, return_texts=False)
            # Decode each target generated for this set
            # out_ids: [1, 1, seq_len]; we want strings for each of set_size prompts
            # But sample currently generates one sequence per batch, not per set element. Use internal prompt selection logic:
            # To ensure one per element, call sample repeatedly with different set element prompts using slicing
            if isinstance(out_ids, torch.Tensor) and out_ids.dim() == 3 and out_ids.size(0) == 1 and out_ids.size(1) == 1:
                # Decode single sequence for the entire set; replicate as conservative fallback
                decoded = generator.tokenizer.decode(out_ids[0, 0].detach().cpu(), skip_special_tokens=True)
                # We need one generated per source. To adhere to requirement, generate per element using loop
                generated_set_texts: List[str] = []
                for set_idx in range(set_size):
                    x_source_elem = {
                        'esm_input_ids': cur_src_esm_ids[set_idx:set_idx+1].to(device),
                        'esm_attention_mask': cur_src_esm_mask[set_idx:set_idx+1].to(device),
                        'progen_input_ids': cur_src_ids[set_idx:set_idx+1].to(device),
                        'progen_attention_mask': cur_src_mask[set_idx:set_idx+1].to(device),
                    }
                    out_ids_elem = generator.sample(x_source_elem, lat_src, lat_tgt, num_samples=1, return_texts=False)
                    if isinstance(out_ids_elem, torch.Tensor) and out_ids_elem.dim() == 3:
                        txt = generator.tokenizer.decode(out_ids_elem[0, 0].detach().cpu(), skip_special_tokens=True)
                    else:
                        # If returns ids without extra dims
                        ids = out_ids_elem.squeeze(0).squeeze(0) if isinstance(out_ids_elem, torch.Tensor) else out_ids_elem
                        txt = generator.tokenizer.decode(ids.detach().cpu(), skip_special_tokens=True)
                    generated_set_texts.append(txt)
                generated_texts.extend(generated_set_texts)
            else:
                # Fallback: try to decode assuming [1, 1, L]
                ids = out_ids[0, 0] if isinstance(out_ids, torch.Tensor) else out_ids
                decoded = generator.tokenizer.decode(ids.detach().cpu(), skip_special_tokens=True)
                generated_texts.append(decoded)

        # Handle remainder by padding with repeats up to set_size
        if remainder > 0:
            cur_src_ids = src_ids_all[-remainder:]
            cur_src_mask = src_mask_all[-remainder:]
            cur_src_esm_ids = src_esm_ids_all[-remainder:]
            cur_src_esm_mask = src_esm_mask_all[-remainder:]

            # pad indices by random sampling from these remainder indices
            pad_needed = set_size - remainder
            pad_idx = torch.randint(low=0, high=remainder, size=(pad_needed,))
            cur_src_ids = torch.cat([cur_src_ids, cur_src_ids[pad_idx]], dim=0)
            cur_src_mask = torch.cat([cur_src_mask, cur_src_mask[pad_idx]], dim=0)
            cur_src_esm_ids = torch.cat([cur_src_esm_ids, cur_src_esm_ids[pad_idx]], dim=0)
            cur_src_esm_mask = torch.cat([cur_src_esm_mask, cur_src_esm_mask[pad_idx]], dim=0)

            # Random target subset of size set_size
            perm_tgt = torch.randperm(tgt_esm_ids_all.shape[0])[:set_size]
            cur_tgt_esm_ids = tgt_esm_ids_all[perm_tgt]
            cur_tgt_esm_mask = tgt_esm_mask_all[perm_tgt]

            # Encode latents
            lat_src = encode_set(cur_src_esm_ids, cur_src_esm_mask)
            lat_tgt = encode_set(cur_tgt_esm_ids, cur_tgt_esm_mask)

            # Generate per original remainder element exactly once
            for set_idx in range(remainder):
                x_source_elem = {
                    'esm_input_ids': cur_src_esm_ids[set_idx:set_idx+1].to(device),
                    'esm_attention_mask': cur_src_esm_mask[set_idx:set_idx+1].to(device),
                    'progen_input_ids': cur_src_ids[set_idx:set_idx+1].to(device),
                    'progen_attention_mask': cur_src_mask[set_idx:set_idx+1].to(device),
                }
                out_ids_elem = generator.sample(x_source_elem, lat_src, lat_tgt, num_samples=1, return_texts=False)
                if isinstance(out_ids_elem, torch.Tensor) and out_ids_elem.dim() >= 2:
                    ids = out_ids_elem.squeeze(0).squeeze(0)
                else:
                    ids = out_ids_elem
                txt = generator.tokenizer.decode(ids.detach().cpu(), skip_special_tokens=True)
                generated_texts.append(txt)

        # Now compare last_texts vs generated_texts
        seqs_a = generated_texts
        seqs_b = last_texts
        print(f"Length of generated sequences: {len(seqs_a)}")
        print(f"Length of last raw_texts: {len(seqs_b)}")
        print(f"Generated {len(seqs_a)} sequences. Comparing to last ({len(seqs_b)} sequences)")
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


