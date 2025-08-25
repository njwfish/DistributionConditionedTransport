import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import hydra

from virus_eval.utils import (
    setup_console_logger,
    resolve_outputs_dir,
    load_cfg_from_run_dir,
    instantiate_models,
    load_weights,
    encode_source_samples,
    encode_target_samples,
    build_fixed_offset_indices,
)



def main():
    parser = argparse.ArgumentParser(description="Script 1: Generate aggregated dataset for time-point model evaluation")
    parser.add_argument("outputs_subdir", type=str, help="Subdirectory under ./outputs (or absolute run dir path)")
    parser.add_argument("--offset", "-n", type=int, required=True, help="Integer N for target_idx - source_idx == N")
    parser.add_argument("--epochs", type=int, default=5, help="Aggregation epochs over dataset sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output .npz path to save aggregated dataset (defaults to virus_eval/dataset_offset_<N>.npz)",
    )
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    logger = setup_console_logger("virus_eval.script1")

    run_dir = resolve_outputs_dir(args.outputs_subdir)
    cfg = load_cfg_from_run_dir(run_dir)

    device = torch.device(args.device)

    # Instantiate dataset and encoder (predictor not required here)
    logger.info("Instantiating dataset from saved config…")
    dataset = hydra.utils.instantiate(cfg.dataset)
    encoder, predictor = instantiate_models(cfg, device, logger)
    load_weights(encoder, predictor, run_dir, device, logger, require_predictor=False)

    # Determine set size used for encoding
    eval_set_size = int(getattr(cfg.dataset, "set_size", 16))

    # Build indices with fixed offset N
    fixed_indices = build_fixed_offset_indices(dataset, args.offset)
    if len(fixed_indices) == 0:
        raise RuntimeError(f"No dataset pairs satisfy target_idx - source_idx == {args.offset}")
    logger.info(f"Found {len(fixed_indices)} pairs with offset N={args.offset}")

    # Aggregation buffers (per-sample, flattened)
    source_latents: List[np.ndarray] = []
    true_target_latents: List[np.ndarray] = []
    source_indices: List[int] = []
    target_indices: List[int] = []

    # Also aggregate by (src_idx, tgt_idx) pairs for convenient later access
    pair_to_src_latents: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    pair_to_true_tgt_latents: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

    # Aggregate across epochs; each epoch we reshuffle eligible pairs
    rng = np.random.RandomState(0)
    for ep in range(1, args.epochs + 1):
        order = fixed_indices.copy()
        rng.shuffle(order)
        logger.info(f"Epoch {ep}/{args.epochs}: sampling {len(order)} pairs")

        with torch.no_grad():
            for k, ds_idx in enumerate(order):
                item = dataset[ds_idx]
                src_idx = int(item["source_idx"])  # scalar
                tgt_idx = int(item["target_idx"])  # scalar

                # Encode source to latent
                src_latent_t = encode_source_samples(
                    encoder=encoder,
                    sample_entry=item,
                    device=device,
                    target_set_size=eval_set_size,
                )  # [D]
                src_lat_np = src_latent_t.numpy()
                source_latents.append(src_lat_np)
                source_indices.append(src_idx)
                target_indices.append(tgt_idx)

                # Record into pair-grouped buffers
                pair_to_src_latents[(src_idx, tgt_idx)].append(src_lat_np)

                # Encode true target latent as well
                tgt_latent_true = encode_target_samples(
                    encoder=encoder,
                    sample_entry=item,
                    device=device,
                    target_set_size=eval_set_size,
                )
                tgt_lat_np = tgt_latent_true.numpy()
                true_target_latents.append(tgt_lat_np)

                # Record into pair-grouped buffers
                pair_to_true_tgt_latents[(src_idx, tgt_idx)].append(tgt_lat_np)

                if args.log_interval > 0 and (k + 1) % args.log_interval == 0:
                    logger.info(f"Processed {k+1}/{len(order)} pairs in epoch {ep}")

    # Stack and save (flat arrays for backwards compatibility)
    feats_np = np.stack(source_latents, axis=0).astype(np.float32)
    true_tgts_np = np.stack(true_target_latents, axis=0).astype(np.float32)
    src_idx_np = np.array(source_indices, dtype=np.int64)
    tgt_idx_np = np.array(target_indices, dtype=np.int64)

    # Build pair-grouped arrays similar to eval_offset_idx_mlp.py true_tgt_by_pair
    # Create a stable ordering of unique pairs
    pair_keys_list: List[Tuple[int, int]] = sorted(list(pair_to_src_latents.keys()), key=lambda t: (t[0], t[1]))
    pair_keys_np = np.array(pair_keys_list, dtype=np.int64) if len(pair_keys_list) > 0 else np.zeros((0, 2), dtype=np.int64)

    # Convert lists to stacked arrays per pair (ragged), stored as object arrays
    grouped_src_obj = np.empty(len(pair_keys_list), dtype=object)
    grouped_true_tgt_obj = np.empty(len(pair_keys_list), dtype=object)
    pair_counts = np.zeros(len(pair_keys_list), dtype=np.int64)
    for i, key in enumerate(pair_keys_list):
        src_list = pair_to_src_latents.get(key, [])
        tgt_list = pair_to_true_tgt_latents.get(key, [])
        if len(src_list) > 0:
            grouped_src_obj[i] = np.stack(src_list, axis=0).astype(np.float32)
        else:
            raise RuntimeError(f"No source latents found for pair {key}")
        if len(tgt_list) > 0:
            grouped_true_tgt_obj[i] = np.stack(tgt_list, axis=0).astype(np.float32)
        else:
            raise RuntimeError(f"No true target latents found for pair {key}")
        pair_counts[i] = grouped_src_obj[i].shape[0]

    if args.out_path is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out_dir = os.path.join(repo_root, "virus_eval")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"dataset_offset_{args.offset}.npz")
    else:
        out_path = args.out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    np.savez_compressed(
        out_path,
        # Flat (backward-compatible)
        source_latents=feats_np,
        true_target_latents=true_tgts_np,
        source_indices=src_idx_np,
        target_indices=tgt_idx_np,
        # Pair-grouped
        pair_keys=pair_keys_np,  # shape [M, 2], dtype int64
        grouped_source_latents=grouped_src_obj,  # dtype object -> arrays [k_i, D]
        grouped_true_target_latents=grouped_true_tgt_obj,  # dtype object -> arrays [k_i, D]
        pair_counts=pair_counts,
        # Meta
        offset=np.array([args.offset], dtype=np.int64),
        set_size=np.array([eval_set_size], dtype=np.int64),
    )
    logger.info(f"Saved aggregated dataset to: {out_path}")


if __name__ == "__main__":
    main()


