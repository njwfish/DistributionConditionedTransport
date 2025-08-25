import os
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import hydra
from sklearn.metrics import mean_squared_error

from virus_eval.utils import (
    setup_file_logger,
    resolve_outputs_dir,
    load_cfg_from_run_dir,
    instantiate_models,
    load_weights,
    encode_source_samples,
    encode_target_samples,
    build_fixed_offset_indices,
    IdxMLP,
)



def main():
    parser = argparse.ArgumentParser(description="Script 3: Evaluate models reproducing eval_offset_idx_mlp metrics")
    parser.add_argument("outputs_subdir", type=str, help="Subdirectory under ./outputs (or absolute run dir path)")
    parser.add_argument("--offset", "-n", type=int, required=True, help="Integer N for target_idx - source_idx == N")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/virus_eval/dataset_offset_1.npz",
        help="Path to dataset_offset_<OFFSET>.npz",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/virus_eval",
        help="Directory where script 2 saved trained models",
    )
    parser.add_argument("--log_path", type=str, default="/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/virus_eval/script3_logger.log")
    args = parser.parse_args()

    logger = setup_file_logger("virus_eval.script3", logfile_path=args.log_path)

    run_dir = resolve_outputs_dir(args.outputs_subdir)
    cfg = load_cfg_from_run_dir(run_dir)
    device = torch.device(args.device)

    # Instantiate encoder and predictor; load weights from training run
    encoder, predictor = instantiate_models(cfg, device, logger)
    load_weights(encoder, predictor, run_dir, device, logger, require_predictor=True)

    # Load dataset from script 1
    data = np.load(args.dataset_path)
    source_latents = data["source_latents"].astype(np.float32)
    true_target_latents = data["true_target_latents"].astype(np.float32)
    source_indices = data["source_indices"].astype(np.float32)
    target_indices = data["target_indices"].astype(np.float32)

    n, d = source_latents.shape
    logger.info(f"Loaded dataset: N={n}, D={d}")

    # Load the time-point prediction models from script 2
    mlp_path = os.path.join(args.models_dir, "mlp_sourceidx.pt")
    ridge_path = os.path.join(args.models_dir, "ridge_sourceidx.npz")
    if not os.path.exists(mlp_path) or not os.path.exists(ridge_path):
        raise FileNotFoundError("Trained models from script 2 not found.")

    mlp_payload = torch.load(mlp_path, map_location="cpu")
    mlp = IdxMLP(input_dim=int(mlp_payload["input_dim"]))
    mlp.load_state_dict(mlp_payload["state_dict"])
    mlp.to(device)
    mlp.eval()

    ridge_payload = np.load(ridge_path)
    ridge_coef = ridge_payload["coef"].astype(np.float32)
    ridge_intercept = float(ridge_payload["intercept"][0])

    def ridge_predict(x: np.ndarray) -> np.ndarray:
        return x @ ridge_coef + ridge_intercept

    # Compute MLP and Ridge predictions of source indices from source_latents, comparable to eval_offset_idx_mlp
    with torch.no_grad():
        mlp_preds_src = mlp(torch.from_numpy(source_latents).to(device)).detach().cpu().numpy()
    ridge_preds_src = ridge_predict(source_latents)

    mse_mlp = float(mean_squared_error(source_indices, mlp_preds_src))
    mse_ridge = float(mean_squared_error(source_indices, ridge_preds_src))
    logger.info(f"Overall MSE on source latents -> source_idx | MLP={mse_mlp:.4f} | Ridge={mse_ridge:.4f}")

    # Also print some pairs of true vs pred for a sample
    logger.info("Sample of predictions (first 50): true vs MLP vs Ridge")
    limit = min(50, n)
    for i in range(limit):
        logger.info(f"i={i:04d} true={int(source_indices[i])} mlp={mlp_preds_src[i]:.3f} ridge={ridge_preds_src[i]:.3f}")

    # Compute predicted target latents p(x_source) using the latent-to-latent predictor
    logger.info("Computing predicted target latents with predictor…")
    pred_target_latents = np.zeros_like(source_latents, dtype=np.float32)
    bs = 2048
    with torch.no_grad():
        total = source_latents.shape[0]
        for start in range(0, total, bs):
            end = min(total, start + bs)
            xb = torch.from_numpy(source_latents[start:end]).to(device)
            d1 = torch.from_numpy(source_indices[start:end]).to(device).float()
            d2 = torch.from_numpy(target_indices[start:end]).to(device).float()
            if getattr(predictor, "requires_condition", False):
                yb = predictor(xb, condition_scalars=(d1, d2))
            else:
                yb = predictor(xb)
            pred_target_latents[start:end] = yb.detach().cpu().numpy().astype(np.float32)

    # Evaluate f(p(x_source)) vs target_idx using both MLP and Ridge
    with torch.no_grad():
        mlp_preds_on_pred = mlp(torch.from_numpy(pred_target_latents).to(device)).detach().cpu().numpy()
    ridge_preds_on_pred = ridge_predict(pred_target_latents)
    mse_mlp_on_pred = float(mean_squared_error(target_indices, mlp_preds_on_pred))
    mse_ridge_on_pred = float(mean_squared_error(target_indices, ridge_preds_on_pred))
    logger.info(f"F(p(x_source)) vs target_idx MSE | MLP={mse_mlp_on_pred:.4f} | Ridge={mse_ridge_on_pred:.4f}")

    # Group predictions by (source_idx, target_idx) and report mean±std and rounded means
    pair_to_pred_latents: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for i in range(n):
        pair_to_pred_latents[(int(source_indices[i]), int(target_indices[i]))].append(pred_target_latents[i])

    keys_sorted_pred = sorted(list(pair_to_pred_latents.keys()), key=lambda x: (x[0], x[1]))
    with torch.no_grad():
        for (src_idx, tgt_idx) in keys_sorted_pred:
            latents_pred = np.stack(pair_to_pred_latents[(src_idx, tgt_idx)], axis=0).astype(np.float32)
            x_pred = torch.from_numpy(latents_pred).to(device)
            mlp_vals = mlp(x_pred).detach().cpu().numpy()
            ridge_vals = ridge_predict(latents_pred)
            mean_mlp = float(np.mean(mlp_vals))
            std_mlp = float(np.std(mlp_vals))
            rounded_mlp = int(np.rint(mean_mlp))
            mean_ridge = float(np.mean(ridge_vals))
            std_ridge = float(np.std(ridge_vals))
            rounded_ridge = int(np.rint(mean_ridge))
            logger.info(
                f"source_idx={src_idx:3d}  true_target_idx={tgt_idx:3d}  "
                f"mlp_on_pred(mean±std)={mean_mlp:.2f}±{std_mlp:.2f}  rounded={rounded_mlp}  "
                f"| ridge_on_pred(mean±std)={mean_ridge:.2f}±{std_ridge:.2f}  rounded={rounded_ridge}"
            )

    # Additionally, reproduce the per-(src_idx, tgt_idx) averaging behavior against true target latents
    # Like the original script, we will compute mean±std of the MLP prediction when fed target_latents
    # Here we use the true target latents encoded and aggregated in dataset 1
    pairs = list(zip(source_indices.astype(int).tolist(), target_indices.astype(int).tolist()))
    pair_to_true_latents: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    for i in range(n):
        pair_to_true_latents[(pairs[i][0], pairs[i][1])].append(true_target_latents[i])

    keys_sorted = sorted(list(pair_to_true_latents.keys()), key=lambda x: (x[0], x[1]))

    with torch.no_grad():
        for (src_idx, tgt_idx) in keys_sorted:
            latents = pair_to_true_latents[(src_idx, tgt_idx)]
            latents_np = np.stack(latents, axis=0).astype(np.float32)
            x_true = torch.from_numpy(latents_np).to(device)
            preds_true_mlp = mlp(x_true).detach().cpu().numpy()
            mean_true_mlp = float(np.mean(preds_true_mlp))
            std_true_mlp = float(np.std(preds_true_mlp))
            rounded_true_mlp = int(np.rint(mean_true_mlp))
            preds_true_ridge = ridge_predict(latents_np)
            mean_true_ridge = float(np.mean(preds_true_ridge))
            std_true_ridge = float(np.std(preds_true_ridge))
            rounded_true_ridge = int(np.rint(mean_true_ridge))
            logger.info(
                f"source_idx={src_idx:3d}  true_target_idx={tgt_idx:3d}  "
                f"mlp_pred_on_true(mean±std)={mean_true_mlp:.2f}±{std_true_mlp:.2f}  rounded_mean={rounded_true_mlp}  "
                f"| ridge_pred_on_true(mean±std)={mean_true_ridge:.2f}±{std_true_ridge:.2f}  rounded_mean={rounded_true_ridge}"
            )


if __name__ == "__main__":
    main()


