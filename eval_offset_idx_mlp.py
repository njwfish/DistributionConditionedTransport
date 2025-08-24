import os
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("eval_offset_idx_mlp")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def resolve_outputs_dir(outputs_subdir: str) -> str:
    # Accept absolute path or subdir name under ./outputs
    if os.path.isdir(outputs_subdir):
        return os.path.abspath(outputs_subdir)
    repo_root = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(repo_root, "outputs", outputs_subdir)
    if os.path.isdir(candidate):
        return candidate
    raise FileNotFoundError(f"Could not find outputs directory: '{outputs_subdir}'")


def load_cfg_from_run_dir(run_dir: str) -> OmegaConf:
    cfg_path_yaml = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(cfg_path_yaml):
        raise FileNotFoundError(f"config.yaml not found in run dir: {run_dir}")
    return OmegaConf.load(cfg_path_yaml)


def instantiate_models(cfg, device: torch.device, logger: logging.Logger):
    logger.info("Instantiating encoder and predictor from saved config…")
    encoder = hydra.utils.instantiate(cfg.encoder)

    predictor = None
    if hasattr(cfg, "predictor") and cfg.predictor is not None:
        predictor = hydra.utils.instantiate(cfg.predictor)
        # Harmonize latent activation if present
        if hasattr(encoder, "latent_act") and not hasattr(predictor, "latent_act"):
            try:
                predictor.latent_act = encoder.latent_act
            except Exception:
                pass

    encoder.to(device)
    encoder.eval()
    if predictor is not None:
        predictor.to(device)
        predictor.eval()
    return encoder, predictor


def load_weights(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    run_dir: str,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    ckpt_path_encoder = os.path.join(run_dir, "best_model.pt")
    ckpt_path_predictor = os.path.join(run_dir, "predictor_training", "predictor_best_model.pt")
    if not os.path.exists(ckpt_path_encoder):
        raise FileNotFoundError(f"best_model.pt not found in {run_dir}")
    logger.info(f"Loading encoder (and possibly predictor) weights from {ckpt_path_encoder}")
    try:
        state = torch.load(ckpt_path_encoder, map_location=device, weights_only=False)
    except Exception:
        state = torch.load(ckpt_path_encoder, map_location=device)

    if not os.path.exists(ckpt_path_predictor):
        raise FileNotFoundError(f"predictor_best_model.pt not found in {run_dir}")
    logger.info(f"Loading predictor weights from {ckpt_path_predictor}")
    try:
        state_predictor = torch.load(ckpt_path_predictor, map_location=device, weights_only=False)
    except Exception:
        state_predictor = torch.load(ckpt_path_predictor, map_location=device)

    if "encoder_state_dict" not in state:
        raise KeyError("Checkpoint missing 'encoder_state_dict'")
    encoder.load_state_dict(state["encoder_state_dict"])
    logger.info("Encoder weights loaded.")

    predictor_loaded = False
    if predictor is not None:
        if "predictor_state_dict" in state_predictor:
            predictor.load_state_dict(state_predictor["predictor_state_dict"])
            predictor_loaded = True
            logger.info("Predictor weights loaded from best_model.pt")
        else:
            # Try posthoc predictor best
            pred_best = os.path.join(run_dir, "predictor_training", "predictor_best_model.pt")
            if os.path.exists(pred_best):
                logger.info(f"Loading predictor weights from {pred_best}")
                pstate = torch.load(pred_best, map_location=device, weights_only=False)
                if "predictor_state_dict" in pstate:
                    predictor.load_state_dict(pstate["predictor_state_dict"])
                    predictor_loaded = True
                    logger.info("Predictor weights loaded (posthoc)")
            # Also consider latest predictor checkpoints if best not present
            if not predictor_loaded:
                # Find latest predictor_checkpoint_epoch_*.pt
                latest_epoch = -1
                latest_path = None
                for fn in os.listdir(os.path.join(run_dir, "predictor_training")):
                    if fn.startswith("predictor_checkpoint_epoch_") and fn.endswith(".pt"):
                        try:
                            ep = int(fn.split("_")[-1].split(".")[0])
                            if ep > latest_epoch:
                                latest_epoch = ep
                                latest_path = os.path.join(run_dir, "predictor_training", fn)
                        except Exception:
                            continue
                if latest_path is not None:
                    logger.info(f"Loading predictor weights from {latest_path}")
                    pstate = torch.load(latest_path, map_location=device, weights_only=False)
                    if "predictor_state_dict" in pstate:
                        predictor.load_state_dict(pstate["predictor_state_dict"])
                        predictor_loaded = True
                        logger.info("Predictor weights loaded (latest checkpoint)")

    if predictor is not None and not predictor_loaded:
        raise RuntimeError("Failed to load predictor weights. Ensure this run trained a predictor.")


@torch.no_grad()
def encode_source_samples(
    encoder: torch.nn.Module,
    sample_entry: Dict,
    device: torch.device,
    target_set_size: int,
) -> torch.Tensor:
    # Expect dict with ESM fields
    samples = sample_entry["source_samples"]
    esm_ids = samples["esm_input_ids"]  # [set_size, L]
    esm_mask = samples["esm_attention_mask"]  # [set_size, L]

    # Subsample to desired set size if needed
    if hasattr(esm_ids, "shape") and esm_ids.shape[0] > target_set_size:
        idx = torch.randperm(esm_ids.shape[0])[:target_set_size]
        esm_ids = esm_ids[idx]
        esm_mask = esm_mask[idx]

    # Add batch dim
    batch = {
        "esm_input_ids": esm_ids.unsqueeze(0).to(device),
        "esm_attention_mask": esm_mask.unsqueeze(0).to(device),
    }

    latent = encoder(batch)  # [1, D]
    return latent.squeeze(0).detach().cpu()


@torch.no_grad()
def encode_target_samples(
    encoder: torch.nn.Module,
    sample_entry: Dict,
    device: torch.device,
    target_set_size: int,
) -> torch.Tensor:
    samples = sample_entry["target_samples"]
    esm_ids = samples["esm_input_ids"]
    esm_mask = samples["esm_attention_mask"]

    if hasattr(esm_ids, "shape") and esm_ids.shape[0] > target_set_size:
        idx = torch.randperm(esm_ids.shape[0])[:target_set_size]
        esm_ids = esm_ids[idx]
        esm_mask = esm_mask[idx]

    batch = {
        "esm_input_ids": esm_ids.unsqueeze(0).to(device),
        "esm_attention_mask": esm_mask.unsqueeze(0).to(device),
    }
    latent = encoder(batch)
    return latent.squeeze(0).detach().cpu()


class IdxMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers: List[torch.nn.Module] = []
        dim = input_dim
        if num_layers <= 1:
            layers.append(torch.nn.Linear(dim, 1))
        else:
            for _ in range(num_layers - 1):
                layers.append(torch.nn.Linear(dim, hidden_dim))
                layers.append(torch.nn.ReLU())
                dim = hidden_dim
            layers.append(torch.nn.Linear(dim, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_idx_mlp(
    features_np: np.ndarray,
    targets_np: np.ndarray,
    input_dim: int,
    device: torch.device,
    logger: logging.Logger,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 128,
    val_frac: float = 0.2,
    patience: int = 20,
) -> IdxMLP:
    x_all = features_np.astype(np.float32)
    y_all = targets_np.astype(np.float32)
    n = x_all.shape[0]
    rs = np.random.RandomState(42)
    perm = rs.permutation(n)
    n_val = int(max(1, round(val_frac * n)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    x_tr = torch.from_numpy(x_all[train_idx]).to(device)
    y_tr = torch.from_numpy(y_all[train_idx]).to(device)
    x_va = torch.from_numpy(x_all[val_idx]).to(device)
    y_va = torch.from_numpy(y_all[val_idx]).to(device)

    train_ds = torch.utils.data.TensorDataset(x_tr, y_tr)
    val_ds = torch.utils.data.TensorDataset(x_va, y_va)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = IdxMLP(input_dim=input_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.detach().item()) * xb.shape[0]
        tr_loss /= max(1, len(train_idx))

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += float(loss.detach().item()) * xb.shape[0]
        va_loss /= max(1, len(val_idx))

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"MLP epoch {epoch:03d} | train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")

        if va_loss + 1e-8 < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def build_fixed_offset_indices(dataset, offset_n: int) -> List[int]:
    if not hasattr(dataset, "index_pairs"):
        raise AttributeError("Dataset is expected to expose 'index_pairs' (as in ViralDataset).")
    pairs = dataset.index_pairs  # numpy array [M, 2]
    if not isinstance(pairs, np.ndarray):
        pairs = np.array(pairs)
    src = pairs[:, 0]
    tgt = pairs[:, 1]
    mask = (tgt - src) == offset_n
    indices = np.nonzero(mask)[0].tolist()
    return indices


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-offset mapping with index-predicting MLP")
    parser.add_argument("outputs_subdir", type=str, help="Subdirectory under ./outputs (or absolute run dir path)")
    parser.add_argument("--offset", "-n", type=int, required=True, help="Integer N for target_idx - source_idx == N")
    parser.add_argument("--epochs", type=int, default=5, help="Aggregation epochs over dataset sampling")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mlp_epochs", type=int, default=200, help="MLP training epochs")
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    logger = setup_logger()

    run_dir = resolve_outputs_dir(args.outputs_subdir)
    cfg = load_cfg_from_run_dir(run_dir)

    device = torch.device(args.device)

    # Instantiate dataset exactly as in training
    logger.info("Instantiating dataset from saved config…")
    dataset = hydra.utils.instantiate(cfg.dataset)

    # Instantiate models and load weights
    encoder, predictor = instantiate_models(cfg, device, logger)
    if predictor is None:
        raise RuntimeError("Predictor is not configured in this run; cannot proceed.")
    load_weights(encoder, predictor, run_dir, device, logger)

    # Determine set size used for encoding
    eval_set_size = int(getattr(cfg.dataset, "set_size", 16))

    # Build indices with fixed offset N
    fixed_indices = build_fixed_offset_indices(dataset, args.offset)
    print("!!!!!!!!!!!!!! FIXED INDICES: ", fixed_indices)
    if len(fixed_indices) == 0:
        raise RuntimeError(f"No dataset pairs satisfy target_idx - source_idx == {args.offset}")
    logger.info(f"Found {len(fixed_indices)} pairs with offset N={args.offset}")

    # Aggregation buffers
    source_latents: List[np.ndarray] = []
    source_indices: List[int] = []
    preds_by_pair: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
    true_tgt_by_pair: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)

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

                # Collect for MLP training (source latent -> source_idx)
                source_latents.append(src_latent_t.numpy())
                source_indices.append(src_idx)

                # Predict target latent (condition on indices if required)
                sl = torch.from_numpy(src_latent_t.numpy()).unsqueeze(0).to(device)
                # TODO: manually remove hardcoded True.
                if True:#getattr(predictor, "requires_condition", False):
                    d1 = torch.tensor([src_idx], device=device, dtype=torch.float32)
                    d2 = torch.tensor([tgt_idx], device=device, dtype=torch.float32)
                    pred_t = predictor(sl, condition_scalars=(d1, d2))
                else:
                    pred_t = predictor(sl)
                preds_by_pair[(src_idx, tgt_idx)].append(pred_t.squeeze(0).detach().cpu().numpy())

                # Encode true target latent as well (for MLP-on-true evaluation)
                tgt_latent_true = encode_target_samples(
                    encoder=encoder,
                    sample_entry=item,
                    device=device,
                    target_set_size=eval_set_size,
                )
                true_tgt_by_pair[(src_idx, tgt_idx)].append(tgt_latent_true.numpy())

                if args.log_interval > 0 and (k + 1) % args.log_interval == 0:
                    logger.info(f"Processed {k+1}/{len(order)} pairs in epoch {ep}")

    # Prepare data for MLP
    feats_np = np.stack(source_latents, axis=0)
    y_np = np.array(source_indices, dtype=np.float32)
    input_dim = feats_np.shape[1]
    logger.info(f"Training MLP on {feats_np.shape[0]} samples, input_dim={input_dim}")

    mlp = train_idx_mlp(
        features_np=feats_np,
        targets_np=y_np,
        input_dim=input_dim,
        device=device,
        logger=logger,
        epochs=args.mlp_epochs,
    )

    # Evaluate on predicted target latents; report mean ± stdev per (source_idx, target_idx)
    logger.info("Evaluating MLP on predicted target latents…")
    mlp.eval()
    with torch.no_grad():
        keys_sorted = sorted(list(preds_by_pair.keys()), key=lambda x: (x[0], x[1]))
        for (src_idx, tgt_idx) in keys_sorted:
            latents = preds_by_pair[(src_idx, tgt_idx)]
            x = torch.from_numpy(np.stack(latents, axis=0).astype(np.float32)).to(device)
            print("!!!!!!!!!!!!!! X: ", x.shape)
            preds = mlp(x).detach().cpu().numpy()
            mean_val = float(np.mean(preds))
            std_val = float(np.std(preds))
            rounded = int(np.rint(mean_val))
            # Also evaluate MLP on true encoded target latents
            true_latents = true_tgt_by_pair[(src_idx, tgt_idx)]
            x_true = torch.from_numpy(np.stack(true_latents, axis=0).astype(np.float32)).to(device)
            preds_true = mlp(x_true).detach().cpu().numpy()
            mean_true = float(np.mean(preds_true))
            std_true = float(np.std(preds_true))
            rounded_true = int(np.rint(mean_true))
            print(
                f"source_idx={src_idx:3d}  true_target_idx={tgt_idx:3d}  "
                f"mlp_pred_on_predicted(mean±std)={mean_val:.2f}±{std_val:.2f}  rounded_mean={rounded}  "
                f"|  mlp_pred_on_true(mean±std)={mean_true:.2f}±{std_true:.2f}  rounded_mean={rounded_true}"
            )


if __name__ == "__main__":
    main()


