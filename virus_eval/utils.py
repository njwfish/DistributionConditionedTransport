import os
import logging
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
import hydra


def setup_file_logger(name: str, logfile_path: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear existing handlers to avoid duplicate logs when re-imported
    while logger.handlers:
        logger.handlers.pop()
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    fh = logging.FileHandler(logfile_path)
    fh.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def setup_console_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def resolve_outputs_dir(outputs_subdir: str) -> str:
    if os.path.isdir(outputs_subdir):
        return os.path.abspath(outputs_subdir)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    predictor: Optional[torch.nn.Module],
    run_dir: str,
    device: torch.device,
    logger: logging.Logger,
    require_predictor: bool = False,
) -> None:
    ckpt_path_encoder = os.path.join(run_dir, "best_model.pt")
    if not os.path.exists(ckpt_path_encoder):
        raise FileNotFoundError(f"best_model.pt not found in {run_dir}")
    logger.info(f"Loading encoder weights from {ckpt_path_encoder}")
    try:
        state = torch.load(ckpt_path_encoder, map_location=device, weights_only=False)
    except Exception:
        state = torch.load(ckpt_path_encoder, map_location=device)

    if "encoder_state_dict" not in state:
        raise KeyError("Checkpoint missing 'encoder_state_dict'")
    encoder.load_state_dict(state["encoder_state_dict"])
    logger.info("Encoder weights loaded.")

    # Predictor (optional)
    if predictor is None:
        if require_predictor:
            raise RuntimeError("Predictor model is required but not configured in this run.")
        logger.info("No predictor configured; skipping predictor weight load.")
        return

    ckpt_path_predictor = os.path.join(run_dir, "predictor_training", "predictor_best_model.pt")
    predictor_loaded = False
    if os.path.exists(ckpt_path_predictor):
        logger.info(f"Loading predictor weights from {ckpt_path_predictor}")
        try:
            state_predictor = torch.load(ckpt_path_predictor, map_location=device, weights_only=False)
        except Exception:
            state_predictor = torch.load(ckpt_path_predictor, map_location=device)
        if "predictor_state_dict" in state_predictor:
            predictor.load_state_dict(state_predictor["predictor_state_dict"])
            predictor_loaded = True
            logger.info("Predictor weights loaded (best).")

    if not predictor_loaded:
        pred_dir = os.path.join(run_dir, "predictor_training")
        latest_epoch = -1
        latest_path = None
        if os.path.isdir(pred_dir):
            for fn in os.listdir(pred_dir):
                if fn.startswith("predictor_checkpoint_epoch_") and fn.endswith(".pt"):
                    try:
                        ep = int(fn.split("_")[-1].split(".")[0])
                        if ep > latest_epoch:
                            latest_epoch = ep
                            latest_path = os.path.join(pred_dir, fn)
                    except Exception:
                        continue
        if latest_path is not None:
            logger.info(f"Loading predictor weights from {latest_path}")
            pstate = torch.load(latest_path, map_location=device, weights_only=False)
            if "predictor_state_dict" in pstate:
                predictor.load_state_dict(pstate["predictor_state_dict"])
                predictor_loaded = True
                logger.info("Predictor weights loaded (latest checkpoint).")

    if require_predictor and not predictor_loaded:
        raise RuntimeError("Failed to load predictor weights. Ensure this run trained a predictor.")


@torch.no_grad()
def encode_source_samples(
    encoder: torch.nn.Module,
    sample_entry: Dict,
    device: torch.device,
    target_set_size: int,
) -> torch.Tensor:
    samples = sample_entry["source_samples"]
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


def build_fixed_offset_indices(dataset, offset_n: int) -> List[int]:
    if not hasattr(dataset, "index_pairs"):
        raise AttributeError("Dataset is expected to expose 'index_pairs' (as in ViralDataset).")
    pairs = dataset.index_pairs
    if not isinstance(pairs, np.ndarray):
        pairs = np.array(pairs)
    src = pairs[:, 0]
    tgt = pairs[:, 1]
    mask = (tgt - src) == offset_n
    indices = np.nonzero(mask)[0].tolist()
    return indices


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


