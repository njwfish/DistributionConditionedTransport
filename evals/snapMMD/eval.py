import os
import sys
import argparse
import logging
import yaml
import json
import itertools
import copy
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from utils.snapMMD import MMDLoss, RBF
from utils.experiment_utils import load_best_model, get_experiment_info
import hydra
from omegaconf import OmegaConf

from TrajectoryNet.optimal_transport.emd import earth_mover_distance



# -----------------------------
# Configuration helpers
# -----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_hydra_resolved_config(path: str) -> Dict[str, Any]:
    """Load a Hydra/OmegaConf YAML config and resolve interpolations like ${a.b}."""
    cfg = OmegaConf.load(path)
    # resolve=True replaces interpolations with their concrete values
    resolved: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    return resolved  # fully standard Python containers


def dict_contains(superset: Any, subset: Any) -> bool:
    """Recursively check if 'subset' is contained in 'superset'."""
    if isinstance(subset, dict):
        if not isinstance(superset, dict):
            return False
        for k, v in subset.items():
            if k not in superset:
                return False
            if not dict_contains(superset[k], v):
                return False
        return True
    elif isinstance(subset, list):
        # For lists, require exact match of all items (order-sensitive)
        return isinstance(superset, list) and superset == subset
    else:
        # Scalars: allow float tolerance
        if isinstance(subset, float) and isinstance(superset, (float, int)):
            return abs(float(superset) - subset) < 1e-8
        return superset == subset


def remove_key_recursive(d: Any, key_to_remove: str) -> Any:
    if isinstance(d, dict):
        return {k: remove_key_recursive(v, key_to_remove) for k, v in d.items() if k != key_to_remove}
    elif isinstance(d, list):
        return [remove_key_recursive(x, key_to_remove) for x in d]
    else:
        return d


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> List[Tuple[str, Any]]:
    """Flatten a nested dict preserving insertion order. Returns list of (key, value)."""
    items: List[Tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep))
        else:
            items.append((new_key, v))
    return items


def sanitize_value_for_name(v: Any) -> str:
    if isinstance(v, (int, float)):
        out = ("%g" % v)
    else:
        out = str(v)
    out = out.replace("/", "-").replace(" ", "_")
    return out


def extract_seed(cfg: Dict[str, Any]) -> int:
    # Prefer top-level 'seed'
    if isinstance(cfg, dict) and 'seed' in cfg and cfg['seed'] is not None:
        return int(cfg['seed'])
    # Common fallbacks
    try_paths = [
        ['dataset', 'seed'],
        ['experiment', 'seed'],
        ['training', 'seed']
    ]
    for path in try_paths:
        cur = cfg
        ok = True
        for p in path:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                ok = False
                break
        if ok and cur is not None:
            return int(cur)
    raise ValueError("Could not extract 'seed' from config")


def extract_dataset_name(cfg: Dict[str, Any]) -> str:
    if isinstance(cfg, dict) and 'dataset' in cfg and isinstance(cfg['dataset'], dict):
        ds = cfg['dataset']
        if 'dataset_name' in ds and ds['dataset_name'] is not None:
            return str(ds['dataset_name'])
    # Fallbacks
    if 'dataset_name' in cfg:
        return str(cfg['dataset_name'])
    raise ValueError("Could not extract 'dataset.dataset_name' from config")


# -----------------------------
# Data config (paths + plotting info)
# -----------------------------

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'LV': {
        # TODO: avoid hard-coding the data path.
        'data_path': 'data/classic/LV_data.npz',
        #'data_path': 'data/classic/LV_simulated_dataset_1000_31_seed0.npz',
        #'data_path': 'data/classic/LV_simulated_dataset_1000_11_seed0.npz',
        'dimensionality': 2,
        'axes_labels': ['Prey', 'Predator'],
        'title': 'Lotka-Volterra',
        'calculate_emd': True,
    },
    'Repressilator': {
        'data_path': 'data/classic/Repressilator_data.npz',
        'dimensionality': 3,
        'axes_labels': ['Gene 1', 'Gene 2', 'Gene 3'],
        'title': 'Repressilator',
        'calculate_emd': True,
    },
    'GoM': {
        'data_path': 'data/realdata/GoM_data.npz',
        'dimensionality': 2,
        'axes_labels': ['X1', 'X2'],
        'title': 'GoM',
        'calculate_emd': True,
    },
    'pbmc': {
        'data_path': 'data/realdata/processed_pbmc_data_sub500_every_2_until20.npz',
        'dimensionality': 30,
        'plot_dimensionality': 3,
        'axes_labels': ['PC1', 'PC2', 'PC3'],
        'title': 'PBMC',
        'calculate_emd': False,
        'requires_pca': True,
    },
}


# -----------------------------
# Model loading and forecasting (CDE only)
# -----------------------------

def load_models_from_experiment(experiment_dir: str, device: torch.device, predictor_source: str = 'separate', predictor_dir: Optional[str] = None, use_true_target_latent: bool = False, use_separate_predictor_dir: bool = True):
    info = get_experiment_info(experiment_dir, load_checkpoints=False)
    cfg = info['config']

    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])
    dataset = hydra.utils.instantiate(cfg['dataset'])

    # Resolve config for simple key access
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg = cfg_resolved.get('experiment', {}) if isinstance(cfg_resolved, dict) else {}
    train_predictor_posthoc = bool(experiment_cfg.get('train_predictor_posthoc', False))

    # Instantiate predictor based on predictor_source
    predictor = None

    if not use_true_target_latent:
        if predictor_source == 'separate':
            if predictor_dir is not None:
                pred_cfg_path = os.path.join(predictor_dir, 'config.yaml')
                pred_cfg_oc = OmegaConf.load(pred_cfg_path)
                pred_cfg_node = pred_cfg_oc.get('predictor')
                predictor = hydra.utils.instantiate(pred_cfg_node)
        elif predictor_source == 'main':
            # Instantiate predictor from the main experiment config
            pred_cfg_node = cfg['predictor']
            
            predictor = hydra.utils.instantiate(pred_cfg_node)

        if predictor is not None and hasattr(enc, 'latent_act'):
            predictor.latent_act = enc.latent_act

    # Load encoder/generator from best model
    state = load_best_model(info['dir'])
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])

    # Load predictor weights from the selected source
    if not use_true_target_latent and predictor is not None:
        if predictor_source == 'separate':
            if predictor_dir is None:
                raise ValueError("predictor_source='separate' but predictor_dir is None")
            pred_ckpt = torch.load(os.path.join(predictor_dir, 'predictor_best_model.pt'), map_location=device, weights_only=False)
            predictor.load_state_dict(pred_ckpt['predictor_state_dict'])
        elif predictor_source == 'main':
            if 'predictor_state_dict' not in state:
                raise KeyError("predictor_state_dict not found in main checkpoint")
            predictor.load_state_dict(state['predictor_state_dict'])
           
    enc.eval(); gen.eval()
    enc.to(device); gen.to('cpu')
    if predictor is not None:
        predictor.eval(); predictor.to(device)

    return cfg, enc, gen, predictor, dataset, train_predictor_posthoc


class UnconditionedRidgePredictor(torch.nn.Module):
    """Torch wrapper for an unconditioned ridge regressor.
    Supports either a bare sklearn estimator, or a bundle dict with
    {'model', 'x_scaler', 'y_scaler', 'normalize_latents'}.
    """
    def __init__(self, ridge_obj):
        super().__init__()
        # Accept either model or bundle
        if isinstance(ridge_obj, dict) and ('model' in ridge_obj):
            self.model = ridge_obj.get('model')
            self.x_scaler = ridge_obj.get('x_scaler')
            self.y_scaler = ridge_obj.get('y_scaler')
            self.normalize_latents = bool(ridge_obj.get('normalize_latents', False))
        else:
            self.model = ridge_obj
            self.x_scaler = None
            self.y_scaler = None
            self.normalize_latents = False
        self._device = torch.device('cpu')
        self.condition_type = 'none'

    def to(self, device):
        self._device = device
        return self

    def _l2_normalize_rows(self, x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        safe_norms = np.where((~np.isfinite(norms)) | (norms <= 0.0), 1.0, norms)
        return x / safe_norms

    def forward(self, latent_source: torch.Tensor, condition_scalars: Optional[tuple] = None) -> torch.Tensor:
        with torch.no_grad():
            x_np = latent_source.detach().cpu().numpy()
            if self.normalize_latents:
                x_np = self._l2_normalize_rows(x_np)
            if self.x_scaler is not None:
                x_np_scaled = self.x_scaler.transform(x_np)
            else:
                x_np_scaled = x_np
            y_np_scaled = self.model.predict(x_np_scaled)
            if self.y_scaler is not None:
                y_np = self.y_scaler.inverse_transform(y_np_scaled)
            else:
                y_np = y_np_scaled
            y_t = torch.from_numpy(y_np).to(latent_source.device, dtype=latent_source.dtype)
            return y_t

def generate_cde_forecast(experiment_dir: str, training_data: Dict[str, Any], predictor_source: str = 'separate', use_true_target_latent: bool = False, forecast_all_timepoints: bool = False, predictor_dir: Optional[str] = None, source_steps_back: int = 1, num_sets: int = 1, cli_set_size: Optional[int] = None, override_predictor: Optional[torch.nn.Module] = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, enc, gen, predictor, dataset, train_predictor_posthoc = load_models_from_experiment(experiment_dir, device, predictor_source=predictor_source, predictor_dir=predictor_dir, use_true_target_latent=use_true_target_latent)

    # If provided, override predictor with external (e.g., ridge) predictor
    if override_predictor is not None:
        predictor = override_predictor.to(device)
        predictor.eval()

    if not use_true_target_latent and predictor is None:
        print(f"[forecast] Skipping run at {experiment_dir}: predictor=None and use_true_target_latent=False.")
        return None

    Xs_training = training_data['Xs']
    n_steps = int(training_data['N_steps'])
    # Validate and compute source index b steps back from the last training timepoint
    if not (1 <= int(source_steps_back) <= (n_steps - 1)):
        raise ValueError(f"source_steps_back must be in [1, {n_steps - 1}], got {source_steps_back}")
    source_idx_value = (n_steps - 1) - int(source_steps_back)  # maps 1->n_steps-2, 2->n_steps-3, ...

    # Resolve set_size from config, falling back to dataset attribute if needed
    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    set_size_cfg = None
    if isinstance(cfg_resolved, dict):
        exp_cfg = cfg_resolved.get('experiment', {})
        if isinstance(exp_cfg, dict) and exp_cfg.get('set_size') is not None:
            set_size_cfg = int(exp_cfg['set_size'])
    if set_size_cfg is None:
        # TODO: attention, hardcoded 32.
        set_size_cfg = int(getattr(dataset, 'set_size', 32))

    # NEW: Override with CLI argument if provided
    if cli_set_size is not None:
        set_size_cfg = cli_set_size

    # Build non-overlapping subsets that cover each source element exactly once
    num_available = int(Xs_training[source_idx_value].shape[0])
    perm_indices = np.random.permutation(num_available)
    num_full_sets = num_available // set_size_cfg
    remainder = num_available % set_size_cfg

    full_subsets: List[np.ndarray] = [
        perm_indices[i * set_size_cfg:(i + 1) * set_size_cfg]
        for i in range(num_full_sets)
    ]

    leftover_indices: Optional[np.ndarray] = None
    padded_latent_indices: Optional[np.ndarray] = None
    if remainder > 0:
        leftover_indices = perm_indices[num_full_sets * set_size_cfg:]
        pad_needed = set_size_cfg - remainder
        if num_full_sets > 0:
            used_pool = perm_indices[:num_full_sets * set_size_cfg]
            pad_extra = np.random.choice(used_pool, size=pad_needed, replace=False)
        else:
            # If no full sets exist, pad for latents from the leftover itself (allow repeats)
            pad_extra = np.random.choice(leftover_indices, size=pad_needed, replace=True)
        padded_latent_indices = np.concatenate([leftover_indices, pad_extra], axis=0)

    # Aggregate forecasts across all constructed subsets
    aggregated_forecasts: List[np.ndarray] = []

    # Process full subsets (latents and generator both of size set_size_cfg)
    for subset_indices in full_subsets:
        src_subset_np_lat = Xs_training[source_idx_value][subset_indices]
        src_subset_t_lat = torch.tensor(src_subset_np_lat, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc_s = enc(src_subset_t_lat)
            if use_true_target_latent:
                # TODO: why are you not doing the unsqueezing like for src_subset_t_lat here?
                tgt_subset_np = training_data['X_val_true'][subset_indices]
                tgt_subset_t = torch.tensor(tgt_subset_np, dtype=torch.float32, device=device).unsqueeze(0)
                enc_t = enc(tgt_subset_t)
                
            else:
                if predictor is not None:
                    # Compute conditioning based on predictor.condition_type
                    if getattr(predictor, 'condition_type', None) == 'index_pair':
                        source_idx = torch.tensor([source_idx_value], device=device, dtype=torch.float32)
                        target_idx = torch.tensor([n_steps - 1], device=device, dtype=torch.float32)
                        condition = (source_idx, target_idx)
                    elif getattr(predictor, 'condition_type', None) == 'scalar_d':
                        d_val = dataset.d_fun(source_idx_value, n_steps - 1)
                        d_tensor = torch.tensor([d_val], device=device, dtype=torch.float32)
                        condition = (d_tensor,)
                    else:
                        condition = None
                    enc_t = predictor(enc_s, condition_scalars=condition)
                
                else: 
                    raise ValueError("Predictor is None and use_true_target_latent is False")

            # Generator input uses the exact subset (no duplicates)
            src_subset_t_gen = src_subset_t_lat
            _, set_size_cur, *data_shape = src_subset_t_gen.shape
            gen_src = src_subset_t_gen.reshape(-1, *data_shape)

            gen.to(device)
            pred_subset = gen.sample(gen_src, enc_s, enc_t)[0]
            gen.to('cpu')
            pred_subset_np = pred_subset.detach().cpu().numpy()
            print("!!!!!!!!!!!!!!!!! PREDICTION SHAPE FOR FORECAST!!!!!!!!!!!!!!!", pred_subset_np.shape, pred_subset_np[None, :, :].shape)
            aggregated_forecasts.append(pred_subset_np)

    # Process leftover subset (generator sees only remaining elements; latents padded to set_size_cfg)
    if leftover_indices is not None and padded_latent_indices is not None:
        src_subset_np_lat = Xs_training[source_idx_value][padded_latent_indices]
        src_subset_t_lat = torch.tensor(src_subset_np_lat, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            enc_s = enc(src_subset_t_lat)
            if use_true_target_latent:
                # TODO: why are you not doing the unsqueezing like for src_subset_t_lat here?
                tgt_subset_np = training_data['X_val_true'][padded_latent_indices]
                tgt_subset_t = torch.tensor(tgt_subset_np, dtype=torch.float32, device=device).unsqueeze(0)
                enc_t = enc(tgt_subset_t)
            else:
                if predictor is not None:
                    if getattr(predictor, 'condition_type', None) == 'index_pair':
                        source_idx = torch.tensor([source_idx_value], device=device, dtype=torch.float32)
                        target_idx = torch.tensor([n_steps - 1], device=device, dtype=torch.float32)
                        condition = (source_idx, target_idx)
                    elif getattr(predictor, 'condition_type', None) == 'scalar_d':
                        d_val = dataset.d_fun(source_idx_value, n_steps - 1)
                        d_tensor = torch.tensor([d_val], device=device, dtype=torch.float32)
                        condition = (d_tensor,)
                    else:
                        condition = None
                    enc_t = predictor(enc_s, condition_scalars=condition)


            # Generator sees only the true leftover indices (no duplicates)
            src_subset_np_gen = Xs_training[source_idx_value][leftover_indices]
            src_subset_t_gen = torch.tensor(src_subset_np_gen, dtype=torch.float32, device=device).unsqueeze(0)
            _, set_size_cur, *data_shape = src_subset_t_gen.shape
            gen_src = src_subset_t_gen.reshape(-1, *data_shape)

            gen.to(device)
            pred_subset = gen.sample(gen_src, enc_s, enc_t)[0]
            gen.to('cpu')
            pred_subset_np = pred_subset.detach().cpu().numpy()
            print("!!!!!!!!!!!!!!!!! PREDICTION SHAPE FOR FORECAST (LEFTOVER)!!!!!!!!!!!!!!!", pred_subset_np.shape)
            aggregated_forecasts.append(pred_subset_np)

    # Concatenate all subset forecasts into a single aggregated set (N, D)
    forecast_structured = np.vstack(aggregated_forecasts)
    print("!!!!!!!!!!!!!!!!! FINAL PREDICTION SHAPE!!!!!!!!!!!!!!!",forecast_structured.shape)
    results: Dict[str, Any] = {
        'forecast': forecast_structured,  # (N, D)
        'X_val': training_data['X_val_true']
    }

    # Optionally produce forecasts for all timepoints (training + final)
    if forecast_all_timepoints:
        n_steps = int(training_data['N_steps'])
        forecasts_seq: List[np.ndarray] = []
        for t in range(1, n_steps):
            # Aggregate per-timepoint forecasts using disjoint subsets that cover all elements once
            per_t_agg: List[np.ndarray] = []
            source_full_np = training_data['Xs'][t - 1]
            target_full_np = training_data['Xs'][t] if t < n_steps - 1 else training_data['X_val_true']

            num_available_t = int(source_full_np.shape[0])
            perm_t = np.random.permutation(num_available_t)
            num_full_t = num_available_t // set_size_cfg
            remainder_t = num_available_t % set_size_cfg

            full_sets_t: List[np.ndarray] = [
                perm_t[i * set_size_cfg:(i + 1) * set_size_cfg]
                for i in range(num_full_t)
            ]

            leftover_t: Optional[np.ndarray] = None
            padded_latent_t: Optional[np.ndarray] = None
            if remainder_t > 0:
                leftover_t = perm_t[num_full_t * set_size_cfg:]
                pad_needed_t = set_size_cfg - remainder_t
                if num_full_t > 0:
                    used_pool_t = perm_t[:num_full_t * set_size_cfg]
                    pad_extra_t = np.random.choice(used_pool_t, size=pad_needed_t, replace=False)
                else:
                    pad_extra_t = np.random.choice(leftover_t, size=pad_needed_t, replace=True)
                padded_latent_t = np.concatenate([leftover_t, pad_extra_t], axis=0)

            # Process full sets
            for subset_indices_t in full_sets_t:
                src_subset_np_t_lat = source_full_np[subset_indices_t]
                src_subset_t_lat = torch.tensor(src_subset_np_t_lat, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    enc_src = enc(src_subset_t_lat)
                    if use_true_target_latent:
                        tgt_subset_np_t_lat = target_full_np[subset_indices_t]
                        tgt_subset_t = torch.tensor(tgt_subset_np_t_lat, dtype=torch.float32, device=device).unsqueeze(0)
                        enc_tgt = enc(tgt_subset_t)
                        
                    else:
                        if predictor is not None:
                            if getattr(predictor, 'condition_type', None) == 'index_pair':
                                src_idx = torch.tensor([t - 1], device=device, dtype=torch.float32)
                                tgt_idx = torch.tensor([t], device=device, dtype=torch.float32)
                                condition = (src_idx, tgt_idx)
                            elif getattr(predictor, 'condition_type', None) == 'scalar_d':
                                d_val = dataset.d_fun(t - 1, t)
                                d_tensor = torch.tensor([d_val], device=device, dtype=torch.float32)
                                condition = (d_tensor,)
                            else:
                                condition = None
                            enc_tgt = predictor(enc_src, condition_scalars=condition)
                    
                        else:
                            raise ValueError("Predictor is None and use_true_target_latent is False")

                    # Reshape and sample
                    src_subset_t_gen = src_subset_t_lat
                    _, set_size_t, *data_shape_t = src_subset_t_gen.shape
                    gen_src_t = src_subset_t_gen.reshape(-1, *data_shape_t)
                    gen.to(device)
                    pred_t = gen.sample(gen_src_t, enc_src, enc_tgt)[0]
                    gen.to('cpu')
                    print("!!!!!!!!!!!!!!!!! PREDICTION SHAPE!!!!!!!!!!!!!!!", pred_t.shape, pred_t.detach().cpu().numpy()[None, :, :].shape)
                    per_t_agg.append(pred_t.detach().cpu().numpy())

            # Process leftover set if present
            if leftover_t is not None and padded_latent_t is not None:
                src_subset_np_t_lat = source_full_np[padded_latent_t]
                src_subset_t_lat = torch.tensor(src_subset_np_t_lat, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    enc_src = enc(src_subset_t_lat)
                    if use_true_target_latent:
                        tgt_subset_np_t_lat = target_full_np[padded_latent_t]
                        tgt_subset_t = torch.tensor(tgt_subset_np_t_lat, dtype=torch.float32, device=device).unsqueeze(0)
                        enc_tgt = enc(tgt_subset_t)
                        
                    else:
                        if predictor is not None:
                            if getattr(predictor, 'condition_type', None) == 'index_pair':
                                src_idx = torch.tensor([t - 1], device=device, dtype=torch.float32)
                                tgt_idx = torch.tensor([t], device=device, dtype=torch.float32)
                                condition = (src_idx, tgt_idx)
                            elif getattr(predictor, 'condition_type', None) == 'scalar_d':
                                d_val = dataset.d_fun(t - 1, t)
                                d_tensor = torch.tensor([d_val], device=device, dtype=torch.float32)
                                condition = (d_tensor,)
                            else:
                                condition = None
                            enc_tgt = predictor(enc_src, condition_scalars=condition)
                        else:
                            raise ValueError("Predictor is None and use_true_target_latent is False")

                    # Generator sees only leftover indices
                    src_subset_np_t_gen = source_full_np[leftover_t]
                    src_subset_t_gen = torch.tensor(src_subset_np_t_gen, dtype=torch.float32, device=device).unsqueeze(0)
                    _, set_size_t, *data_shape_t = src_subset_t_gen.shape
                    gen_src_t = src_subset_t_gen.reshape(-1, *data_shape_t)
                    gen.to(device)
                    pred_t = gen.sample(gen_src_t, enc_src, enc_tgt)[0]
                    gen.to('cpu')
                    print("!!!!!!!!!!!!!!!!! PREDICTION SHAPE (LEFTOVER)!!!!!!!!!!!!!!!", pred_t.shape)
                    per_t_agg.append(pred_t.detach().cpu().numpy())

            # Concatenate aggregated per-timepoint predictions (N, D)
            forecasts_seq.append(np.vstack(per_t_agg))
            print("!!!!!!!!!!!!!!!!! FINAL PREDICTION SHAPE INTERMEDIATE!!!!!!!!!!!!!!!", np.vstack(per_t_agg).shape)


        if len(forecasts_seq) > 0:
            # Shape: (n_steps-1, N, D)
            results['forecast_sequence'] = np.stack(forecasts_seq, axis=0)

    return results


# -----------------------------
# Metrics
# -----------------------------

def calculate_emd_my_implementation(x: np.ndarray, y: np.ndarray) -> float:
    # Linear programming EMD (balanced, uniform weights)
    from scipy.optimize import linprog
    n, m = x.shape[0], y.shape[0]
    C = np.linalg.norm(x[:, None] - y[None, :], axis=2).ravel()
    A_eq = []
    b_eq = []
    for i in range(n):
        row = np.zeros(n * m)
        row[i * m:(i + 1) * m] = 1
        A_eq.append(row)
        b_eq.append(1 / n)
    for j in range(m):
        row = np.zeros(n * m)
        row[j::m] = 1
        A_eq.append(row)
        b_eq.append(1 / m)
    res = linprog(C, A_eq=np.vstack(A_eq), b_eq=np.array(b_eq), bounds=(0, None), method='highs')
    if res.success:
        return float(res.fun)
    else:
        return float('nan')

def calculate_emd(x: np.ndarray, y: np.ndarray) -> float:
    return earth_mover_distance(x, y)

def compute_mmd_and_emd(dataset_name: str, forecast_NxD: np.ndarray, logger: logging.Logger, enable_emd: bool = True) -> Tuple[float, float]:
    cfg = DATASET_CONFIGS[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training data (ground-truth final timepoint)
    if dataset_name == 'pbmc':
        training_data = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        training_data = np.load(cfg['data_path'])

    X_val = training_data['Xs'][-1]

    rbf_legacy = RBF(bandwidth=2.0).to(device)
    rbf_paper = RBF(1.0).to(device)
    mmd_loss_legacy = MMDLoss(kernel=rbf_legacy).to(device)
    mmd_loss_paper = MMDLoss(kernel=rbf_paper).to(device)

    # forecast_NxD is numpy with shape (N, D)
    forecast_final_t = torch.from_numpy(forecast_NxD).to(device)
    X_val_t = torch.tensor(X_val).to(device)
    mmd_legacy = np.sqrt(mmd_loss_legacy(forecast_final_t, X_val_t).item())
    mmd_paper = mmd_loss_paper(forecast_final_t, X_val_t).item()
    

    emd = None
    if enable_emd and cfg.get('calculate_emd', False):

        forecast_for_emd = forecast_NxD
        X_val_for_emd = X_val

        # EMD expects numpy arrays with shape (N, D) and (M, D)
        emd_val = calculate_emd(forecast_for_emd, X_val_for_emd)
        emd = float(emd_val) if not np.isnan(emd_val) else None

    logger.info(f"Computed metrics -> MMD_legacy: {mmd_legacy:.6f}, MMD_paper: {mmd_paper:.6f}, EMD: {('%.6f' % emd) if emd is not None else 'n/a'}")
    return mmd_legacy, mmd_paper, emd


# -----------------------------
# PCA helpers for PBMC plotting
# -----------------------------

def setup_pca_for_pbmc(logger: logging.Logger) -> PCA:
    logger.info("Computing PCA for PBMC datasets...")
    data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
    Xs1 = data1["Xs"]; Xs2 = data2["Xs"]
    #if Xs1.shape[0] == 21 and Xs2.shape[0] == 20:
    #    Xs1, Xs2 = Xs2, Xs1
    Xs_combined = np.concatenate([Xs1, Xs2], axis=0)
    n_timepoints, n_cells, n_genes = Xs_combined.shape
    X_reshaped = Xs_combined.reshape(n_timepoints * n_cells, n_genes)
    pca = PCA(n_components=3)
    pca.fit(X_reshaped)
    logger.info(f"PCA total explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca


def transform_for_plot(dataset_name: str, data: np.ndarray, pca: PCA = None) -> np.ndarray:
    if dataset_name == 'pbmc' and pca is not None:
        return pca.transform(data)
    return data


# -----------------------------
# Plotting
# -----------------------------

def plot_main_results(dataset_name: str, results: Dict[str, Any], out_dir: str, logger: logging.Logger, pca: PCA = None, title_suffix: str = "", plot_all_timepoints: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASET_CONFIGS[dataset_name]
    training = results['training_data']
    forecast = results['forecast_data']

    is_3d = cfg.get('plot_dimensionality', cfg['dimensionality']) == 3
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    fig.suptitle(f"{cfg['title']} Results (CDE){(' | ' + title_suffix) if title_suffix else ''}")

    # Plot colored training points across time
    Xs = training['Xs']
    n_sequences = len(Xs)
    all_training = []
    colors = []
    for i, X in enumerate(Xs):
        Xp = transform_for_plot(dataset_name, X, pca)
        all_training.append(Xp)
        colors.extend([i + 1] * len(Xp))
    all_training = np.concatenate(all_training, axis=0)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",all_training.shape)
    if is_3d:
        scatter = ax.scatter(all_training[:, 0], all_training[:, 1], all_training[:, 2], alpha=0.7, s=3.0, c=colors, cmap='coolwarm', vmin=1, vmax=n_sequences)
    else:
        scatter = ax.scatter(all_training[:, 0], all_training[:, 1], alpha=0.7, s=3.0, c=colors, cmap='coolwarm', vmin=1, vmax=n_sequences)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Time Point', rotation=270, labelpad=15)

    if plot_all_timepoints and ('forecast_sequence' in forecast and forecast['forecast_sequence'] is not None):
        forecast_seq = forecast['forecast_sequence']  # shape: (T, N, D) with T=n_steps-1
        n_seq_training = len(training['Xs'])  # equals T
        first_overlay = True
        for idx in range(forecast_seq.shape[0]):
            if idx < n_seq_training - 1:
                true_np = training['Xs'][idx + 1]
            else:
                true_np = training['X_val_true']

            pred_np = forecast_seq[idx]
            true_p = transform_for_plot(dataset_name, true_np, pca)
            pred_p = transform_for_plot(dataset_name, pred_np, pca)

            gt_label = 'Ground Truth (all times)' if first_overlay else None
            fc_label = 'Forecast (all times)' if first_overlay else None

            if is_3d:
                ax.scatter(true_p[:, 0], true_p[:, 1], true_p[:, 2], alpha=0.7, s=6.0, color='darkgreen', label=gt_label, marker='o', edgecolor='white', linewidth=0.3)
                ax.scatter(pred_p[:, 0], pred_p[:, 1], pred_p[:, 2], alpha=0.7, s=6.0, color='darkorange', label=fc_label, marker='s', edgecolor='white', linewidth=0.3)
            else:
                ax.scatter(true_p[:, 0], true_p[:, 1], alpha=0.7, s=6.0, color='darkgreen', label=gt_label, marker='o', edgecolor='white', linewidth=0.3)
                ax.scatter(pred_p[:, 0], pred_p[:, 1], alpha=0.7, s=6.0, color='darkorange', label=fc_label, marker='s', edgecolor='white', linewidth=0.3)
            first_overlay = False
        if not is_3d:
            ax.grid(True)
    else:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",training['X_val_true'].shape, forecast['forecast'].shape)
        true_data = transform_for_plot(dataset_name, training['X_val_true'], pca)
        forecast_data = transform_for_plot(dataset_name, forecast['forecast'], pca)

        if is_3d:
            ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5)
            ax.scatter(forecast_data[:, 0], forecast_data[:, 1], forecast_data[:, 2], alpha=0.9, s=8.0, color='darkorange', label='Forecast', marker='s', edgecolor='white', linewidth=0.5)
            ax.set_zlabel(cfg['axes_labels'][2] if len(cfg['axes_labels']) > 2 else 'Z')
        else:
            ax.scatter(true_data[:, 0], true_data[:, 1], alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5)
            ax.scatter(forecast_data[:, 0], forecast_data[:, 1], alpha=0.9, s=8.0, color='darkorange', label='Forecast', marker='s', edgecolor='white', linewidth=0.5)
            ax.grid(True)

    ax.set_xlabel(cfg['axes_labels'][0])
    ax.set_ylabel(cfg['axes_labels'][1])
    ax.set_title('Training Data, Ground Truth & Forecast Phase Portrait')
    ax.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{dataset_name}_results_CDE.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Main figure saved to: {out_path}")


# -----------------------------
# Core logic
# -----------------------------

def build_output_folder_name(experiment_name: str, match_criteria: Dict[str, Any], naming_parameters: List[str] = None) -> str:
    if naming_parameters is None or len(naming_parameters) == 0:
        flat = flatten_dict(match_criteria)
        parts = [experiment_name] + [sanitize_value_for_name(v) for (_, v) in flat]
    else:
        parts = [experiment_name]
        for path in naming_parameters:
            # Walk the nested dict using the dot path
            cur: Any = match_criteria
            for p in path.split('.'):
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            parts.append(sanitize_value_for_name(cur))
    return "_".join(parts)


def build_parameters_label(match_criteria: Dict[str, Any], naming_parameters: List[str] = None) -> str:
    """Build a human-readable label of the naming parameters for figure titles.
    If naming_parameters is empty/None, include all flattened key=value pairs from match_criteria.
    """
    def value_to_str(v: Any) -> str:
        if isinstance(v, (int, float)):
            return ("%g" % v)
        return str(v)

    if naming_parameters is None or len(naming_parameters) == 0:
        flat = flatten_dict(match_criteria)
        parts = [f"{k}={value_to_str(v)}" for (k, v) in flat]
    else:
        parts = []
        for path in naming_parameters:
            cur: Any = match_criteria
            for p in path.split('.'):
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                else:
                    cur = None
                    break
            parts.append(f"{path}={value_to_str(cur)}")
    return ", ".join(parts)


def summarize_differences(configs: List[Dict[str, Any]]) -> str:
    # Remove all 'seed' keys recursively
    cleaned = [remove_key_recursive(c, 'seed') for c in configs]
    # Flatten and collect values per key
    values_by_key: Dict[str, List[Any]] = {}
    for c in cleaned:
        for k, v in flatten_dict(c):
            values_by_key.setdefault(k, []).append(v)
    # Keep only keys with more than one unique value
    diffs = {}
    for k, vals in values_by_key.items():
        uniq = []
        seen = set()
        for v in vals:
            s = json.dumps(v, sort_keys=True, default=str)
            if s not in seen:
                seen.add(s)
                uniq.append(v)
        if len(uniq) > 1:
            diffs[k] = uniq
    # Format message
    if not diffs:
        return ""
    lines = ["Ambiguous matches: configurations differ on the following keys (excluding 'seed'):"]
    for k, vals in sorted(diffs.items()):
        preview = ", ".join([sanitize_value_for_name(v) for v in vals])
        lines.append(f"  - {k}: {preview}")
    return "\n".join(lines)


def load_training_data_for_dataset(dataset_name: str) -> Dict[str, Any]:
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} is not supported. Supported: {list(DATASET_CONFIGS.keys())}")
    cfg = DATASET_CONFIGS[dataset_name]
    training_data = np.load(cfg['data_path'])
    N_steps = training_data['N_steps']
    Xs_training = [training_data["Xs"][i] for i in range(N_steps - 1)]
    X_val_true = training_data["Xs"][ -1]
    return {
        'N_steps': N_steps,
        'Xs': Xs_training,
        'X_val_true': X_val_true,
        'dts': training_data['dts'],
        'y0': training_data['y0'],
        'time_scale': training_data['time_scale']
    }


def setup_logger(out_dir: str, log_name: str) -> logging.Logger:
    os.makedirs(out_dir, exist_ok=True)
    logger = logging.getLogger(f"analysis_{log_name}")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(os.path.join(out_dir, f"{log_name}.log"), mode='w')
    fh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


# -----------------------------
# Predictor subdir discovery
# -----------------------------

def find_matching_predictor_subdir(experiment_dir: str, predictor_match_criteria: Dict[str, Any], seed: Optional[int]) -> Optional[str]:
    """Find a hashed predictor subdirectory under experiment_dir that matches the given criteria.
    Prefers a match with the same seed when multiple candidates are available.
    Returns the full path or None if not found.
    """
    if not os.path.isdir(experiment_dir):
        print(f"[predictor-discovery] Experiment directory does not exist or is not a directory: {experiment_dir}")
        return None

    candidates: List[str] = []
    
    for d in os.listdir(experiment_dir):
        if d.startswith('predictor_training_'):
            full = os.path.join(experiment_dir, d)
            if os.path.isdir(full):
                candidates.append(full)

    if not candidates:
        print(f"[predictor-discovery] No predictor_training_* subdirectories found under {experiment_dir}; seed={seed}")

    matched: List[Tuple[str, Dict[str, Any]]] = []
    for cand in candidates:
        cfg_path = os.path.join(cand, 'config.yaml')
        if not os.path.exists(cfg_path):
            continue
        # TODO: understand what the difference between the two different config loading functions is.
        try:
            cfg = load_hydra_resolved_config(cfg_path)
        except Exception:
            try:
                cfg = load_yaml(cfg_path)
            except Exception:
                print(f"[predictor-discovery] Failed to read predictor config at {cfg_path}; skipping")
                continue
        if predictor_match_criteria:
            if dict_contains(cfg, predictor_match_criteria):
                matched.append((cand, cfg))
        else:
            matched.append((cand, cfg))

    if not matched:
        print(f"[predictor-discovery] Found {len(candidates)} predictor dirs but none matched criteria={json.dumps(predictor_match_criteria)}; seed={seed}")
        return None

    # If multiple remain, ensure they are identical modulo seed
    if len(matched) > 1:
        diff_msg = summarize_differences([cfg for _, cfg in matched])
        # If truly ambiguous, raise to prompt explicit criteria
        raise ValueError("Multiple predictor subdirectories match the criteria.\n" + diff_msg)

    return matched[0][0]


def main():
    parser = argparse.ArgumentParser(description='Flexible analysis of CDE results across seeds')
    parser.add_argument('--config', type=str, default='analysis_config.yaml', help='Path to analysis config file')
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Directory containing experiment subdirectories')
    parser.add_argument('--predictor-source', type=str, default='separate', help="Where to load the predictor from: 'separate' (use predictor_training_* subdir config + checkpoint) or 'main' (use main experiment config + checkpoint)")
    parser.add_argument('--skip-plots', action='store_true', help='Skip plotting, only compute metrics')
    parser.add_argument('--disable-emd', action='store_true', help='Disable EMD computation to reduce memory usage')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override config values with dot-notation (e.g., match_criteria.sampling.mode=bidirectional). Can be used multiple times.')
    parser.add_argument('--use-true-target-latent', action='store_true',
                        help='Feed true target latent (encoder on held-out target) into generator instead of predictor output')
    parser.add_argument('--plot-all-timepoints', action='store_true',
                        help='Overlay forecast vs ground truth for all timepoints (training + last)')
    parser.add_argument('--source-steps-back', type=int, default=1,
                        help='How many steps earlier to use as source for forecasting (1 means last training timepoint)')
    parser.add_argument('--num-sets', type=int, default=1,
                        help='Number of random subsets (of size set_size) to forecast and aggregate')
    parser.add_argument('--set-size', type=int, default=None,
                        help='Override the set_size parameter for forecasting (number of points in each subset).')
    parser.add_argument('--use-ridge-predictor', action='store_true',
                        help='Use the ridge regressor saved by snapMMD_visualize_latents.py as the unconditioned predictor')
    # No explicit CLI flags for predictor matching; use config/overrides via --set predictor_match_criteria.*
    args = parser.parse_args()

    # Load analysis config and apply CLI overrides
    cfg_oc = OmegaConf.load(args.config)
    if args.overrides:
        # Support both "key=value" and "--key=value" forms
        dotlist = [ov.lstrip('-') for ov in args.overrides]
        cli_oc = OmegaConf.from_dotlist(dotlist)
        cfg_oc = OmegaConf.merge(cfg_oc, cli_oc)
    config = OmegaConf.to_container(cfg_oc, resolve=True)  # standard Python containers
    print("!!!!",config)
    experiment_name: str = config['experiment_name']
    match_criteria: Dict[str, Any] = config.get('match_criteria', {})
    predictor_match_criteria: Dict[str, Any] = config.get('predictor_match_criteria', {})
    naming_parameters: List[str] = config.get('naming_parameters', [])
    output_folder: str = config.get('output_folder', 'figures')
    default_seed_for_plots: int = int(config.get('default_seed_for_plots', 0))

    # Build output directory name
    folder_name = build_output_folder_name(experiment_name, match_criteria, naming_parameters)
    out_dir = os.path.join(output_folder, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, f"{folder_name}_analysis_CDE")

    logger.info(f"Experiment name (prefix): {experiment_name}")
    logger.info(f"Predictor source: {args.predictor_source}")
    logger.info(f"Outputs directory: {os.path.abspath(args.outputs_dir)}")
    logger.info(f"Match criteria: {json.dumps(match_criteria, indent=2)}")
    parameters_label_for_title = build_parameters_label(match_criteria, naming_parameters)

    # Find candidate experiment directories
    if not os.path.exists(args.outputs_dir):
        print(f"Error: outputs directory not found: {args.outputs_dir}")
        sys.exit(1)

    prefix = f"{experiment_name}_"
    candidates: List[str] = []
    for item in os.listdir(args.outputs_dir):
        full_path = os.path.join(args.outputs_dir, item)
        if os.path.isdir(full_path) and item.startswith(prefix):
            candidates.append(full_path)

    if not candidates:
        msg = f"No experiment directories found matching pattern {prefix}<hash> in {args.outputs_dir}"
        logger.error(msg)
        print(msg)
        sys.exit(1)

    # Filter by config match
    matched: List[Tuple[str, Dict[str, Any]]] = []
    for exp_dir in candidates:
        cfg_path = os.path.join(exp_dir, 'config.yaml')
        if not os.path.exists(cfg_path):
            continue
        try:
            cfg = load_hydra_resolved_config(cfg_path)
        except Exception:
            # Fallback to plain YAML if OmegaConf fails for any reason
            try:
                cfg = load_yaml(cfg_path)
            except Exception:
                continue
        if dict_contains(cfg, match_criteria):
            matched.append((exp_dir, cfg))

    if not matched:
        msg = "No experiments matched the provided criteria."
        logger.error(msg)
        print(msg)
        sys.exit(1)

    # Check whether differences are only seeds
    seeds: List[int] = []
    cleaned_signatures = set()
    cleaned_cfgs = []
    for _, cfg in matched:
        cleaned = remove_key_recursive(cfg, 'seed')
        cleaned_cfgs.append(cleaned)
        cleaned_signatures.add(json.dumps(cleaned, sort_keys=True, default=str))
        try:
            seeds.append(extract_seed(cfg))
        except Exception:
            seeds.append(None)

    if len(cleaned_signatures) > 1:
        diff_msg = summarize_differences([cfg for _, cfg in matched])
        logger.error(diff_msg)
        print(diff_msg)
        sys.exit(1)

    # Ensure we have seeds
    if any(s is None for s in seeds):
        msg = "One or more matched runs do not specify a 'seed'; cannot aggregate by seed."
        logger.error(msg)
        print(msg)
        sys.exit(1)

    # Sort matched by seed
    matched_with_seed = list(zip(seeds, matched))
    matched_with_seed.sort(key=lambda x: x[0])
    seeds_sorted = [s for s, _ in matched_with_seed]
    logger.info(f"Matched runs (seeds): {seeds_sorted}")

    # Load dataset training data (use dataset_name from first match)
    example_cfg = matched_with_seed[0][1][1]
    dataset_name = extract_dataset_name(example_cfg)
    if dataset_name == 'PBMC':
        dataset_name = 'pbmc'
    training_data = load_training_data_for_dataset(dataset_name)

    # PCA for PBMC plotting
    pca = None
    if dataset_name == 'pbmc' and DATASET_CONFIGS['pbmc'].get('requires_pca', False) and not args.skip_plots:
        pca = setup_pca_for_pbmc(logger)

    # Compute metrics per seed
    per_seed_results: List[Tuple[int, float, Any]] = []  
    forecast_for_plot = None

    # Prepare to load per-seed ridge predictors from seed-specific subdirectories
    def load_ridge_predictor_for_seed(seed_value: int) -> Optional[UnconditionedRidgePredictor]:
        ridge_dir = os.path.join(out_dir, f"seed_{seed_value}")
        ridge_path = os.path.join(ridge_dir, 'ridge_regressor.pkl')
        if not os.path.exists(ridge_path):
            logger.warning(f"--use-ridge-predictor set but ridge file not found for seed {seed_value} at {ridge_path}; skipping this seed")
            return None
        try:
            ridge_obj_local = joblib.load(ridge_path)
            logger.info(f"Loaded ridge predictor for seed {seed_value} from {ridge_path}")
            return UnconditionedRidgePredictor(ridge_obj_local)
        except Exception as e:
            logger.error(f"Failed to load ridge predictor for seed {seed_value} at {ridge_path}: {e}")
            return None
    for seed, (exp_seed, (exp_dir, cfg)) in zip(seeds_sorted, matched_with_seed):
        # Identify predictor hashed subdir only if using 'separate' source
        predictor_dir = None
        if (args.predictor_source == 'separate') and (not args.use_true_target_latent):
            try:
                predictor_dir = find_matching_predictor_subdir(exp_dir, predictor_match_criteria, seed)
            except Exception as e:
                logger.error(f"Predictor subdir selection error for {exp_dir}: {e}")
                print(f"Predictor subdir selection error for {exp_dir}: {e}")
                sys.exit(1)

        # Load ridge predictor for this seed if requested
        ridge_predictor_obj = None
        if args.use_ridge_predictor:
            ridge_predictor_obj = load_ridge_predictor_for_seed(seed)
            if ridge_predictor_obj is None:
                logger.info(f"Skipping seed {seed}: ridge predictor unavailable.")
                continue

        forecast = generate_cde_forecast(
            exp_dir,
            training_data,
            predictor_source=args.predictor_source,
            use_true_target_latent=args.use_true_target_latent,
            forecast_all_timepoints=args.plot_all_timepoints,
            predictor_dir=predictor_dir,
            source_steps_back=args.source_steps_back,
            num_sets=args.num_sets,
            cli_set_size=args.set_size,
            override_predictor=ridge_predictor_obj,
        )
        if forecast is None:
            logger.info(f"Skipping seed {seed}: no predictor available and --use-true-target-latent is False.")
            continue
        mmd_legacy, mmd_paper, emd = compute_mmd_and_emd(dataset_name, forecast['forecast'], logger, enable_emd=not args.disable_emd)
        per_seed_results.append((seed, mmd_legacy, mmd_paper, emd))
        if forecast_for_plot is None and seed == default_seed_for_plots:
            forecast_for_plot = forecast

    if not per_seed_results:
        msg = "Failed to compute metrics for any matched run."
        logger.error(msg)
        print(msg)
        sys.exit(1)

    # Print and log per-seed metrics
    print("Individual results (by seed):")
    logger.info("Individual results (by seed):")
    mmd_paper_list = []
    mmd_legacy_list = []
    emd_list = []
    for seed, mmd_legacy, mmd_paper, emd in per_seed_results:
        emd_str = ("%.6f" % emd) if emd is not None else "n/a"
        line = f"  Seed {seed}: MMD_legacy = {mmd_legacy:.6f}, MMD_paper = {mmd_paper:.6f}, EMD = {emd_str}"
        print(line)
        logger.info(line)
        mmd_paper_list.append(mmd_paper)
        mmd_legacy_list.append(mmd_legacy)
        if emd is not None:
            emd_list.append(emd)

    # Aggregate stats
    mmd_paper_arr = np.array(mmd_paper_list)
    mmd_legacy_arr = np.array(mmd_legacy_list)
    mmd_paper_mean = float(mmd_paper_arr.mean())
    mmd_legacy_mean = float(mmd_legacy_arr.mean())
    mmd_paper_std = float(mmd_paper_arr.std())
    mmd_legacy_std = float(mmd_legacy_arr.std())
    print(f"MMD_paper: {mmd_paper_mean:.6f} ± {mmd_paper_std:.6f}")
    print(f"MMD_legacy: {mmd_legacy_mean:.6f} ± {mmd_legacy_std:.6f}")
    #print(f"MMD^2: {mmd2_arr.mean():.6f} ± {mmd2_arr.std():.6f}")
    logger.info(f"MMD_paper: {mmd_paper_mean:.6f} ± {mmd_paper_std:.6f}")
    logger.info(f"MMD_legacy: {mmd_legacy_mean:.6f} ± {mmd_legacy_std:.6f}")
    #logger.info(f"MMD^2: {mmd2_arr.mean():.6f} ± {mmd2_arr.std():.6f}")

    emd_title_segment = ""
    if len(emd_list) > 0:
        emd_arr = np.array(emd_list)
        emd_mean = float(emd_arr.mean())
        emd_std = float(emd_arr.std())
        print(f"EMD: {emd_mean:.6f} ± {emd_std:.6f}")
        logger.info(f"EMD: {emd_mean:.6f} ± {emd_std:.6f}")
        emd_title_segment = f" | EMD={emd_mean:.4g}±{emd_std:.2g}"

    # Plot for default seed
    if not args.skip_plots:
        if forecast_for_plot is None:
            # If not found seed==default, try the first matched run
            default_exp_dir = matched_with_seed[0][1][0]
            default_predictor_dir = None
            if (args.predictor_source == 'separate') and (not args.use_true_target_latent):
                try:
                    default_predictor_dir = find_matching_predictor_subdir(default_exp_dir, predictor_match_criteria, seeds_sorted[0])
                except Exception as e:
                    logger.error(f"Predictor subdir selection error for default run {default_exp_dir}: {e}")
                    print(f"Predictor subdir selection error for default run {default_exp_dir}: {e}")
                    sys.exit(1)
            default_ridge_predictor_obj = None
            if args.use_ridge_predictor:
                default_ridge_predictor_obj = load_ridge_predictor_for_seed(seeds_sorted[0])
                if default_ridge_predictor_obj is None:
                    logger.warning(f"No ridge predictor available for default seed {seeds_sorted[0]}; skipping plots.")
                    forecast_for_plot = None
                
            forecast_for_plot = generate_cde_forecast(
                default_exp_dir,
                training_data,
                predictor_source=args.predictor_source,
                use_true_target_latent=args.use_true_target_latent,
                forecast_all_timepoints=args.plot_all_timepoints,
                predictor_dir=default_predictor_dir,
                source_steps_back=args.source_steps_back,
                num_sets=args.num_sets,
                cli_set_size=args.set_size,
                override_predictor=default_ridge_predictor_obj,
            )

        if forecast_for_plot is not None:
            # Prepare title suffix with naming parameters and metrics
            metrics_title_segment = f"MMD={mmd_paper_mean:.4g}±{mmd_paper_std:.2g}, MMD_legacy={mmd_legacy_mean:.4g}±{mmd_legacy_std:.2g}{emd_title_segment}"
            title_suffix = f"{parameters_label_for_title} | {metrics_title_segment}"

            results_struct = {
                'training_data': training_data,
                'forecast_data': {
                    'forecast': forecast_for_plot['forecast'],
                    'X_val_forecast': forecast_for_plot['X_val'],
                    'forecast_sequence': forecast_for_plot.get('forecast_sequence')
                },
                'metadata': {
                    'task_name': dataset_name,
                    'config': DATASET_CONFIGS[dataset_name],
                    'forecast_method': 'CDE'
                }
            }
            plot_main_results(dataset_name, results_struct, out_dir, logger, pca, title_suffix=title_suffix, plot_all_timepoints=args.plot_all_timepoints)
        else:
            logger.warning("No forecast available for plotting; skipping plots.")

    print(f"\n✓ Analysis complete. Figures and logs saved to: {out_dir}/")
    logger.info(f"Analysis complete. Output saved to: {out_dir}/")


if __name__ == "__main__":
    main()