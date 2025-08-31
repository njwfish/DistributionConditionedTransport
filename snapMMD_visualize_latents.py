import os
import sys
import argparse
import logging
import yaml
import json
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib

import hydra
from omegaconf import OmegaConf

from utils.experiment_utils import load_best_model, get_experiment_info


# -----------------------------
# Configuration helpers (mirrors backbone)
# -----------------------------

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_hydra_resolved_config(path: str) -> Dict[str, Any]:
    """Load a Hydra/OmegaConf YAML config and resolve interpolations like ${a.b}."""
    cfg = OmegaConf.load(path)
    resolved: Dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    return resolved


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
        return isinstance(superset, list) and superset == subset
    else:
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
    if isinstance(cfg, dict) and 'seed' in cfg and cfg['seed'] is not None:
        return int(cfg['seed'])
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
    if 'dataset_name' in cfg:
        return str(cfg['dataset_name'])
    raise ValueError("Could not extract 'dataset.dataset_name' from config")


# -----------------------------
# Data config (paths + plotting info)
# -----------------------------

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'LV': {
        'data_path': 'data/classic/LV_data.npz',
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
# Model loading (reusing backbone logic)
# -----------------------------

def load_models_from_experiment(experiment_dir: str, device: torch.device, predictor_source: str = 'separate', predictor_dir: Optional[str] = None, use_true_target_latent: bool = False):
    info = get_experiment_info(experiment_dir, load_checkpoints=False)
    cfg = info['config']

    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])
    dataset = hydra.utils.instantiate(cfg['dataset'])

    cfg_resolved = OmegaConf.to_container(cfg, resolve=True)
    experiment_cfg = cfg_resolved.get('experiment', {}) if isinstance(cfg_resolved, dict) else {}
    train_predictor_posthoc = bool(experiment_cfg.get('train_predictor_posthoc', False))

    predictor = None

    if not use_true_target_latent:
        if predictor_dir is not None:
            pred_cfg_path = os.path.join(predictor_dir, 'config.yaml')
            if not os.path.exists(pred_cfg_path):
                predictor = None
            else:
                try:
                    pred_cfg_oc = OmegaConf.load(pred_cfg_path)
                except Exception:
                    pred_cfg_oc = None
                if pred_cfg_oc is not None:
                    pred_cfg_node = pred_cfg_oc.get('predictor')
                    if pred_cfg_node is None:
                        predictor = None
                    else:
                        try:
                            predictor = hydra.utils.instantiate(pred_cfg_node)
                        except Exception:
                            predictor = None
        else:
            predictor = None

        if predictor is not None and hasattr(enc, 'latent_act'):
            predictor.latent_act = enc.latent_act
    else:
        predictor = None
        train_predictor_posthoc = False

    state = load_best_model(info['dir'])
    enc.load_state_dict(state['encoder_state_dict'])
    if 'generator_state_dict' in state:
        gen.load_state_dict(state['generator_state_dict'])
    else:
        gen.model.load_state_dict(state['generator_state_dict'])

    enc.eval(); gen.eval()
    enc.to(device); gen.to(device)
    if predictor is not None:
        predictor.eval(); predictor.to(device)

    return cfg, enc, gen, predictor, dataset, train_predictor_posthoc


# -----------------------------
# Dataset loading
# -----------------------------

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
# Naming helpers
# -----------------------------

def build_output_folder_name(experiment_name: str, match_criteria: Dict[str, Any], naming_parameters: List[str] = None) -> str:
    if naming_parameters is None or len(naming_parameters) == 0:
        flat = flatten_dict(match_criteria)
        parts = [experiment_name] + [sanitize_value_for_name(v) for (_, v) in flat]
    else:
        parts = [experiment_name]
        for path in naming_parameters:
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
    cleaned = [remove_key_recursive(c, 'seed') for c in configs]
    values_by_key: Dict[str, List[Any]] = {}
    for c in cleaned:
        for k, v in flatten_dict(c):
            values_by_key.setdefault(k, []).append(v)
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
    if not diffs:
        return ""
    lines = ["Ambiguous matches: configurations differ on the following keys (excluding 'seed'):"]
    for k, vals in sorted(diffs.items()):
        preview = ", ".join([sanitize_value_for_name(v) for v in vals])
        lines.append(f"  - {k}: {preview}")
    return "\n".join(lines)


# -----------------------------
# Latent extraction helpers
# -----------------------------

def _extract_latent_vector(enc_out: Any) -> np.ndarray:
    """Extract a 1D latent vector from encoder output in a robust way.
    - If tensor with leading batch dim 1, flatten from dim 1 onward -> (F,)
    - If tuple/list: take first element and recurse
    - If dict: take first value and recurse
    """
    if isinstance(enc_out, torch.Tensor):
        if enc_out.dim() == 0:
            return enc_out.detach().cpu().numpy().reshape(1)
        if enc_out.shape[0] == 1:
            flat = enc_out.reshape(1, -1)[0]
            return flat.detach().cpu().numpy()
        else:
            flat = enc_out.reshape(enc_out.shape[0], -1)
            return flat.detach().cpu().numpy()[0]
    if isinstance(enc_out, (list, tuple)) and len(enc_out) > 0:
        return _extract_latent_vector(enc_out[0])
    if isinstance(enc_out, dict) and len(enc_out) > 0:
        return _extract_latent_vector(next(iter(enc_out.values())))
    raise ValueError("Unsupported encoder output type for latent extraction: %r" % type(enc_out))


def _sample_indices(num_items: int, set_size: int, rng: np.random.RandomState) -> np.ndarray:
    replace = set_size > num_items
    return rng.choice(num_items, size=set_size, replace=replace)


# -----------------------------
# Core visualization logic
# -----------------------------

def visualize_latents_for_run(exp_dir: str, cfg: Dict[str, Any], out_dir: str, logger: logging.Logger, num_sets: int, set_size_cli: Optional[int] = None, title_suffix: str = "", skip_plots: bool = False, ridge_alpha: float = 1.0) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg_model, enc, _gen, _pred, dataset, _ = load_models_from_experiment(exp_dir, device, use_true_target_latent=True)

    cfg_resolved = OmegaConf.to_container(cfg_model, resolve=True)
    exp_cfg = cfg_resolved.get('experiment', {}) if isinstance(cfg_resolved, dict) else {}
    set_size = None
    if isinstance(exp_cfg, dict) and exp_cfg.get('set_size') is not None:
        try:
            set_size = int(exp_cfg['set_size'])
        except Exception:
            set_size = None
    if set_size is None:
        set_size = int(getattr(dataset, 'set_size', 32))
    if set_size_cli is not None:
        set_size = int(set_size_cli)

    dataset_name = extract_dataset_name(cfg)
    if dataset_name == 'PBMC':
        dataset_name = 'pbmc'
    training = load_training_data_for_dataset(dataset_name)

    # Build latent collection
    latents: List[np.ndarray] = []
    time_labels: List[int] = []
    
    # For ridge regression: collect latents organized by timepoint
    latents_by_timepoint: List[List[np.ndarray]] = []

    n_training_steps = len(training['Xs'])
    n_total_steps = n_training_steps + 1  # include held-out final

    # RNG seeded by the experiment seed for reproducibility
    try:
        seed_val = extract_seed(cfg)
    except Exception:
        seed_val = None
    rng = np.random.RandomState(seed_val if seed_val is not None else None)

    with torch.no_grad():
        # Process training timepoints
        for t in range(n_training_steps):
            X_np = training['Xs'][t]
            timepoint_latents = []
            for _ in range(num_sets):
                idx = _sample_indices(X_np.shape[0], set_size, rng)
                subset_np = X_np[idx]
                subset_t = torch.tensor(subset_np, dtype=torch.float32, device=device).unsqueeze(0)
                enc_out = enc(subset_t)
                lat = _extract_latent_vector(enc_out)
                latents.append(lat)
                timepoint_latents.append(lat)
                time_labels.append(t + 1)  # 1-indexed
            latents_by_timepoint.append(timepoint_latents)

        # Process held-out final timepoint
        X_final_np = training['X_val_true']
        timepoint_latents = []
        for _ in range(num_sets):
            idx = _sample_indices(X_final_np.shape[0], set_size, rng)
            subset_np = X_final_np[idx]
            subset_t = torch.tensor(subset_np, dtype=torch.float32, device=device).unsqueeze(0)
            enc_out = enc(subset_t)
            lat = _extract_latent_vector(enc_out)
            latents.append(lat)
            timepoint_latents.append(lat)
            time_labels.append(n_total_steps)  # last index
        latents_by_timepoint.append(timepoint_latents)

    if len(latents) == 0:
        logger.error("No latents were collected; aborting visualization.")
        return

    latents_arr = np.vstack(latents)
    time_labels_arr = np.array(time_labels)

    # Train ridge regression for latent forecasting
    ridge_r2 = None
    ridge_mse = None
    ridge_model = None
    predicted_latents = []
    predicted_time_labels = []
    
    print("LENGTH OF LATENTS BY TIMEPOINT: ", len(latents_by_timepoint))
    if len(latents_by_timepoint) >= 2:  # Need at least 2 timepoints for transitions
        # Prepare training datat: X = latents at time t, y = latents at time t+1
        X_ridge = []
        y_ridge = []
        
        for t in range(len(latents_by_timepoint) - 2):  # Exclude last timepoint as target
            current_latents = latents_by_timepoint[t]
            next_latents = latents_by_timepoint[t + 1]
            
            # Pair each latent at time t with each latent at time t+1
            for curr_lat in current_latents:
                for next_lat in next_latents:
                    X_ridge.append(curr_lat)
                    y_ridge.append(next_lat)
        
        if len(X_ridge) > 0:
            X_ridge = np.vstack(X_ridge)
            y_ridge = np.vstack(y_ridge)
            
            # Train ridge regression
            ridge_model = Ridge(alpha=ridge_alpha, random_state=seed_val)
            ridge_model.fit(X_ridge, y_ridge)
            
            # Evaluate on training data (since we don't have separate test data)
            y_pred = ridge_model.predict(X_ridge)
            ridge_r2 = r2_score(y_ridge, y_pred)
            ridge_mse = mean_squared_error(y_ridge, y_pred)
            
            logger.info(f"Ridge regression latent forecasting: R² = {ridge_r2:.6f}, MSE = {ridge_mse:.6f}")

            # Save trained ridge regressor and metadata alongside figures
            try:
                ridge_path = os.path.join(out_dir, 'ridge_regressor.pkl')
                joblib.dump(ridge_model, ridge_path)
                ridge_meta = {
                    'alpha': float(ridge_alpha),
                    'r2': float(ridge_r2) if ridge_r2 is not None else None,
                    'mse': float(ridge_mse) if ridge_mse is not None else None,
                    'seed': int(seed_val) if seed_val is not None else None,
                    'latent_dim': int(latents_arr.shape[1]) if latents_arr.ndim == 2 else None,
                    'num_training_pairs': int(X_ridge.shape[0]),
                    'num_timepoints': int(len(latents_by_timepoint)),
                }
                with open(os.path.join(out_dir, 'ridge_regressor_meta.json'), 'w') as f:
                    json.dump(ridge_meta, f, indent=2)
                logger.info(f"Saved ridge regressor to: {ridge_path}")
            except Exception as e:
                logger.warning(f"Failed to save ridge regressor: {e}")
            
            # Generate predictions for visualization: predict t+1 from each t
            for t in range(len(latents_by_timepoint) - 1):  # Exclude final timepoint
                current_latents = latents_by_timepoint[t]
                predictions = ridge_model.predict(np.vstack(current_latents))
                
                for pred_lat in predictions:
                    predicted_latents.append(pred_lat)
                    predicted_time_labels.append(t + 2)  # Predicted timepoint is t+1 (1-indexed: t+2)
                    
        else:
            logger.warning("No latent pairs available for ridge regression training")
    else:
        logger.warning("Insufficient timepoints for ridge regression (need at least 2)")

    # Fit PCA only on ground-truth latents (no ridge predictions to avoid leakage)
    n_components = min(5, latents_arr.shape[1]) if latents_arr.ndim == 2 else 2
    pca = PCA(n_components=n_components)
    pca.fit(latents_arr)
    evr = pca.explained_variance_ratio_
    evr_str = ", ".join([f"{v:.6f}" for v in evr])
    logger.info(f"Explained variance ratio (first {n_components} PCs): {evr_str}")

    # Plot 2D scatter on first two PCs
    if not skip_plots:
        # Transform ground truth latents
        if n_components >= 2:
            latents_2d = pca.transform(latents_arr)[:, :2]
        else:
            comp1 = pca.transform(latents_arr)[:, 0]
            latents_2d = np.stack([comp1, np.zeros_like(comp1)], axis=1)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        n_sequences = n_total_steps

        # Plot ground truth latents
        sc_gt = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], c=time_labels_arr, cmap='coolwarm', 
                          vmin=1, vmax=n_sequences, s=15.0, alpha=0.8, edgecolors='white', 
                          linewidth=0.3, label='Ground Truth', marker='o')

        # Plot predicted latents if available
        if len(predicted_latents) > 0:
            predicted_latents_arr = np.vstack(predicted_latents)
            predicted_time_labels_arr = np.array(predicted_time_labels)
            
            if n_components >= 2:
                predicted_2d = pca.transform(predicted_latents_arr)[:, :2]
            else:
                pred_comp1 = pca.transform(predicted_latents_arr)[:, 0]
                predicted_2d = np.stack([pred_comp1, np.zeros_like(pred_comp1)], axis=1)

            sc_pred = ax.scatter(predicted_2d[:, 0], predicted_2d[:, 1], c=predicted_time_labels_arr, 
                               cmap='coolwarm', vmin=1, vmax=n_sequences, s=15.0, alpha=0.6, 
                               edgecolors='black', linewidth=0.5, label='Ridge Predictions', marker='s')

        # Colorbar and labels
        cbar = plt.colorbar(sc_gt, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Time Point', rotation=270, labelpad=15)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')

        ds_title = DATASET_CONFIGS.get(dataset_name, {}).get('title', dataset_name)
        title_text = f"{ds_title} Encoder Latent PCA"
        if len(predicted_latents) > 0:
            title_text += " (GT + Ridge Predictions)"
        if title_suffix:
            title_text += f" | {title_suffix}"
        ax.set_title(title_text)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{dataset_name}_latent_PCA.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Latent PCA figure saved to: {out_path}")


# -----------------------------
# Main (experiment discovery + per-seed visualization)
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Visualize encoder latent space across timepoints')
    parser.add_argument('--config', type=str, default='analysis_config.yaml', help='Path to analysis config file')
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Directory containing experiment subdirectories')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plotting (still logs explained variance)')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override config values with dot-notation (e.g., match_criteria.sampling.mode=bidirectional). Can be used multiple times.')
    parser.add_argument('--num-sets', type=int, default=1, help='Number of random subsets per timepoint to encode')
    parser.add_argument('--set-size', type=int, default=None, help='Override the set_size parameter (points per subset)')
    parser.add_argument('--ridge-alpha', type=float, default=1.0, help='Ridge regression regularization parameter (default: 1.0)')
    args = parser.parse_args()

    # Load analysis config and apply CLI overrides
    cfg_oc = OmegaConf.load(args.config)
    if args.overrides:
        dotlist = [ov.lstrip('-') for ov in args.overrides]
        cli_oc = OmegaConf.from_dotlist(dotlist)
        cfg_oc = OmegaConf.merge(cfg_oc, cli_oc)
    config = OmegaConf.to_container(cfg_oc, resolve=True)
    experiment_name: str = config['experiment_name']
    match_criteria: Dict[str, Any] = config.get('match_criteria', {})
    naming_parameters: List[str] = config.get('naming_parameters', [])
    output_folder: str = config.get('output_folder', 'figures')
    default_seed_for_plots: int = int(config.get('default_seed_for_plots', 0))

    # Build output directory name
    folder_name = build_output_folder_name(experiment_name, match_criteria, naming_parameters)
    out_dir = os.path.join(output_folder, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, f"{folder_name}_latent_vis")

    logger.info(f"Experiment name (prefix): {experiment_name}")
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

    # Run visualization + ridge training for ALL matched seeds
    for seed_val, (exp_dir_selected, cfg_selected) in matched_with_seed:
        title_suffix = f"{parameters_label_for_title} | seed={seed_val}"
        out_dir_seed = os.path.join(out_dir, f"seed_{seed_val}")
        os.makedirs(out_dir_seed, exist_ok=True)
        seed_logger = setup_logger(out_dir_seed, f"{folder_name}_latent_vis_seed_{seed_val}")

        seed_logger.info(f"Processing seed={seed_val} at exp_dir={exp_dir_selected}")
        visualize_latents_for_run(
            exp_dir_selected,
            cfg_selected,
            out_dir_seed,
            seed_logger,
            num_sets=int(args.num_sets),
            set_size_cli=args.set_size,
            title_suffix=title_suffix,
            skip_plots=bool(args.skip_plots),
            ridge_alpha=float(args.ridge_alpha),
        )

    print(f"\n✓ Latent visualization complete for seeds {seeds_sorted}. Outputs saved under: {out_dir}/")
    logger.info(f"Latent visualization complete for seeds {seeds_sorted}. Output saved under: {out_dir}/")


if __name__ == "__main__":
    main()


