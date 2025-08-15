import os
import sys
import argparse
import logging
import yaml
import json
import itertools
import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils.snapMMD import MMDLoss, RBF
from utils.experiment_utils import load_best_model, get_experiment_info
import hydra
from omegaconf import OmegaConf


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
        'data_path': 'data/classic/LV_simulated_dataset_1000_31_seed0.npz',
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

def load_models_from_experiment(experiment_dir: str, device: torch.device):
    info = get_experiment_info(experiment_dir, load_checkpoints=False)
    cfg = info['config']

    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])

    # Attach predictor if present
    predictor = None
    if 'predictor' in cfg:
        predictor = hydra.utils.instantiate(cfg['predictor'])
        # Match latent activation if encoder defines it
        if hasattr(enc, 'latent_act'):
            try:
                predictor.latent_act = enc.latent_act
            except Exception:
                pass
        enc.predictor = predictor

    state = load_best_model(info['dir'])

    enc.load_state_dict(state['encoder_state_dict'])
    # Newer configs may store generator_state_dict under 'generator_state_dict'
    if 'generator_state_dict' in state:
        gen.load_state_dict(state['generator_state_dict'])
    else:
        # Backward compatibility
        gen.model.load_state_dict(state['generator_state_dict'])

    enc.eval(); gen.eval()
    enc.to(device); gen.to(device)
    return cfg, enc, gen


def generate_cde_forecast(experiment_dir: str, training_data: Dict[str, Any]):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg, enc, gen = load_models_from_experiment(experiment_dir, device)

    Xs_training = training_data['Xs']
    samples_s = torch.tensor(Xs_training[-1]).unsqueeze(0).to(device).float()
    samples_t = torch.tensor(training_data['X_val_true']).unsqueeze(0).to(device).float()

    with torch.no_grad():
        enc_s = enc(samples_s)
        # Determine enc_t
        enc_t = None
        if hasattr(enc, 'predictor') and enc.predictor is not None:
            predictor = enc.predictor
            if hasattr(predictor, 'requires_dt') and predictor.requires_dt:
                # Use actual dt as the difference between the last two entries of 'dts'
                dts = training_data.get('dts', None)
                dt_value = float(dts[-1] - dts[-2])

                dt = torch.full((enc_s.shape[0],), dt_value, device=device, dtype=enc_s.dtype)
                enc_t = predictor(enc_s, dt)
            else:
                enc_t = predictor(enc_s)
        else:
            # Fallback: encode target directly
            enc_t = enc(samples_t)

    # Reshape source samples for generator
    batch_size, set_size, *data_shape = samples_s.shape
    samples_s = samples_s.reshape(-1, *data_shape)

    forecast = gen.sample(samples_s, enc_s, enc_t)
    forecast_np = forecast.detach().cpu().numpy()
    forecast_structured = forecast_np[None, :, :]  # (1, N, D)
    return {
        'forecast': forecast_structured,
        'X_val': training_data['X_val_true']
    }


# -----------------------------
# Metrics
# -----------------------------

def calculate_emd(x: np.ndarray, y: np.ndarray) -> float:
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


def compute_mmd_and_emd(dataset_name: str, forecast_1xNxD: np.ndarray, logger: logging.Logger, enable_emd: bool = True) -> Tuple[float, float]:
    cfg = DATASET_CONFIGS[dataset_name]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load training data (ground-truth final timepoint)
    if dataset_name == 'pbmc':
        training_data = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    else:
        training_data = np.load(cfg['data_path'])

    X_val = training_data['Xs'][-1]

    rbf = RBF(bandwidth=2.0).to(device)
    myMMD = MMDLoss(kernel=rbf).to(device)

    # forecast_1xNxD is numpy with shape (1, N, D). Take final timepoint (N, D)
    # TODO: to me this indicates that there is an issue.
    forecast_final_np = forecast_1xNxD[-1][-1]
    # Torch tensors for MMD
    forecast_final_t = torch.from_numpy(forecast_final_np).to(device)
    X_val_t = torch.tensor(X_val).to(device)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",forecast_1xNxD.shape,forecast_1xNxD[-1].shape, X_val_t.shape)
    mmd_squared = myMMD(forecast_final_t, X_val_t).item()
    mmd = float(np.sqrt(mmd_squared))

    emd = None
    if enable_emd and cfg.get('calculate_emd', False):

        forecast_for_emd = forecast_final_np
        X_val_for_emd = X_val

        # EMD expects numpy arrays with shape (N, D) and (M, D)
        emd_val = calculate_emd(forecast_for_emd, X_val_for_emd)
        emd = float(emd_val) if not np.isnan(emd_val) else None

    logger.info(f"Computed metrics -> MMD: {mmd:.6f}, MMD^2: {mmd_squared:.6f}, EMD: {('%.6f' % emd) if emd is not None else 'n/a'}")
    return mmd_squared, emd


# -----------------------------
# PCA helpers for PBMC plotting
# -----------------------------

def setup_pca_for_pbmc(logger: logging.Logger) -> PCA:
    logger.info("Computing PCA for PBMC datasets...")
    data1 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20.npz")
    data2 = np.load("data/realdata/processed_pbmc_data_sub500_every_2_until20_interp_val.npz")
    Xs1 = data1["Xs"]; Xs2 = data2["Xs"]
    if Xs1.shape[0] == 21 and Xs2.shape[0] == 20:
        Xs1, Xs2 = Xs2, Xs1
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

def plot_main_results(dataset_name: str, results: Dict[str, Any], out_dir: str, logger: logging.Logger, pca: PCA = None):
    os.makedirs(out_dir, exist_ok=True)
    cfg = DATASET_CONFIGS[dataset_name]
    training = results['training_data']
    forecast = results['forecast_data']
    
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!",forecast['forecast'].shape, forecast['X_val_forecast'].shape)
    
    
    is_3d = cfg.get('plot_dimensionality', cfg['dimensionality']) == 3
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    fig.suptitle(f"{cfg['title']} Results (CDE)")

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

    if is_3d:
        scatter = ax.scatter(all_training[:, 0], all_training[:, 1], all_training[:, 2], alpha=0.7, s=3.0, c=colors, cmap='coolwarm', vmin=1, vmax=n_sequences)
    else:
        scatter = ax.scatter(all_training[:, 0], all_training[:, 1], alpha=0.7, s=3.0, c=colors, cmap='coolwarm', vmin=1, vmax=n_sequences)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Time Point', rotation=270, labelpad=15)

    true_data = transform_for_plot(dataset_name, training['X_val_true'], pca)
    forecast_data = transform_for_plot(dataset_name, forecast['forecast'][-1][-1], pca)

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


def main():
    parser = argparse.ArgumentParser(description='Flexible analysis of CDE results across seeds')
    parser.add_argument('--config', type=str, default='analysis_config.yaml', help='Path to analysis config file')
    parser.add_argument('--outputs-dir', type=str, default='outputs', help='Directory containing experiment subdirectories')
    parser.add_argument('--skip-plots', action='store_true', help='Skip plotting, only compute metrics')
    parser.add_argument('--disable-emd', action='store_true', help='Disable EMD computation to reduce memory usage')
    parser.add_argument('--set', dest='overrides', action='append', default=[], help='Override config values with dot-notation (e.g., match_criteria.sampling.mode=bidirectional). Can be used multiple times.')
    args = parser.parse_args()

    # Load analysis config and apply CLI overrides
    cfg_oc = OmegaConf.load(args.config)
    if args.overrides:
        # Support both "key=value" and "--key=value" forms
        dotlist = [ov.lstrip('-') for ov in args.overrides]
        cli_oc = OmegaConf.from_dotlist(dotlist)
        cfg_oc = OmegaConf.merge(cfg_oc, cli_oc)
    config = OmegaConf.to_container(cfg_oc, resolve=True)  # standard Python containers
    experiment_name: str = config['experiment_name']
    match_criteria: Dict[str, Any] = config.get('match_criteria', {})
    naming_parameters: List[str] = config.get('naming_parameters', [])
    output_folder: str = config.get('output_folder', 'figures')
    default_seed_for_plots: int = int(config.get('default_seed_for_plots', 0))

    # Build output directory name
    folder_name = build_output_folder_name(experiment_name, match_criteria, naming_parameters)
    out_dir = os.path.join(output_folder, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir, f"{folder_name}_analysis_CDE")

    logger.info(f"Experiment name (prefix): {experiment_name}")
    logger.info(f"Outputs directory: {os.path.abspath(args.outputs_dir)}")
    logger.info(f"Match criteria: {json.dumps(match_criteria, indent=2)}")

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
    per_seed_results: List[Tuple[int, float, Any]] = []  # (seed, mmd^2, emd)
    forecast_for_plot = None
    for seed, (exp_seed, (exp_dir, cfg)) in zip(seeds_sorted, matched_with_seed):
        forecast = generate_cde_forecast(exp_dir, training_data)
        mmd2, emd = compute_mmd_and_emd(dataset_name, forecast['forecast'], logger, enable_emd=not args.disable_emd)
        per_seed_results.append((seed, mmd2, emd))
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
    mmd2_list = []
    emd_list = []
    for seed, mmd2, emd in per_seed_results:
        mmd = float(np.sqrt(mmd2))
        emd_str = ("%.6f" % emd) if emd is not None else "n/a"
        line = f"  Seed {seed}: MMD = {mmd:.6f}, MMD^2 = {mmd2:.6f}, EMD = {emd_str}"
        print(line)
        logger.info(line)
        mmd2_list.append(mmd2)
        if emd is not None:
            emd_list.append(emd)

    # Aggregate stats
    mmd2_arr = np.array(mmd2_list)
    mmd_arr = np.sqrt(mmd2_arr)
    print(f"MMD: {mmd_arr.mean():.6f} ± {mmd_arr.std():.6f}")
    print(f"MMD^2: {mmd2_arr.mean():.6f} ± {mmd2_arr.std():.6f}")
    logger.info(f"MMD: {mmd_arr.mean():.6f} ± {mmd_arr.std():.6f}")
    logger.info(f"MMD^2: {mmd2_arr.mean():.6f} ± {mmd2_arr.std():.6f}")

    if len(emd_list) > 0:
        emd_arr = np.array(emd_list)
        print(f"EMD: {emd_arr.mean():.6f} ± {emd_arr.std():.6f}")
        logger.info(f"EMD: {emd_arr.mean():.6f} ± {emd_arr.std():.6f}")

    # Plot for default seed
    if not args.skip_plots:
        if forecast_for_plot is None:
            # If not found seed==default, just take first
            forecast_for_plot = generate_cde_forecast(matched_with_seed[0][1][0], training_data)

        results_struct = {
            'training_data': training_data,
            'forecast_data': {
                'forecast': forecast_for_plot['forecast'],
                'X_val_forecast': forecast_for_plot['X_val']
            },
            'metadata': {
                'task_name': dataset_name,
                'config': DATASET_CONFIGS[dataset_name],
                'forecast_method': 'CDE'
            }
        }
        plot_main_results(dataset_name, results_struct, out_dir, logger, pca)

    print(f"\n✓ Analysis complete. Figures and logs saved to: {out_dir}/")
    logger.info(f"Analysis complete. Output saved to: {out_dir}/")


if __name__ == "__main__":
    main()


