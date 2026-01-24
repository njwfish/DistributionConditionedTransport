import os

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import joblib

from utils.snapMMD import MMDLoss, RBF
from utils.experiment_utils import load_best_model, get_experiment_info
from utils.seed import seed_everything  # Import seeding utility
from utils.predictor_training2 import get_ridge
from utils.latents import normalize_latent


import hydra
from omegaconf import OmegaConf

from TrajectoryNet.optimal_transport.emd import earth_mover_distance



# Set random seed for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED, deterministic=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    'PBMC': {
        'data_path': 'data/realdata/processed_pbmc_data_sub500_every_2_until20.npz',
        'dimensionality': 30,
        'plot_dimensionality': 3,
        'axes_labels': ['PC1', 'PC2', 'PC3'],
        'title': 'PBMC',
        'calculate_emd': False,
        'requires_pca': True,
    },
}
def load_cfg_and_ckpt(ckpt_dir,outputs_dir="outputs"):
    # Resolve and validate checkpoint directory
    experiment_dir = os.path.join(os.path.abspath(os.path.expanduser(outputs_dir)), ckpt_dir)
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    #ckpt_path = os.path.join(experiment_dir, 'checkpoint_epoch_1000.pt')
    # Load trained config and use it as the active config
    cfg = OmegaConf.load(cfg_path)
    return cfg, ckpt_path



def load_models(cfg,ckpt_path):
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    generator = hydra.utils.instantiate(cfg.generator).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.eval()
    generator.eval()

    return encoder, generator


def find_matching_ckpt_dirs(ckpt_dir,outputs_dir="outputs"):
    """
    Find all checkpoint directories where the config differs only by the seed.
    
    Args:
        cfg: The reference config to compare against
    
    Returns:
        List of checkpoint directory names that have configs differing only by seed
    """
    outputs_dir = os.path.abspath(os.path.expanduser(outputs_dir))
    matching_dirs = []
    
    cfg_path = os.path.join(outputs_dir, ckpt_dir, 'config.yaml')
    cfg = OmegaConf.load(cfg_path)
    
    # Parse experiment_name from reference_ckpt_dir
    parts = ckpt_dir.rsplit('_', 1)
    experiment_name = parts[0]

    # Create a copy of the reference config without the seed for comparison
    ref_cfg_no_seed = OmegaConf.create(cfg)
    if 'seed' in ref_cfg_no_seed:
        del ref_cfg_no_seed['seed']
    
   
    for dir_name in os.listdir(outputs_dir):
        dir_path = os.path.join(outputs_dir, dir_name)
        
        # Check if it's a directory and starts with the experiment name
        if (os.path.isdir(dir_path) and 
            dir_name.startswith(f"{experiment_name}_")):
            
            config_path = os.path.join(dir_path, 'config.yaml')
            
            # Check if config.yaml exists
            if os.path.exists(config_path):
                try:
                    # Load the config from this directory
                    candidate_cfg = OmegaConf.load(config_path)
                    
                    # Create a copy without the seed for comparison
                    candidate_cfg_no_seed = OmegaConf.create(candidate_cfg)
                    if 'seed' in candidate_cfg_no_seed:
                        del candidate_cfg_no_seed['seed']
                    
                    # Compare configs without seed
                    if OmegaConf.to_yaml(ref_cfg_no_seed) == OmegaConf.to_yaml(candidate_cfg_no_seed):
                        matching_dirs.append(dir_name)
                        
                except Exception as e:
                    print(f"Warning: Could not load config from {config_path}: {e}")
                    continue

    return matching_dirs



def generate_cde_forecast(cfg, data, encoder, generator, predictor = None, two_step = False):

    # Re-seed for deterministic forecasting per run
    try:
        cfg_seed = int(getattr(cfg, 'seed', None) or OmegaConf.to_container(cfg, resolve=True).get('seed', None))
    except Exception:
        cfg_seed = None
    if cfg_seed is not None:
        seed_everything(cfg_seed, deterministic=True)

    Xs = data['Xs']
    
    # Feed entire data at once into the encoder (no chunking)
    Xs_third_last = torch.tensor(Xs[-3], dtype=torch.float).to(device).unsqueeze(0)
    Xs_second_last = torch.tensor(Xs[-2], dtype=torch.float).to(device).unsqueeze(0)
    Xs_last = torch.tensor(Xs[-1], dtype=torch.float).to(device).unsqueeze(0)

    if two_step:
        src_latent_1 = encoder(Xs_third_last)
        src_latent_2 = encoder(Xs_second_last)
        
        src_latent_combined = torch.cat([src_latent_1, src_latent_2], dim=-1)
        src_latent = src_latent_2
    else:
        src_latent = encoder(Xs_second_last)
        src_latent_combined = src_latent
    
    if predictor is not None:
        tgt_latent = torch.tensor(predictor.predict(src_latent_combined.detach().cpu().numpy()), dtype=torch.float).to(device)
        tgt_latent = normalize_latent(tgt_latent)
    else:
        # Source-only case: target_latent should be None
        tgt_latent = None

    gen = generator.sample(Xs_second_last.squeeze(0), src_latent, tgt_latent)
    return gen.squeeze(0).cpu().numpy()


def compute_scores(cfg, data, forecast):
    dataset_cfg = DATASET_CONFIGS[cfg.dataset_name]

    Xs = data['Xs']
    Xs_last = torch.tensor(Xs[-1], dtype=torch.float).to(device)

    rbf_paper = RBF(1.0).to(device)
    mmd_loss_paper = MMDLoss(kernel=rbf_paper).to(device)

    mmd_paper = mmd_loss_paper(Xs_last, torch.from_numpy(forecast).to(device)).item()
    #mmd_paper = mmd_loss_paper(Xs_last, forecast).item()

    emd_val = earth_mover_distance(Xs_last.cpu().numpy(), forecast)
    emd = float(emd_val) if not np.isnan(emd_val) else None

    return mmd_paper, emd

    
    
def setup_pca_for_pbmc() -> PCA:
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
    return pca


def transform_for_plot(dataset_name: str, data: np.ndarray, pca: PCA = None) -> np.ndarray:
    if dataset_name == 'pbmc' and pca is not None:
        return pca.transform(data)
    return data


def plot_forecast(cfg, data, forecast):

    dataset_name = cfg.dataset_name
    dataset_cfg = DATASET_CONFIGS[dataset_name]
    
    pca = None
    if dataset_name == 'pbmc' and DATASET_CONFIGS['pbmc'].get('requires_pca', False):
        pca = setup_pca_for_pbmc()

    Xs = data['Xs']
    Xs_training = Xs[:-1]
    Xs_last = Xs[-1]
    
    is_3d = dataset_cfg.get('plot_dimensionality', dataset_cfg['dimensionality']) == 3
    if is_3d:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    n_sequences = len(Xs_training)
    
    
    all_training = []
    colors = []
    for i, X in enumerate(Xs_training):
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

    true_data = transform_for_plot(dataset_name, Xs_last, pca)
    forecast_data = transform_for_plot(dataset_name, forecast, pca)

    if is_3d:
        ax.scatter(true_data[:, 0], true_data[:, 1], true_data[:, 2], alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_data[:, 0], forecast_data[:, 1], forecast_data[:, 2], alpha=0.9, s=8.0, color='darkorange', label='Forecast', marker='s', edgecolor='white', linewidth=0.5)
        ax.set_zlabel(dataset_cfg['axes_labels'][2] if len(dataset_cfg['axes_labels']) > 2 else 'Z')
    else:
        ax.scatter(true_data[:, 0], true_data[:, 1], alpha=0.9, s=8.0, color='darkgreen', label='Ground Truth', marker='o', edgecolor='white', linewidth=0.5)
        ax.scatter(forecast_data[:, 0], forecast_data[:, 1], alpha=0.9, s=8.0, color='darkorange', label='Forecast', marker='s', edgecolor='white', linewidth=0.5)
        ax.grid(True)

    ax.set_xlabel(dataset_cfg['axes_labels'][0])
    ax.set_ylabel(dataset_cfg['axes_labels'][1])
    ax.set_title('Training Data, Ground Truth & Forecast Phase Portrait')
    ax.legend()
    plt.tight_layout()
    plt.show()

    



#outputs_dir = "outputs_gom_looks_great_but_pbmc_buggy_01_20_2026"
outputs_dir = "outputs"

ckpt_dir_ref = None
for ckpt_dir in os.listdir(outputs_dir):
    if ckpt_dir.startswith("snapMMD_MFM_gnn_P"):
        try:
            cfg_ref,_ = load_cfg_and_ckpt(ckpt_dir,outputs_dir=outputs_dir)
            if True:
                ckpt_dir_ref = ckpt_dir
                break
        except Exception as e:
            # Skip directories that fail to load (e.g., missing config.yaml)
            continue

    

#print(cfg_ref['experiment']['predictor_loss_weight'])
#print(cfg_ref['experiment']['selective_pairing_mode'])

print(f"Using reference checkpoint directory: {ckpt_dir_ref}")

matching_dirs = find_matching_ckpt_dirs(ckpt_dir_ref,outputs_dir=outputs_dir)

#print(cfg_ref)

plot_seed = 0
all_mmd = []
all_emd = []
use_predictor = False
predictor_source = "posthoc"
two_step = True

for j, ckpt_dir in enumerate(matching_dirs):
    cfg, ckpt_path = load_cfg_and_ckpt(ckpt_dir,outputs_dir=outputs_dir)
    encoder, generator = load_models(cfg, ckpt_path)
    data = np.load(DATASET_CONFIGS[cfg.dataset_name]['data_path'])

    if use_predictor and predictor_source == "posthoc":
        predictor = get_ridge(encoder, data, device=device, alpha = 0.001, seed=42, two_step=two_step)
    
    elif use_predictor and predictor_source == "cotrained":
        pass
        
    else:
        predictor = None

    forecast = generate_cde_forecast(cfg, data, encoder, generator, predictor = predictor, two_step=two_step)
    mmd, emd = compute_scores(cfg, data, forecast)
    print(mmd, emd)
    all_mmd.append(mmd)
    all_emd.append(emd)
    if j == plot_seed:
        print(forecast.shape, type(forecast), forecast.device)
        plot_forecast(cfg, data, forecast)


all_mmd = np.array(all_mmd)
all_emd = np.array(all_emd)

print("MMD: ", np.mean(all_mmd), np.std(all_mmd))
print("EMD: ", np.mean(all_emd), np.std(all_emd))

ckpt_dir_ref = None
matching_dirs = None
cfg = None
ckpt_path = None
encoder = None
generator = None
predictor = None
        

