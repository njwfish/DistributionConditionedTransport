# imports
from pathlib import Path
import sys  
import torch
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as mcolors

import hydra

from utils.experiment_utils import get_all_experiments_info, load_best_model

from scipy.linalg import sqrtm
# Get my_package directory path from Notebook
parent_dir = str(Path().resolve().parents[0])

# Add to sys.path
sys.path.insert(0, parent_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
configs = get_all_experiments_info('outputs/', False)
print(len(configs))
cfgs = [
    c for c in configs if 'mvn_exp_' in c['name']]
print(len(cfgs))
print(cfgs[0])
# load + prep dataset
def prepare_dataset(dataset_cfg):
    # probs = np.column_stack((np.linspace(0, 1, num_probs), 1 - np.linspace(0, 1, num_probs)))
    dataset = hydra.utils.instantiate(dataset_cfg)
    # dataset.probs = probs
    # dataset.data, _, _ = dataset.make_sets()
    return dataset

# load encoder and move to device
def load_model(cfg, path, device):
    enc = hydra.utils.instantiate(cfg['encoder'])
    gen = hydra.utils.instantiate(cfg['generator'])
    if hasattr(cfg, "predictor"):
        predictor = hydra.utils.instantiate(cfg.predictor)
        # SELU by default but adding this to make sure it's the same as the encoder
        predictor.latent_act = enc.latent_act
        enc.predictor = predictor
    state = load_best_model(path)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.load_state_dict(state['generator_state_dict'])
    enc.eval()
    gen.eval()
    enc.to(device)
    gen.to(device)
    return enc, gen

# X is (n, m, d)
def vectorized_covariance(X):
    # Center the data
    X_centered = X - X.mean(axis=1, keepdims=True)  # shape (n, m, d)

    # Compute covariance: cov = (X^T X) / (m - 1)
    cov = np.einsum('nmd,nme->nde', X_centered, X_centered) / (X.shape[1] - 1)
    return cov

def process_single_config(cfg, device):
    """Process a single config and return GDE trajectory means and covariances"""
    dir_name = cfg['dir']
    enc, gen = load_model(cfg['config'], dir_name, device)

    ds = prepare_dataset(cfg['config']['dataset'])
    s1 = ds.sample(ds.mu[2][None, :], ds.cov[2][None, :, :], 1, 100_000, (2,))
    s2 = ds.sample(ds.mu[1][None, :], ds.cov[1][None, :, :], 1, 100_000, (2,))

    lat = enc(torch.from_numpy(np.concatenate([s1, s2], axis=0)).float().to(device))
    # linearly interpolate between the two points
    lat_interp = lat[0] + (lat[1] - lat[0]) * torch.linspace(0, 1, 20)[:, None].to(device)
    lat_start = lat[0] + torch.zeros_like(lat[1] - lat[0]) * torch.linspace(0, 1, 20)[:, None].to(device)
    
    # Convert s1 to torch tensor and expand to match lat_start/lat_interp first dimension
    s1_expanded = torch.from_numpy(s1).float().to(device).expand(20, -1, -1)
    # Reshape to 2D: (20 * 100000, 2) for ODE integration
    s1_tensor = s1_expanded.reshape(-1, s1_expanded.shape[-1])
    resample = gen.sample(s1_tensor, lat_start, lat_interp, 1_000_000, return_trajectory=False)

    # compute means and covariances of all the resampled points
    resample_means = np.mean(resample.cpu().numpy(), axis=1)
    resample_covs = vectorized_covariance(resample.cpu().numpy())
    
    return resample_means, resample_covs

# Group configs by predictor_loss_weight and predictor type, filtering out "unidirectional" sampling mode
configs_by_weight_and_type = defaultdict(lambda: defaultdict(list))
for cfg in cfgs:
    # Skip configs with unidirectional sampling mode
    sampling_mode = cfg['config']['sampling']['mode']
    if sampling_mode == 'unidirectional':
        continue
        
    predictor_weight = cfg['config']['experiment']['predictor_loss_weight']
    predictor_target = cfg['config']['predictor']['_target_']
    # Extract predictor type from target string
    if 'DTConditionedRidgePredictor' in predictor_target:
        predictor_type = 'Ridge'
    elif 'DTConditionedMLPPredictor' in predictor_target:
        predictor_type = 'MLP'
    else:
        predictor_type = 'Unknown'
    
    configs_by_weight_and_type[predictor_weight][predictor_type].append(cfg)

print(f"Predictor loss weights found: {list(configs_by_weight_and_type.keys())}")
for weight, type_dict in configs_by_weight_and_type.items():
    print(f"Weight {weight}:")
    for pred_type, configs in type_dict.items():
        print(f"  {pred_type}: {len(configs)} configs")


def generate_ot_trajectory(mean1, cov1, mean2, cov2, n_steps=10):
    """
    Generate the correct Optimal Transport trajectory between two Gaussian distributions.
    
    Parameters:
    -----------
    mean1 : array-like
        Initial mean (2D vector)
    cov1 : array-like
        Initial covariance matrix (2x2)
    mean2 : array-like
        Final mean (2D vector)
    cov2 : array-like
        Final covariance matrix (2x2)
    n_steps : int
        Number of interpolation steps
        
    Returns:
    --------
    means : array-like
        List of interpolated means
    covs : array-like
        List of interpolated covariance matrices
    """
    mean1 = np.array(mean1)
    mean2 = np.array(mean2)
    cov1 = np.array(cov1)
    cov2 = np.array(cov2)
    
    # Generate interpolation parameters
    ts = np.linspace(0, 1, n_steps)
    
    # Initialize lists for means and covariances
    means = []
    covs = []
    
    # Compute the square root of cov1
    sqrt_cov1 = sqrtm(cov1)
    inv_sqrt_cov1 = np.linalg.inv(sqrt_cov1)
    
    # Compute A = (cov1^(1/2) * cov2 * cov1^(1/2))^(1/2)
    A = sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1)
    
    for t in ts:
        # Interpolate means linearly
        mean_t = (1 - t) * mean1 + t * mean2
        
        # Correct interpolation formula for covariances
        inner = ((1 - t) * cov1 + t * sqrtm(sqrt_cov1 @ cov2 @ sqrt_cov1))
        cov_t = inv_sqrt_cov1 @ (inner @ inner) @ inv_sqrt_cov1
        
        means.append(mean_t)
        covs.append(cov_t)
    
    return np.array(means), np.array(covs)

# These lines are no longer needed as we process all configs individually



plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('axes', titlesize=16)

def plot_gaussian_trajectories_on_ax(ax, means, covs, alpha=0.2, n_std=2, title=None):
    """
    Plot trajectories of Gaussian distributions with their confidence ellipses on given axis.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    means : array-like
        List of 2D means, shape (n_timesteps, 2)
    covs : array-like
        List of 2x2 covariance matrices, shape (n_timesteps, 2, 2)
    alpha : float
        Transparency of ellipses
    n_std : float
        Number of standard deviations for ellipse size
    title : str
        Title for the subplot
    """
    
    # Create color gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(means)))
    
    # Plot trajectories and ellipses
    for i in range(len(means)):
        mean = means[i]
        cov = covs[i]
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate angle and axes lengths for ellipse
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width, height = 2 * n_std * np.sqrt(eigenvals)
        
        # Create and add ellipse
        ellipse = Ellipse(xy=mean,
                         width=width,
                         height=height,
                         angle=angle,
                         facecolor=colors[i],
                         alpha=alpha,
                         edgecolor=colors[i],
                         linewidth=2,
                         zorder=10)
        ax.add_patch(ellipse)
        
        # Plot mean point
        ax.plot(mean[0], mean[1], 'o', color=colors[i], markersize=3, zorder=50)
        
    # Plot trajectory line
    means_array = np.array(means)
    ax.plot(means_array[:, 0], means_array[:, 1], '-', color='black', alpha=0.5, zorder=49)
    
    # Set equal aspect ratio and grid
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)

    if title is not None:
        ax.set_title(title, fontsize=10)

def create_panel_for_weight(weight, configs_by_type, device):
    """Create a single 2x4 panel with Ridge (left 2x2) and MLP (right 2x2) for the same predictor_loss_weight"""
    
    # Create subplot grid (2 rows x 4 cols)
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    
    # Process Ridge configs (left 2x2 panel: columns 0-1)
    if 'Ridge' in configs_by_type:
        ridge_results = []
        for cfg in configs_by_type['Ridge']:
            try:
                gde_means, gde_covs = process_single_config(cfg, device)
                mode = cfg['config']['sampling']['mode']
                conditioning_mode = cfg['config']['predictor']['conditioning_mode']
                ridge_results.append((gde_means, gde_covs, mode, conditioning_mode))
                print(f"Processed Ridge config with weight={weight}, mode={mode}, conditioning_mode={conditioning_mode}")
            except Exception as e:
                print(f"Error processing Ridge config: {e}")
                continue
        
        # Fill left 2x2 panel (Ridge)
        for i, (gde_means, gde_covs, mode, conditioning_mode) in enumerate(ridge_results):
            if i < 4:  # Safety check for 2x2 grid
                row = i // 2
                col = i % 2
                title = f"Ridge\nsampling mode={mode}\nconditioning method={conditioning_mode}"
                plot_gaussian_trajectories_on_ax(axes[row, col], gde_means, gde_covs, title=title)
    
    # Process MLP configs (right 2x2 panel: columns 2-3)
    if 'MLP' in configs_by_type:
        mlp_results = []
        for cfg in configs_by_type['MLP']:
            try:
                gde_means, gde_covs = process_single_config(cfg, device)
                mode = cfg['config']['sampling']['mode']
                conditioning_mode = cfg['config']['predictor']['conditioning_mode']
                mlp_results.append((gde_means, gde_covs, mode, conditioning_mode))
                print(f"Processed MLP config with weight={weight}, mode={mode}, conditioning_mode={conditioning_mode}")
            except Exception as e:
                print(f"Error processing MLP config: {e}")
                continue
        
        # Fill right 2x2 panel (MLP)
        for i, (gde_means, gde_covs, mode, conditioning_mode) in enumerate(mlp_results):
            if i < 4:  # Safety check for 2x2 grid
                row = i // 2
                col = 2 + (i % 2)  # Offset by 2 to use columns 2-3
                title = f"MLP\nsampling mode={mode}\nconditioning method={conditioning_mode}"
                plot_gaussian_trajectories_on_ax(axes[row, col], gde_means, gde_covs, title=title)
    
    fig.suptitle(f'GDE Trajectories - Predictor Loss Weight = {weight}', fontsize=18)
    plt.tight_layout()
    
    return fig


# Process all predictor_loss_weight groups and create panels
for weight in sorted(configs_by_weight_and_type.keys()):
    configs_by_type = configs_by_weight_and_type[weight]
    total_configs = sum(len(configs) for configs in configs_by_type.values())
    print(f"\nProcessing predictor_loss_weight = {weight} with {total_configs} configs")
    
    # Create panel for this weight (includes both Ridge and MLP)
    panel_fig = create_panel_for_weight(weight, configs_by_type, device)
    
    # Save the panel
    filename = f'mvn_panel_weight_{weight}.png'
    panel_fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(panel_fig)  # Free memory
    
    print(f"Saved panel for weight {weight} to {filename}")

print("\nAll panels created successfully!")

