from utils.experiment_utils import load_best_model, get_experiment_info
import hydra
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import matplotlib.cm as cm
import argparse

from snapMMD.dls import MMDLoss, snapMMD, RBF

device = 'cuda'
original_cwd = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/" #hydra.utils.get_original_cwd()

def find_experiment_with_hash(dataset_name, base_dir):
    """
    Find the experiment directory that matches the pattern snapMMD_{dataset}_coupled_exp_<hash>
    
    Args:
        dataset_name: One of ["LV", "Repressilator", "GoM", "PBMC"]
        base_dir: The outputs directory containing experiment directories
        
    Returns:
        The full path to the matching experiment directory
    """
    pattern = f"snapMMD_{dataset_name}_coupled_exp_"
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory {base_dir} does not exist")
    
    matching_dirs = []
    for item in os.listdir(base_dir):
        full_path = os.path.join(base_dir, item)
        if os.path.isdir(full_path) and item.startswith(pattern):
            # We want the one with the hash, not the bare name
            if item != f"snapMMD_{dataset_name}_coupled_exp":
                matching_dirs.append(item)
    
    if len(matching_dirs) == 0:
        raise ValueError(f"No experiment directory found matching pattern {pattern}<hash>")
    elif len(matching_dirs) > 1:
        # If multiple matches, we might want to pick the most recent or ask user
        # For now, let's just pick the first one and warn
        print(f"Warning: Multiple directories found matching {pattern}*, using {matching_dirs[0]}")
    
    return os.path.join(base_dir, matching_dirs[0])

def load_experiment_by_dataset(dataset_name, base_dir):
    """
    Load experiment configuration and directory for a given dataset.
    
    Args:
        dataset_name: One of ["LV", "Repressilator", "GoM", "PBMC"]
        base_dir: The outputs directory containing experiment directories
        
    Returns:
        Dictionary containing experiment info (config, dir, etc.)
    """
    valid_datasets = ["LV", "Repressilator", "GoM", "PBMC"]
    if dataset_name not in valid_datasets:
        raise ValueError(f"Dataset name must be one of {valid_datasets}, got {dataset_name}")
    
    experiment_dir = find_experiment_with_hash(dataset_name, base_dir)
    experiment_info = get_experiment_info(experiment_dir, load_checkpoints=False)
    
    return experiment_info

# Parse command line arguments
parser = argparse.ArgumentParser(description='SnapMMD forecasting with different datasets')
parser.add_argument('--dataset', type=str, default='GoM', 
                   choices=['LV', 'Repressilator', 'GoM', 'PBMC'],
                   help='Dataset to use for forecasting (default: GoM)')
args = parser.parse_args()

# Load experiment configuration
experiment_info = load_experiment_by_dataset(args.dataset, os.path.join(original_cwd, 'outputs'))
print(f"Loading experiment from: {experiment_info['dir']}")
print(f"Using dataset: {args.dataset}")


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
    state = load_best_model(path)
    enc.load_state_dict(state['encoder_state_dict'])
    gen.model.load_state_dict(state['generator_state_dict'])
    enc.eval()
    gen.eval()
    enc.to(device)
    gen.to(device)
    return enc, gen

enc, gen = load_model(experiment_info['config'], experiment_info['dir'], device)


data = np.load(f"{os.path.join(original_cwd, 'data/realdata', 'GoM_data.npz')}")["Xs"]
samples_s = torch.tensor(data[-2]).unsqueeze(0).to(device).float()  # Add batch dimension and convert to float32
samples_t = torch.tensor(data[-1]).unsqueeze(0).to(device).float()  # Add batch dimension and convert to float32
print(samples_s.shape)
print(samples_t.shape)
print(type(samples_s))
print(type(samples_t))
print(samples_s.dtype)
print(samples_t.dtype)
print("shapes printed")

with torch.no_grad():
    enc_s = enc(samples_s)#.detach().cpu().numpy()
    enc_t = enc(samples_t)#.detach().cpu().numpy()

batch_size, set_size, *data_shape = samples_s.shape
samples_s = samples_s.reshape(-1, *data_shape)

print(type(enc_s))
print("encodings printed")
print(enc_s.shape)
print(enc_t.shape)
print("encodings shapes printed")


forecast = gen.sample(samples_s, enc_s, enc_t)
print(forecast.shape)
print(forecast.dtype)
print("forecast printed")

rbf = RBF().to(device)
myMMD = MMDLoss(kernel = rbf).to(device)

mmd_squared = myMMD(forecast[0], samples_t[0])

print(f"MMD^2 between forecast and validation data: {mmd_squared.item():.6f}")

# Create scatterplot visualization
plt.figure(figsize=(12, 8))

# Plot data[i] for i = 0,...,10 with different colors
colors = cm.get_cmap('tab20')(np.linspace(0, 1, 11))  # Generate 11 distinct colors
for i in range(11):
    data_points = data[i]  # Shape: (200, 2)
    plt.scatter(data_points[:, 0], data_points[:, 1], 
               c=[colors[i]], alpha=0.6, s=20, label=f'Timepoint {i}')

# Plot forecast[0] with a distinct color
forecast_cpu = forecast[0].detach().cpu().numpy()  # Convert to numpy and move to CPU
#plt.scatter(forecast_cpu[:, 0], forecast_cpu[:, 1], 
#           alpha=0.6, s=30, c='red', label='CDE Forecast (Ours)', marker='x')
#
## Load and plot the saved snapMMD forecast for comparison
#saved_forecast_file = os.path.join(original_cwd, 'snapMMD_forecasts/GoM_forecast_42.npz')
#if os.path.exists(saved_forecast_file):
#    saved_data = np.load(saved_forecast_file)
#    saved_forecast = saved_data['forecast']
#    
#    # Get the final forecast point (same logic as in plot_forecast_vs_true)
#    if saved_forecast.ndim == 3:
#        saved_forecast_final = saved_forecast[-1]  # Final time point
#    else:
#        saved_forecast_final = saved_forecast
#        
#    plt.scatter(saved_forecast_final[:, 0], saved_forecast_final[:, 1], 
#               alpha=0.6, s=30, c='orange', label='SnapMMD Forecast', marker='+')
#else:
#    print(f"Warning: Saved forecast file not found at {saved_forecast_file}")

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Gulf of Mexico, Forecasting Final Timepoint')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the figure
plt.savefig('data_forecast_scatterplot__.png', dpi=300, bbox_inches='tight')
plt.savefig('data_forecast_scatterplot__.pdf', bbox_inches='tight')
print("Figure saved as 'data_forecast_scatterplot.png' and 'data_forecast_scatterplot.pdf'")

plt.show()
