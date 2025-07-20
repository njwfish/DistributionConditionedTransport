from utils.experiment_utils import get_all_experiments_info, load_best_model
import hydra
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os
import matplotlib.cm as cm

from snapMMD.dls import MMDLoss, snapMMD, RBF

device = 'cuda'
original_cwd = "/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/" #hydra.utils.get_original_cwd()
configs = get_all_experiments_info(os.path.join(original_cwd, 'outputs'), False)
print(len(configs))


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

enc, gen = load_model(configs[1]['config'], configs[1]['dir'], device)


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
