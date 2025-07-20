from utils.experiment_utils import get_all_experiments_info, load_best_model
import hydra
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

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

enc, gen = load_model(configs[0]['config'], configs[0]['dir'], device)


data = np.load(f"{os.path.join(original_cwd, 'data/realdata', 'processed_pbmc_data_sub500_every_2_until20.npz')}")["Xs"]
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
