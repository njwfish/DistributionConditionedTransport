import os
os.chdir("/orcd/archive/abugoot/001/Projects/paolo/main_tde/")

import hydra
import argparse
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from utils.seed import seed_everything  # Import seeding utility
from torch.utils.data import DataLoader

from encoder.esm_baseline2 import ProteinSetEncoder

# Set random seed for reproducibility
RANDOM_SEED = 42
seed_everything(RANDOM_SEED, deterministic=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
def load_all(ckpt_dir):
    # Resolve and validate checkpoint directory
    experiment_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    # Load trained config and use it as the active config
    cfg = OmegaConf.load(cfg_path)

    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    generator = hydra.utils.instantiate(cfg.generator).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    encoder.eval()
    generator.eval()
    
    dataset = hydra.utils.instantiate(cfg.dataset)

    return cfg, ckpt_path, encoder, generator, dataset


ckpt_dir = "outputs/virus_time_and_location_74949f8e6ff8497f6610f31cef547b87"
cfg, ckpt_path, encoder, generator, dataset = load_all(ckpt_dir)
idx = 0
print(dataset[idx]["source_samples"]["esm_input_ids"].shape)


ESM_baseline_model = ProteinSetEncoder().to(device)

loader = DataLoader(dataset, batch_size=1)
batch = next(iter(loader))

# For dictionary samples (like PubMed dataset), move tensors to device
source_samples = {}
target_samples = {}

for key, value in batch['source_samples'].items():
    if isinstance(value, torch.Tensor):
        source_samples[key] = value.to(device)
    else:
        source_samples[key] = value
        
for key, value in batch['target_samples'].items():
    if isinstance(value, torch.Tensor):
        target_samples[key] = value.to(device)
    else:
        target_samples[key] = value
        
#x_source = x_samples["source_samples"]
#x_target = x_samples["target_samples"]

latent_source = encoder(source_samples)
latent_target = encoder(target_samples)


_, texts = generator.sample(source_samples, latent_source, latent_target, return_texts = True)

loader = DataLoader(dataset, batch_size=1)

for j, batch in enumerate(loader):
    if j > 3:
        break
    # For dictionary samples (like PubMed dataset), move tensors to device
    source_samples = {}
    target_samples = {}
    for key, value in batch['source_samples'].items():
        if isinstance(value, torch.Tensor):
            source_samples[key] = value.to(device)
        else:
            source_samples[key] = value
            
    for key, value in batch['target_samples'].items():
        if isinstance(value, torch.Tensor):
            target_samples[key] = value.to(device)
        else:
            target_samples[key] = value
    
    print(source_samples.keys())
            
    #x_source = x_samples["source_samples"]
    #x_target = x_samples["target_samples"]

    latent_source = encoder(source_samples)
    latent_target = encoder(target_samples)
    
    print(source_samples['esm_input_ids'].shape)
    print(target_samples['esm_attention_mask'].shape)
    print(latent_source.shape)
    print(latent_target.shape)
    
    latent_source_baseline = ESM_baseline_model(source_samples)
    latent_target_baseline = ESM_baseline_model(target_samples)
    
    print("!",latent_source_baseline.shape)
    print(latent_target_baseline.shape)

    _, texts = generator.sample(source_samples, latent_source, latent_target, return_texts = True)
    for text in texts:
        print(text)
    
print(texts)