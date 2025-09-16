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

loader = DataLoader(dataset, batch_size=1, shuffle=True)

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
    print(batch['source_idx'].shape)
    print(batch['target_idx'].shape)
    print(batch['d'].shape)
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

# =============================
# Collect features over 5 epochs and train MLP to predict source_idx
# =============================

from torch import nn
from torch.optim import Adam


num_collect_epochs = 5
max_draws_per_epoch = getattr(dataset, 'max_draws_per_epoch', None)

all_feats = []
all_labels = []

with torch.no_grad():
    for epoch in range(num_collect_epochs):
        draws = 0
        for batch in loader:
            # Move nested dict tensors to device
            source_samples = {}
            for key, value in batch['source_samples'].items():
                if isinstance(value, torch.Tensor):
                    source_samples[key] = value.to(device)
                else:
                    source_samples[key] = value

            lat = ESM_baseline_model(source_samples)  # [B, latent_dim]
            all_feats.append(lat.detach().cpu())

            src_idx = batch['source_idx']
            if isinstance(src_idx, torch.Tensor):
                src_idx = src_idx.view(-1).long().cpu()
            else:
                src_idx = torch.tensor([src_idx], dtype=torch.long)
            all_labels.append(src_idx)

            draws += 1
            if (max_draws_per_epoch is not None) and (draws >= max_draws_per_epoch):
                break

X = torch.cat(all_feats, dim=0)
y = torch.cat(all_labels, dim=0)

latent_dim = X.shape[1]
num_classes = len(dataset.data)

class MLPClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLPClassifier(latent_dim, 128, num_classes).to(device)
optimizer = Adam(mlp.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

mlp.train()
batch_size = 64
num_train_epochs = 20

for e in range(num_train_epochs):
    perm = torch.randperm(X.size(0))
    total_loss = 0.0
    correct = 0
    count = 0
    for i in range(0, X.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = X[idx].to(device)
        yb = y[idx].to(device)

        optimizer.zero_grad()
        logits = mlp(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        count += xb.size(0)

    avg_loss = total_loss / max(count, 1)
    acc = correct / max(count, 1)
    print(f"[MLP Train] epoch={e+1}/{num_train_epochs} loss={avg_loss:.4f} acc={acc:.4f} N={count}")

mlp.eval()
with torch.no_grad():
    logits = mlp(X.to(device))
    preds = logits.argmax(dim=1).cpu()
    final_acc = (preds == y).float().mean().item()
print(f"Final train accuracy on collected dataset: {final_acc:.4f}")