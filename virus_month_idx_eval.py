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
# Collect features over 5 epochs and train Ridge regression to predict source_idx
# =============================

from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score


num_collect_epochs = 5
max_draws_per_epoch = getattr(dataset, 'max_draws_per_epoch', None)

all_feats_baseline = []
all_feats_trained = []
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

            # Get features from both models
            lat_baseline = ESM_baseline_model(source_samples)  # [B, latent_dim]
            lat_trained = encoder(source_samples)  # [B, latent_dim]
            
            all_feats_baseline.append(lat_baseline.detach().cpu())
            all_feats_trained.append(lat_trained.detach().cpu())

            src_idx = batch['source_idx']
            if isinstance(src_idx, torch.Tensor):
                src_idx = src_idx.view(-1).long().cpu()
            else:
                src_idx = torch.tensor([src_idx], dtype=torch.long)
            all_labels.append(src_idx)

            draws += 1
            if (max_draws_per_epoch is not None) and (draws >= max_draws_per_epoch):
                break

# Concatenate features from both models
X_baseline = torch.cat(all_feats_baseline, dim=0)
X_trained = torch.cat(all_feats_trained, dim=0)
y = torch.cat(all_labels, dim=0)

# Combine features from both models
X_combined = torch.cat([X_baseline, X_trained], dim=1)

num_classes = len(dataset.data)

# Convert to numpy arrays for sklearn
X_baseline_numpy = X_baseline.numpy()
X_trained_numpy = X_trained.numpy()
X_combined_numpy = X_combined.numpy()
y_numpy = y.numpy()

print(f"Dataset shapes:")
print(f"  Baseline features: {X_baseline_numpy.shape}")
print(f"  Trained encoder features: {X_trained_numpy.shape}")
print(f"  Combined features: {X_combined_numpy.shape}")
print(f"  Labels: {y_numpy.shape}")

# Diagnostic information about X_trained_numpy
print(f"\n=== Diagnostic Info for Trained Encoder Features ===")
print(f"X_trained_numpy shape: {X_trained_numpy.shape}")
print(f"Data type: {X_trained_numpy.dtype}")

# Basic statistics
print(f"\nBasic statistics:")
print(f"  Mean: {X_trained_numpy.mean():.6f}")
print(f"  Std:  {X_trained_numpy.std():.6f}")
print(f"  Min:  {X_trained_numpy.min():.6f}")
print(f"  Max:  {X_trained_numpy.max():.6f}")

# Check for special cases
all_zeros = np.all(X_trained_numpy == 0)
all_same = np.all(X_trained_numpy == X_trained_numpy[0, 0])
print(f"\nSpecial cases:")
print(f"  All zeros: {all_zeros}")
print(f"  All same value: {all_same}")

# Check if all rows are identical
if X_trained_numpy.shape[0] > 1:
    rows_identical = np.all(X_trained_numpy[0] == X_trained_numpy[1:], axis=1).all()
    print(f"  All rows identical: {rows_identical}")
else:
    print(f"  Only one row, can't check row similarity")

# Show some sample values from different parts of the matrix
print(f"\nSample values:")
print(f"  First row, first 10 elements: {X_trained_numpy[0, :10]}")
if X_trained_numpy.shape[0] > 1:
    print(f"  Second row, first 10 elements: {X_trained_numpy[1, :10]}")
if X_trained_numpy.shape[0] > 2:
    print(f"  Last row, first 10 elements: {X_trained_numpy[-1, :10]}")

# Check variance across features
feature_variances = np.var(X_trained_numpy, axis=0)
zero_variance_features = np.sum(feature_variances == 0)
print(f"\nFeature analysis:")
print(f"  Features with zero variance: {zero_variance_features}/{X_trained_numpy.shape[1]}")
print(f"  Feature variance range: [{feature_variances.min():.8f}, {feature_variances.max():.8f}]")

# Check for NaN or inf values
has_nan = np.isnan(X_trained_numpy).any()
has_inf = np.isinf(X_trained_numpy).any()
print(f"\nData quality:")
print(f"  Contains NaN: {has_nan}")
print(f"  Contains Inf: {has_inf}")

# Compare first few samples to see if there's variation
if X_trained_numpy.shape[0] >= 3:
    print(f"\nRow similarity (Euclidean distances):")
    from scipy.spatial.distance import euclidean
    dist_01 = euclidean(X_trained_numpy[0], X_trained_numpy[1])
    dist_02 = euclidean(X_trained_numpy[0], X_trained_numpy[2])
    dist_12 = euclidean(X_trained_numpy[1], X_trained_numpy[2])
    print(f"  Distance row 0-1: {dist_01:.6f}")
    print(f"  Distance row 0-2: {dist_02:.6f}")
    print(f"  Distance row 1-2: {dist_12:.6f}")

print("="*50)

# Train Ridge regression on baseline features only
print("\n=== Ridge Regression on Baseline ESM Features ===")
ridge_baseline = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_baseline.fit(X_baseline_numpy, y_numpy)
preds_baseline = ridge_baseline.predict(X_baseline_numpy)
preds_baseline_rounded = np.round(preds_baseline).astype(int)
acc_baseline = accuracy_score(y_numpy, preds_baseline_rounded)
print(f"Baseline ESM accuracy: {acc_baseline:.4f}")

# Train Ridge regression on trained encoder features only
print("\n=== Ridge Regression on Trained Encoder Features ===")
ridge_trained = Ridge(alpha=0.00001, random_state=RANDOM_SEED)
ridge_trained.fit(X_trained_numpy, y_numpy)
preds_trained = ridge_trained.predict(X_trained_numpy)
preds_trained_rounded = np.round(preds_trained).astype(int)
acc_trained = accuracy_score(y_numpy, preds_trained_rounded)
print(f"Trained encoder accuracy: {acc_trained:.4f}")

# Train Ridge regression on combined features
print("\n=== Ridge Regression on Combined Features ===")
ridge_combined = Ridge(alpha=1.0, random_state=RANDOM_SEED)
ridge_combined.fit(X_combined_numpy, y_numpy)
preds_combined = ridge_combined.predict(X_combined_numpy)
preds_combined_rounded = np.round(preds_combined).astype(int)
acc_combined = accuracy_score(y_numpy, preds_combined_rounded)
print(f"Combined features accuracy: {acc_combined:.4f}")

# Summary comparison
print("\n=== Performance Comparison ===")
print(f"Baseline ESM model:     {acc_baseline:.4f}")
print(f"Trained encoder model:  {acc_trained:.4f}")
print(f"Combined features:      {acc_combined:.4f}")

improvement_trained = acc_trained - acc_baseline
improvement_combined = acc_combined - max(acc_baseline, acc_trained)

print(f"\nImprovements:")
print(f"Trained over baseline:  {improvement_trained:+.4f}")
print(f"Combined over best single: {improvement_combined:+.4f}")

# Additional statistics
print(f"\nPrediction statistics:")
print(f"True labels range: [{y_numpy.min()}, {y_numpy.max()}]")
print(f"Baseline pred range: [{preds_baseline.min():.3f}, {preds_baseline.max():.3f}]")
print(f"Trained pred range: [{preds_trained.min():.3f}, {preds_trained.max():.3f}]")
print(f"Combined pred range: [{preds_combined.min():.3f}, {preds_combined.max():.3f}]")