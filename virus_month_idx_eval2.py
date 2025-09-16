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


ckpt_dir = "outputs/virus_time_and_location_4202a380ab61bc2c774fe6cbb52c7e83"
cfg, ckpt_path, encoder, generator, dataset = load_all(ckpt_dir)
idx = 0
print(dataset[idx]["source_samples"]["esm_input_ids"].shape)


ESM_baseline_model = ProteinSetEncoder().to(device).eval()

loader = DataLoader(dataset, batch_size=2, shuffle=True)



# =============================
# Collect features over 5 epochs and train Ridge regression to predict source_idx
# =============================

from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error


num_collect_epochs = 1

all_feats_baseline = []
all_feats_trained = []
all_labels = []

with torch.no_grad():
    for epoch in range(num_collect_epochs):
        draws = 0
        for j, batch in enumerate(loader):
            if j > 500:
                break
            with open("auxillary_log.log", "a") as f:
                f.write(f"epoch: {epoch}, draws: {draws}\n")
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
            print("!",lat_baseline.shape)
            print("!",lat_trained.shape)
            all_feats_baseline.append(lat_baseline.detach().cpu())
            all_feats_trained.append(lat_trained.detach().cpu())

            src_idx = batch['source_idx']
            if isinstance(src_idx, torch.Tensor):
                src_idx = src_idx.view(-1).long().cpu()
            else:
                src_idx = torch.tensor([src_idx], dtype=torch.long)
            all_labels.append(src_idx)

            draws += 1


# Concatenate features from both models
X_baseline = torch.cat(all_feats_baseline, dim=0)
X_trained = torch.cat(all_feats_trained, dim=0)
y = torch.cat(all_labels, dim=0)

# Convert to numpy arrays for sklearn
X_baseline_numpy = X_baseline.numpy()
X_trained_numpy = X_trained.numpy()
y_numpy = y.numpy()

# Save the data arrays to npz file
data_filename = f"features_and_labels_{len(y_numpy)}_samples.npz"
np.savez(data_filename, 
         X_baseline=X_baseline_numpy, 
         X_trained=X_trained_numpy, 
         y=y_numpy)
print(f"Saved data arrays to {data_filename}")
print(f"Data shapes: X_baseline={X_baseline_numpy.shape}, X_trained={X_trained_numpy.shape}, y={y_numpy.shape}")


# Configuration for model type
USE_MLP = True  # Set to True for MLP, False for Ridge regression

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evaluate a model and return comprehensive metrics"""
    # Train predictions
    preds_train = model.predict(X_train)
    preds_train_rounded = np.round(preds_train).astype(int)
    
    # Validation predictions
    preds_val = model.predict(X_val)
    preds_val_rounded = np.round(preds_val).astype(int)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, preds_train)
    train_mae = mean_absolute_error(y_train, preds_train)
    train_accuracy = accuracy_score(y_train, preds_train_rounded)
    
    val_mse = mean_squared_error(y_val, preds_val)
    val_mae = mean_absolute_error(y_val, preds_val)
    val_accuracy = accuracy_score(y_val, preds_val_rounded)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Training   - MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, Accuracy: {train_accuracy:.4f}")
    print(f"Validation - MSE: {val_mse:.4f}, MAE: {val_mae:.4f}, Accuracy: {val_accuracy:.4f}")
    
    return {
        'train_mse': train_mse, 'train_mae': train_mae, 'train_accuracy': train_accuracy,
        'val_mse': val_mse, 'val_mae': val_mae, 'val_accuracy': val_accuracy,
        'train_preds': preds_train, 'val_preds': preds_val
    }

idx = np.arange(len(y_numpy))
train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=None)
X_baseline_train, X_baseline_val = X_baseline_numpy[train_idx], X_baseline_numpy[val_idx]
X_trained_train, X_trained_val = X_trained_numpy[train_idx],  X_trained_numpy[val_idx]
y_train,  y_val  = y_numpy[train_idx],          y_numpy[val_idx]

print(f"\nData split - Train: {len(y_train)} samples, Validation: {len(y_val)} samples")
print(f"Train labels range: [{y_train.min()}, {y_train.max()}]")
print(f"Val labels range: [{y_val.min()}, {y_val.max()}]")

# Train and evaluate models
if USE_MLP:
    print(f"\n{'='*50}")
    print("Training MLP Regressors")
    print(f"{'='*50}")
    
    # MLP for baseline features
    mlp_baseline = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    mlp_baseline.fit(X_baseline_train, y_train)
    baseline_results = evaluate_model(mlp_baseline, X_baseline_train, X_baseline_val, 
                                    y_train, y_val, "MLP on Baseline ESM Features")
    
    # MLP for trained encoder features
    mlp_trained = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20
    )
    mlp_trained.fit(X_trained_train, y_train)
    trained_results = evaluate_model(mlp_trained, X_trained_train, X_trained_val, 
                                   y_train, y_val, "MLP on Trained Encoder Features")
    
else:
    print(f"\n{'='*50}")
    print("Training Ridge Regressors")
    print(f"{'='*50}")
    
    # Ridge regression for baseline features
    ridge_baseline = Ridge(alpha=1.0)
    ridge_baseline.fit(X_baseline_train, y_train)
    baseline_results = evaluate_model(ridge_baseline, X_baseline_train, X_baseline_val, 
                                    y_train, y_val, "Ridge on Baseline ESM Features")
    
    # Ridge regression for trained encoder features
    ridge_trained = Ridge(alpha=1.0)
    ridge_trained.fit(X_trained_train, y_train)
    trained_results = evaluate_model(ridge_trained, X_trained_train, X_trained_val, 
                                   y_train, y_val, "Ridge on Trained Encoder Features")

# Summary comparison
model_type = "MLP" if USE_MLP else "Ridge"
print(f"\n{'='*50}")
print(f"Performance Comparison ({model_type})")
print(f"{'='*50}")
print(f"{'Model':<25} {'Train Acc':<10} {'Val Acc':<10} {'Train MSE':<10} {'Val MSE':<10}")
print("-" * 70)
print(f"{'Baseline ESM':<25} {baseline_results['train_accuracy']:<10.4f} {baseline_results['val_accuracy']:<10.4f} {baseline_results['train_mse']:<10.4f} {baseline_results['val_mse']:<10.4f}")
print(f"{'Trained Encoder':<25} {trained_results['train_accuracy']:<10.4f} {trained_results['val_accuracy']:<10.4f} {trained_results['train_mse']:<10.4f} {trained_results['val_mse']:<10.4f}")

# Additional statistics
print(f"\nDetailed Prediction Statistics:")
print(f"Baseline - Train pred range: [{baseline_results['train_preds'].min():.3f}, {baseline_results['train_preds'].max():.3f}]")
print(f"Baseline - Val pred range: [{baseline_results['val_preds'].min():.3f}, {baseline_results['val_preds'].max():.3f}]")
print(f"Trained - Train pred range: [{trained_results['train_preds'].min():.3f}, {trained_results['train_preds'].max():.3f}]")
print(f"Trained - Val pred range: [{trained_results['val_preds'].min():.3f}, {trained_results['val_preds'].max():.3f}]")