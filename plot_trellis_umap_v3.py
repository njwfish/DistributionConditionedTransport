"""
Script to visualize multiple samples from the trellis dataset using UMAP.
Uses the trellis_dataset class to instantiate the dataset and visualizes
the source and target cells from multiple samples in a panel of subplots.
Each subplot shows a separate UMAP for one sample's source-target pair.
"""

import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
import sys

# Add the current directory to the path to import the dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.mfm_trellis import trellis_dataset

# Try to use GPU UMAP (cuML), fall back to CPU if not available
try:
    from cuml.manifold import UMAP
    USE_GPU = True
    print("Using GPU-accelerated UMAP (cuML)")
except ImportError:
    from umap import UMAP
    USE_GPU = False
    print("cuML not available, using CPU UMAP")

# Configuration
SPLIT_NAME = 'pdo21'  # Options: "replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"
NUM_SAMPLES = 10  # Number of samples to visualize
SAMPLE_INDICES = None  # List of specific sample indices or None for random selection
RANDOM_SEED = 42  # Random seed for sample selection
MAX_CELLS_PLOT = 1000000  # Maximum cells to plot per source/target
MAX_CELLS_UMAP = None  # Maximum cells for UMAP computation (None = use all cells)
CONTROL = set(["DMSO", "AH", "H2O"])
TREATMENT = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]
CELL_TYPE = ["PDOs", "Fibs"]

# Instantiate the dataset
print(f"Loading dataset with split_name={SPLIT_NAME}...")
dataset = trellis_dataset(
    control=CONTROL,
    treatment=TREATMENT,
    culture=["PDO", "PDOF", "F"],
    cell_type=CELL_TYPE,
    split_name=SPLIT_NAME,
    set_size=32,  # This parameter is for the __getitem__ method, not relevant for our use
    seed=0
)

# Access the samples from the dataset
samples = dataset.samples
print(f"Total samples available: {len(samples)}")

# Build a concentration mapping for each sample
# Since concentration isn't stored in samples, we need to extract it from the split data
def build_concentration_map(dataset):
    """Build a mapping from sample index to concentration used."""
    conc_map = {}
    sample_idx = 0
    
    for i in range(len(dataset.split)):
        if dataset.split_name in ["replicas-1", "replicas-2"]:
            exp = dataset.split[i]
        else:
            exp_patient = dataset.split[i]
            pdo_num = list(exp_patient.keys())[0]
            exp = exp_patient[pdo_num]
        
        x0_treatment = list(set(exp.keys()).intersection(dataset.control))[0]
        treatkeys = [key for key in exp.keys() if key not in dataset.control]
        
        for t in treatkeys:
            concentration = list(exp[t].keys())
            max_conc = str(max(map(int, concentration)))
            
            cultures_keys = list(exp[t][max_conc].keys())
            for culture in cultures_keys:
                conc_map[sample_idx] = max_conc
                sample_idx += 1
    
    return conc_map

concentration_map = build_concentration_map(dataset)

# Select samples
np.random.seed(RANDOM_SEED)

if SAMPLE_INDICES is not None:
    # Use specified indices
    selected_indices = [idx for idx in SAMPLE_INDICES if 0 <= idx < len(samples)]
    print(f"Using specified sample indices: {selected_indices}")
else:
    # Random selection
    num_to_select = min(NUM_SAMPLES, len(samples))
    selected_indices = np.random.choice(len(samples), num_to_select, replace=False).tolist()
    print(f"Randomly selected {num_to_select} sample indices: {selected_indices}")

# Determine subplot layout
n_samples = len(selected_indices)
n_cols = min(5, n_samples)  # Maximum 5 columns
n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

# Create figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4.5*n_rows))
if n_samples == 1:
    axes = np.array([axes])
axes = axes.flatten()

print(f"\nProcessing {n_samples} samples...")
print(f"Figure layout: {n_rows} rows x {n_cols} columns")

total_start_time = time.time()

# Process each sample
for plot_idx, sample_idx in enumerate(selected_indices):
    print(f"\n{'='*60}")
    print(f"Processing sample {plot_idx+1}/{n_samples} (index {sample_idx})")
    print(f"{'='*60}")
    
    # Get the sample data
    culture, x0, x1, cond_cell, cond_treat, patient = samples[sample_idx]
    
    print(f"  Culture: {culture}")
    print(f"  Patient: {patient}")
    print(f"  Source cells (x0): {x0.shape[0]}")
    print(f"  Target cells (x1): {x1.shape[0]}")
    
    # Decode cell condition (which cell types are present)
    cell_types_present = []
    if cond_cell[:, 0].sum() > 0:  # PDOs present
        cell_types_present.append(CELL_TYPE[0])
    if cond_cell[:, 1].sum() > 0:  # Fibs present
        cell_types_present.append(CELL_TYPE[1])
    cell_str = "+".join(cell_types_present) if cell_types_present else "None"
    
    # Decode treatment condition (which treatment)
    treat_idx = np.argmax(cond_treat[0])  # Get treatment index (same for all cells in sample)
    treatment_name = TREATMENT[treat_idx]
    
    # Get concentration for this sample
    concentration = concentration_map.get(sample_idx, "unknown")
    
    print(f"  Cell types: {cell_str}")
    print(f"  Treatment: {treatment_name}")
    print(f"  Concentration: {concentration}")
    
    # Prepare data for UMAP
    all_data = np.vstack([x0, x1])
    is_target = np.array([False] * x0.shape[0] + [True] * x1.shape[0])
    
    # Subsample for UMAP if requested
    if MAX_CELLS_UMAP is not None and all_data.shape[0] > MAX_CELLS_UMAP:
        subsample_indices = np.random.choice(all_data.shape[0], MAX_CELLS_UMAP, replace=False)
        all_data_umap = all_data[subsample_indices]
        is_target_umap = is_target[subsample_indices]
        print(f"  Subsampled to {MAX_CELLS_UMAP} cells for UMAP computation")
    else:
        all_data_umap = all_data
        is_target_umap = is_target
    
    print(f"  Total cells for UMAP: {all_data_umap.shape[0]} (source: {(~is_target_umap).sum()}, target: {is_target_umap.sum()})")
    
    # Run UMAP
    print(f"  Running UMAP...")
    start_time = time.time()
    
    if USE_GPU:
        reducer = UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            n_components=2, 
            random_state=42,
            verbose=False  # Less verbose for multiple samples
        )
    else:
        reducer = UMAP(
            n_neighbors=15, 
            min_dist=0.1, 
            n_components=2, 
            random_state=42,
            verbose=False,
            low_memory=True
        )
    
    embedding = reducer.fit_transform(all_data_umap)
    
    elapsed_time = time.time() - start_time
    print(f"  UMAP completed in {elapsed_time:.1f} seconds")
    
    # Get indices for source and target
    indices_source = np.where(~is_target_umap)[0]
    indices_target = np.where(is_target_umap)[0]
    
    # Subsample for plotting if needed
    if len(indices_source) > MAX_CELLS_PLOT:
        indices_source = np.random.choice(indices_source, MAX_CELLS_PLOT, replace=False)
    if len(indices_target) > MAX_CELLS_PLOT:
        indices_target = np.random.choice(indices_target, MAX_CELLS_PLOT, replace=False)
    
    # Plot in subplot
    ax = axes[plot_idx]
    
    # Plot source with blue circles
    ax.scatter(
        embedding[indices_source, 0], 
        embedding[indices_source, 1], 
        c='blue', 
        marker='o',
        s=15, 
        alpha=0.05,
        edgecolors='none',
        label=f'Source (n={len(indices_source)})'
    )
    
    # Plot target with red X markers
    ax.scatter(
        embedding[indices_target, 0], 
        embedding[indices_target, 1], 
        c='red', 
        marker='x',
        s=15, 
        alpha=0.05,
        linewidths=1,
        label=f'Target (n={len(indices_target)})'
    )
    
    ax.set_xlabel('UMAP 1', fontsize=9)
    ax.set_ylabel('UMAP 2', fontsize=9)
    ax.set_title(f'S{sample_idx}: p={patient}, {culture}, {cell_str}\nt={treatment_name}, conc={concentration}', fontsize=9)
    ax.legend(loc='best', fontsize=7, markerscale=0.8)
    ax.tick_params(labelsize=8)

# Hide unused subplots
for idx in range(n_samples, len(axes)):
    axes[idx].axis('off')

total_elapsed_time = time.time() - total_start_time
print(f"\n{'='*60}")
print(f"All {n_samples} samples completed in {total_elapsed_time:.1f} seconds ({total_elapsed_time/60:.1f} minutes)")
print(f"Average time per sample: {total_elapsed_time/n_samples:.1f} seconds")

gpu_str = " [GPU]" if USE_GPU else " [CPU]"
fig.suptitle(f'UMAP of {n_samples} Samples (split: {SPLIT_NAME}){gpu_str}\nEach sample shows source (control) vs target (treated) cells', 
             fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = f'umap_panel_{n_samples}samples_{SPLIT_NAME}.png'
plt.savefig(os.path.join(base_dir, output_file), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to {output_file}")

