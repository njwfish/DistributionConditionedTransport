"""
Script to visualize a single sample from the trellis dataset using UMAP.
Uses the trellis_dataset class to instantiate the dataset and visualizes
the source and target cells from one sample.
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
SAMPLE_INDEX = None  # Which sample to visualize (None = random selection)
RANDOM_SEED = 42  # Random seed for sample selection
MAX_CELLS_PLOT = 1000000  # Maximum cells to plot per source/target
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

# Select a single sample
np.random.seed(RANDOM_SEED)

if SAMPLE_INDEX is not None and 0 <= SAMPLE_INDEX < len(samples):
    selected_idx = SAMPLE_INDEX
    print(f"Using specified sample index: {selected_idx}")
else:
    selected_idx = np.random.randint(len(samples))
    print(f"Randomly selected sample index: {selected_idx}")

# Get the sample data
culture, x0, x1, cond_cell, cond_treat, patient = samples[selected_idx]

print(f"\nSample {selected_idx} information:")
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

print(f"  Cell types: {cell_str}")
print(f"  Treatment: {treatment_name}")

# Prepare data for UMAP
all_data = np.vstack([x0, x1])
is_target = np.array([False] * x0.shape[0] + [True] * x1.shape[0])

print(f"\nTotal cells for UMAP: {all_data.shape[0]} (source: {(~is_target).sum()}, target: {is_target.sum()})")
print(f"Features per cell: {all_data.shape[1]}")

# Run UMAP with progress tracking
print("\nRunning UMAP...")
start_time = time.time()

if USE_GPU:
    # cuML UMAP with verbose output (levels 1-7, higher = more verbose)
    reducer = UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        random_state=42,
        verbose=5
    )
else:
    # CPU UMAP with verbose output
    reducer = UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        random_state=42,
        verbose=True,
        low_memory=True  # Helps with large datasets on CPU
    )

embedding = reducer.fit_transform(all_data)

elapsed_time = time.time() - start_time
print(f"\nUMAP completed in {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")

# Plot source vs target
plt.figure(figsize=(12, 10))

# Get indices for source and target
indices_source = np.where(~is_target)[0]
indices_target = np.where(is_target)[0]

# Subsample if needed
np.random.seed(43)
if len(indices_source) > MAX_CELLS_PLOT:
    indices_source = np.random.choice(indices_source, MAX_CELLS_PLOT, replace=False)
    print(f"Subsampled source to {MAX_CELLS_PLOT} cells for plotting")

if len(indices_target) > MAX_CELLS_PLOT:
    indices_target = np.random.choice(indices_target, MAX_CELLS_PLOT, replace=False)
    print(f"Subsampled target to {MAX_CELLS_PLOT} cells for plotting")

# Plot source with blue circles
plt.scatter(
    embedding[indices_source, 0], 
    embedding[indices_source, 1], 
    c='blue', 
    marker='o',
    s=30, 
    alpha=0.4,
    edgecolors='none',
    label=f'Source (control, n={len(indices_source)})'
)

# Plot target with red X markers
plt.scatter(
    embedding[indices_target, 0], 
    embedding[indices_target, 1], 
    c='red', 
    marker='x',
    s=30, 
    alpha=0.6,
    linewidths=1.5,
    label=f'Target (treated, n={len(indices_target)})'
)

print(f"\nPlotted {len(indices_source)} source cells and {len(indices_target)} target cells")

plt.legend(loc='best', markerscale=1.5, fontsize=11)
plt.xlabel('UMAP 1', fontsize=12)
plt.ylabel('UMAP 2', fontsize=12)
gpu_str = " [GPU]" if USE_GPU else " [CPU]"
sample_info = f"Sample {selected_idx}: p={patient}, {culture}, {cell_str}, treatment={treatment_name}"
plt.title(f'UMAP of Single Sample (split: {SPLIT_NAME}){gpu_str}\n{sample_info}', fontsize=13)
plt.tight_layout()

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = f'umap_single_sample_{selected_idx}_{SPLIT_NAME}_p{patient}_{treatment_name}.png'
plt.savefig(os.path.join(base_dir, output_file), dpi=150, bbox_inches='tight')
plt.show()

print(f"\nPlot saved to {output_file}")

