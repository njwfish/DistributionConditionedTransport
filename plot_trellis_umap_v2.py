"""
Script to visualize the trellis dataset samples using UMAP.
Uses the trellis_dataset class to instantiate the dataset and access self.samples.
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
NUM_SAMPLES = 5  # Number of samples (sets) to use for UMAP
PATIENT_MODE = "any"  # Options: "same" (all from one patient), "different" (all from different patients), "any" (random)
PATIENT_ID = None  # If PATIENT_MODE="same", optionally specify which patient (e.g., "21"), or None for random
MAX_CELLS_PLOT = 1000000  # Maximum cells to plot per source/target (for visualization only, UMAP uses all data)
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

# Get unique patients
all_patients = list(set(s[5] for s in samples))
print(f"Unique patients: {all_patients}")

# Select samples based on PATIENT_MODE
np.random.seed(40)

if PATIENT_MODE == "same":
    # All samples from the same patient
    if PATIENT_ID is not None:
        target_patient = str(PATIENT_ID)
    else:
        target_patient = np.random.choice(all_patients)
    
    patient_samples = [s for s in samples if s[5] == target_patient]
    if len(patient_samples) < NUM_SAMPLES:
        print(f"Warning: Patient {target_patient} only has {len(patient_samples)} samples (requested {NUM_SAMPLES})")
    selected_indices = np.random.choice(len(patient_samples), min(NUM_SAMPLES, len(patient_samples)), replace=False)
    selected_samples = [patient_samples[i] for i in selected_indices]
    print(f"Using {len(selected_samples)} samples from patient {target_patient}")

elif PATIENT_MODE == "different":
    # All samples from different patients
    selected_samples = []
    available_patients = all_patients.copy()
    np.random.shuffle(available_patients)
    
    for patient in available_patients:
        if len(selected_samples) >= NUM_SAMPLES:
            break
        patient_samples = [s for s in samples if s[5] == patient]
        if patient_samples:
            selected_samples.append(patient_samples[np.random.randint(len(patient_samples))])
    
    if len(selected_samples) < NUM_SAMPLES:
        print(f"Warning: Only {len(available_patients)} unique patients available (requested {NUM_SAMPLES})")
    print(f"Using {len(selected_samples)} samples from {len(selected_samples)} different patients")

else:  # PATIENT_MODE == "any"
    # Random selection (original behavior)
    selected_indices = np.random.choice(len(samples), min(NUM_SAMPLES, len(samples)), replace=False)
    selected_samples = [samples[i] for i in selected_indices]
    print(f"Using {len(selected_samples)} samples (random selection)")

# Collect all x0 and x1 data with sample indices and source/target labels
all_data = []
sample_indices = []
is_target = []  # False for x0 (source), True for x1 (target)

for idx, (culture, x0, x1, cond_cell, cond_treat, patient) in enumerate(selected_samples):
    all_data.append(x0)
    sample_indices.extend([idx] * x0.shape[0])
    is_target.extend([False] * x0.shape[0])
    
    all_data.append(x1)
    sample_indices.extend([idx] * x1.shape[0])
    is_target.extend([True] * x1.shape[0])
    
    print(f"  Sample {idx}: {culture}, patient={patient}, source={x0.shape[0]} cells, target={x1.shape[0]} cells")

all_data = np.vstack(all_data)
sample_indices = np.array(sample_indices)
is_target = np.array(is_target)

print(f"Total cells: {all_data.shape[0]} (source: {(~is_target).sum()}, target: {is_target.sum()})")
print(f"Features per cell: {all_data.shape[1]}")

# Run UMAP with progress tracking
print("Running UMAP...")
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

# Plot with distinct colors for each sample and different markers for source/target
plt.figure(figsize=(14, 10))

# Use tab10 colormap for distinct colors (enough for 5 samples)
cmap = plt.cm.tab10
colors = [cmap(i) for i in range(NUM_SAMPLES)]

# Plot each sample with its own color, distinguishing source vs target
np.random.seed(43)  # Different seed for plotting subsample
for idx in range(len(selected_samples)):
    culture, _, _, cond_cell, cond_treat, patient = selected_samples[idx]
    
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
    
    # Get source (x0) points for this sample
    mask_source = (sample_indices == idx) & (~is_target)
    indices_source = np.where(mask_source)[0]
    
    # Subsample source if needed
    if len(indices_source) > MAX_CELLS_PLOT:
        indices_source = np.random.choice(indices_source, MAX_CELLS_PLOT, replace=False)
    
    # Get target (x1) points for this sample
    mask_target = (sample_indices == idx) & is_target
    indices_target = np.where(mask_target)[0]
    
    # Subsample target if needed
    if len(indices_target) > MAX_CELLS_PLOT:
        indices_target = np.random.choice(indices_target, MAX_CELLS_PLOT, replace=False)
    
    # Create detailed label with all information
    label_info = f'S{idx} (p={patient}, {culture}, {cell_str}, t={treatment_name})'
    
    # Plot source with circles
    plt.scatter(
        embedding[indices_source, 0], 
        embedding[indices_source, 1], 
        c=[colors[idx]], 
        marker='o',
        s=20, 
        alpha=0.3,
        edgecolors='none',
        label=f'{label_info} - Src'
    )
    
    # Plot target with X markers
    plt.scatter(
        embedding[indices_target, 0], 
        embedding[indices_target, 1], 
        c=[colors[idx]], 
        marker='x',
        s=20, 
        alpha=0.5,
        linewidths=1.5,
        label=f'{label_info} - Tgt'
    )
    
    print(f"  Plotted sample {idx}: {len(indices_source)} source cells, {len(indices_target)} target cells")
    print(f"    Culture: {culture}, Cells: {cell_str}, Treatment: {treatment_name}, Patient: {patient}")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.5, fontsize=9)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
gpu_str = " [GPU]" if USE_GPU else " [CPU]"
mode_str = f" [{PATIENT_MODE} patients]"
plt.title(f'UMAP of {NUM_SAMPLES} Samples (split: {SPLIT_NAME}){gpu_str}\n{mode_str} - Source (circles) vs Target (X) - Max {MAX_CELLS_PLOT} cells/type for visualization')
plt.tight_layout()

base_dir = os.path.dirname(os.path.abspath(__file__))
output_file = f'umap_{NUM_SAMPLES}samples_{SPLIT_NAME}_{PATIENT_MODE}_source_target_v2.png'
plt.savefig(os.path.join(base_dir, output_file), dpi=150, bbox_inches='tight')
plt.show()

print(f"Plot saved to {output_file}")

