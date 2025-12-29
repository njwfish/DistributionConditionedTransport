"""
Script to visualize the trellis dataset samples using UMAP.
Creates self.samples in the same way as trellis_dataset and plots a UMAP 
colored by sample index.
"""

import numpy as np
import torch
import pickle
import os
import time
import matplotlib.pyplot as plt

# Try to use GPU UMAP (cuML), fall back to CPU if not available
try:
    from cuml.manifold import UMAP
    USE_GPU = True
    print("Using GPU-accelerated UMAP (cuML)")
except ImportError:
    from umap import UMAP
    USE_GPU = False
    print("cuML not available, using CPU UMAP")

# Configuration (same defaults as trellis_dataset)
SPLIT_NAME = 'pdo21'  # Options: "replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"
NUM_SAMPLES = 5  # Number of samples (sets) to use for UMAP
PATIENT_MODE = "any"  # Options: "same" (all from one patient), "different" (all from different patients), "any" (random)
PATIENT_ID = None  # If PATIENT_MODE="same", optionally specify which patient (e.g., "21"), or None for random
MAX_CELLS_PLOT = 1000  # Maximum cells to plot per source/target (for visualization only, UMAP uses all data)
CONTROL = set(["DMSO", "AH", "H2O"])
TREATMENT = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]
CELL_TYPE = ["PDOs", "Fibs"]

# Set paths based on split_name
if SPLIT_NAME == "replicas-1":
    split_source = "organoid_data_preprocessed/replica_holdout/replica_1_holdout/data_splits_replicas_1.pickle"
    data_path = "organoid_data_preprocessed/replica_holdout/replica_1_holdout/trellis_replicas_1_normalized.npy"
elif SPLIT_NAME == "replicas-2":
    split_source = "organoid_data_preprocessed/replica_holdout/replica_2_holdout/data_splits_replicas_2.pickle"
    data_path = "organoid_data_preprocessed/replica_holdout/replica_2_holdout/trellis_replicas_2_normalized.npy"
elif SPLIT_NAME == "pdo21":
    split_source = "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo21.pickle"
    data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo21_normalized.npy"
elif SPLIT_NAME == "pdo27":
    split_source = "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo27.pickle"
    data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo27_normalized.npy"
elif SPLIT_NAME == "pdo75":
    split_source = "organoid_data_preprocessed/patient_holdout/split_patient_test_pdo75.pickle"
    data_path = "organoid_data_preprocessed/patient_holdout/trellis_patients_pdo75_normalized.npy"

# Load data
base_dir = os.path.dirname(os.path.abspath(__file__))
split_path = os.path.join(base_dir, split_source)
data_path = os.path.join(base_dir, data_path)

with open(split_path, "rb") as handle:
    data_splits = pickle.load(handle)

data = np.load(data_path)[:, :-1]  # Shape: (total_cells, 43)
split = data_splits["train"]

# Filter out entries with empty elements
def has_empty_element(nested_dict):
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            if not value:
                return True
            if has_empty_element(value):
                return True
    return False

split = [ls for ls in split if not has_empty_element(ls)]

# Build samples (same logic as select_experiments)
samples = []

for i in range(len(split)):
    if SPLIT_NAME in ["replicas-1", "replicas-2"]:
        exp = split[i]
        pdo_num = -1
    else:
        exp_patient = split[i]
        pdo_num = list(exp_patient.keys())[0]
        exp = exp_patient[pdo_num]

    x0_treatment = list(set(exp.keys()).intersection(CONTROL))[0]
    treatkeys = [key for key in exp.keys() if key not in CONTROL]

    for t in treatkeys:
        concentration = list(exp[t].keys())
        max_conc = str(max(map(int, concentration)))
        cultures_keys = list(exp[t][max_conc].keys())

        for culture in cultures_keys:
            x0_pdos_idx, x1_pdos_idx, x0_fibs_idx, x1_fibs_idx = [], [], [], []

            if culture in ["PDOF", "PDO"]:
                x0_pdos_idx = exp[x0_treatment]["0"][culture][CELL_TYPE[0]].copy().tolist()
                x1_pdos_idx = exp[t][max_conc][culture][CELL_TYPE[0]].copy().tolist()

            if culture in ["PDOF", "F"]:
                x0_fibs_idx = exp[x0_treatment]["0"][culture][CELL_TYPE[1]].copy().tolist()
                x1_fibs_idx = exp[t][max_conc][culture][CELL_TYPE[1]].copy().tolist()

            x0_idx = x0_pdos_idx + x0_fibs_idx
            x1_idx = x1_pdos_idx + x1_fibs_idx

            x0 = np.array(data[x0_idx])
            x1 = np.array(data[x1_idx])

            # Cell type one-hot encoding
            x0_cell_pdos_idx = range(0, len(x0_pdos_idx))
            x0_cell_fibs_idx = range(len(x0_pdos_idx), len(x0_idx))
            cond_cell = np.zeros((x0.shape[0], len(CELL_TYPE)))
            cond_cell[list(x0_cell_pdos_idx), 0] = 1
            cond_cell[list(x0_cell_fibs_idx), 1] = 1

            # Treatment one-hot encoding
            treat_idx = TREATMENT.index(t)
            cond_treat = torch.nn.functional.one_hot(
                torch.tensor(treat_idx).long(), num_classes=len(TREATMENT)
            )
            cond_treat = cond_treat.expand(x0.shape[0], -1).detach().numpy()

            samples.append((culture, x0, x1, cond_cell, cond_treat, str(pdo_num)))

print(f"Total samples available: {len(samples)}")

# Get unique patients
all_patients = list(set(s[5] for s in samples))
print(f"Unique patients: {all_patients}")

# Select samples based on PATIENT_MODE
np.random.seed(42)

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
    culture, _, _, _, _, patient = selected_samples[idx]
    
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
    
    # Plot source with circles
    plt.scatter(
        embedding[indices_source, 0], 
        embedding[indices_source, 1], 
        c=[colors[idx]], 
        marker='o',
        s=20, 
        alpha=0.3,
        edgecolors='none',
        label=f'Sample {idx} (p={patient}) - Source'
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
        label=f'Sample {idx} (p={patient}) - Target'
    )
    
    print(f"  Plotted sample {idx}: {len(indices_source)} source cells, {len(indices_target)} target cells")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=1.5, fontsize=9)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
gpu_str = " [GPU]" if USE_GPU else " [CPU]"
mode_str = f" [{PATIENT_MODE} patients]"
plt.title(f'UMAP of {NUM_SAMPLES} Samples (split: {SPLIT_NAME}){gpu_str}\n{mode_str} - Source (circles) vs Target (X) - Max {MAX_CELLS_PLOT} cells/type for visualization')
plt.tight_layout()

output_file = f'umap_{NUM_SAMPLES}samples_{SPLIT_NAME}_{PATIENT_MODE}_source_target.png'
plt.savefig(os.path.join(base_dir, output_file), dpi=150, bbox_inches='tight')
plt.show()

print(f"Plot saved to {output_file}")
