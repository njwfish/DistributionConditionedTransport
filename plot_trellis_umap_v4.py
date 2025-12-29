"""
Script to analyze pRB feature differences between source and target cells
for specific patient-treatment combinations in the trellis dataset.
"""

import numpy as np
import sys
import os

# Add the current directory to the path to import the dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets.mfm_trellis import trellis_dataset

# Configuration
SPLIT_NAME = 'pdo27'  # Options: "replicas-1", "replicas-2", "pdo21", "pdo27", "pdo75"
CONTROL = set(["DMSO", "AH", "H2O"])
TREATMENT = ["O", "S", "VS", "L", "V", "F", "C", "SF", "CS", "CF", "CSF"]
CELL_TYPE = ["PDOs", "Fibs"]

# Feature list to find pRB index
FEATURES = ['pHH3', 'RFP', 'mCHERRY', 'Vimentin', 'EpCAM', 'CK18', 'Pan_CK', 
            'GFP', 'IdU', 'pPDK1', 'cCaspase_3', 'Geminin', 'pMEK1_2', 'pNDRG',
            'pMKK4_SEK1', 'pBTK', 'pSRC', 'p4EBP1', 'pRB', 'pAKT308', 'pCREB',
            'pSMAD1_5_9', 'pAKT473', 'pNF_kB', 'pMKK3_MKK6', 'pP38', 'pMAPKAPK',
            'pAMPKa', 'pBAD', 'pHistone_H2A', 'p90RSK', 'pP120_catenin',
            'Beta_catenin_active', 'pGSK', 'pERK1_2', 'pSMAD2_3', 'PLK', 'CHGA',
            'pDNAPK', 'pS6', 'CD90', 'cPARP', 'pCHK1']

# Find pRB index
PRB_INDEX = FEATURES.index('pRB')
print(f"pRB feature is at index: {PRB_INDEX}")

# Filtering criteria
FILTER_CRITERIA = [
    {'patient': 'pdo-99', 'treatment': 'O'},
    {'patient': 'pdo-21', 'treatment': 'S'}
]

# Instantiate the dataset
print(f"\nLoading dataset with split_name={SPLIT_NAME}...")
dataset = trellis_dataset(
    control=CONTROL,
    treatment=TREATMENT,
    culture=["PDO", "PDOF", "F"],
    cell_type=CELL_TYPE,
    split_name=SPLIT_NAME,
    set_size=32,
    seed=0
)

# Access the samples from the dataset
samples = dataset.samples
print(f"Total samples available: {len(samples)}")

# Analyze samples for each filter criterion
for criterion in FILTER_CRITERIA:
    target_patient = criterion['patient']
    target_treatment = criterion['treatment']
    
    print(f"\n{'='*70}")
    print(f"Filtering for Patient {target_patient} with Treatment {target_treatment}")
    print(f"{'='*70}")
    
    # Get treatment index
    treatment_idx = TREATMENT.index(target_treatment)
    
    # Filter samples
    matching_samples = []
    for idx, (culture, x0, x1, cond_cell, cond_treat, patient) in enumerate(samples):
        # Check if patient matches
        if patient != target_patient:
            continue
        
        # Check if treatment matches
        treat_idx = np.argmax(cond_treat[0])
        if treat_idx != treatment_idx:
            continue
        
        matching_samples.append((idx, culture, x0, x1, cond_cell, cond_treat, patient))
    
    print(f"Found {len(matching_samples)} matching samples")
    
    if len(matching_samples) == 0:
        print(f"  No samples found for Patient {target_patient} with Treatment {target_treatment}")
        continue
    
    # Analyze each matching sample
    for sample_idx, culture, x0, x1, cond_cell, cond_treat, patient in matching_samples:
        # Decode cell types
        cell_types_present = []
        if cond_cell[:, 0].sum() > 0:
            cell_types_present.append(CELL_TYPE[0])
        if cond_cell[:, 1].sum() > 0:
            cell_types_present.append(CELL_TYPE[1])
        cell_str = "+".join(cell_types_present) if cell_types_present else "None"
        
        # Calculate mean and std pRB for source (x0) and target (x1)
        mean_prb_source = np.mean(x0[:, PRB_INDEX])
        std_prb_source = np.std(x0[:, PRB_INDEX])
        mean_prb_target = np.mean(x1[:, PRB_INDEX])
        std_prb_target = np.std(x1[:, PRB_INDEX])
        
        # Calculate difference (target - source)
        prb_difference = mean_prb_target - mean_prb_source
        
        print(f"\n  Sample {sample_idx}:")
        print(f"    Culture: {culture}")
        print(f"    Cell types: {cell_str}")
        print(f"    Patient: {patient}")
        print(f"    Treatment: {target_treatment}")
        print(f"    Source cells: {x0.shape[0]}")
        print(f"    Target cells: {x1.shape[0]}")
        print(f"    Mean pRB (source): {mean_prb_source:.6f} ± {std_prb_source:.6f}")
        print(f"    Mean pRB (target): {mean_prb_target:.6f} ± {std_prb_target:.6f}")
        print(f"    Difference (target - source): {prb_difference:.6f}")

# Summary statistics across all criteria
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")

all_differences = []
for criterion in FILTER_CRITERIA:
    target_patient = criterion['patient']
    target_treatment = criterion['treatment']
    treatment_idx = TREATMENT.index(target_treatment)
    
    criterion_differences = []
    for idx, (culture, x0, x1, cond_cell, cond_treat, patient) in enumerate(samples):
        if patient != target_patient:
            continue
        treat_idx = np.argmax(cond_treat[0])
        if treat_idx != treatment_idx:
            continue
        
        mean_prb_source = np.mean(x0[:, PRB_INDEX])
        mean_prb_target = np.mean(x1[:, PRB_INDEX])
        prb_difference = mean_prb_target - mean_prb_source
        criterion_differences.append(prb_difference)
        all_differences.append(prb_difference)
    
    if len(criterion_differences) > 0:
        print(f"\nPatient {target_patient}, Treatment {target_treatment}:")
        print(f"  Number of samples: {len(criterion_differences)}")
        print(f"  Mean pRB difference: {np.mean(criterion_differences):.6f}")
        print(f"  Std pRB difference: {np.std(criterion_differences):.6f}")
        print(f"  Min pRB difference: {np.min(criterion_differences):.6f}")
        print(f"  Max pRB difference: {np.max(criterion_differences):.6f}")

if len(all_differences) > 0:
    print(f"\nOverall across all filtered samples:")
    print(f"  Total samples: {len(all_differences)}")
    print(f"  Mean pRB difference: {np.mean(all_differences):.6f}")
    print(f"  Std pRB difference: {np.std(all_differences):.6f}")

