import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
import pandas as pd

class TCRDataset(Dataset):
    def __init__(
            self,
            seed: Optional[int] = None,
            set_size: int = 32,
            data_dir: str = 'tcr_dataset',
            **kwargs,  # absorb any extra keyword args without failing
            ):

        if seed is not None:
            np.random.seed(seed)
        
        self.set_size = set_size
        
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            base_dir = hydra.utils.get_original_cwd()
        else:
            base_dir = os.getcwd()
        
        # Load repertoire index
        repertoire_index_path = os.path.join(base_dir, data_dir, 'repertoire_index.tsv')
        repertoire_index = pd.read_csv(repertoire_index_path, sep='\t')
        
        # Initialize data and metadata lists
        self.data = []
        self.metadata = []
        
        # Load data for each repertoire
        for _, row in repertoire_index.iterrows():
            subject_id = row['subject_id']
            timepoint = row['timepoint']
            
            # Construct path to full_data_unit.tsv
            data_unit_path = os.path.join(
                base_dir, 
                data_dir, 
                'tcr_data', 
                f'subject={subject_id}', 
                f'time={timepoint}', 
                'full_data_unit.tsv'
            )
            
            # Load the data unit and extract sequences
            data_unit = pd.read_csv(data_unit_path, sep='\t')
            sequences = data_unit['sequence'].tolist()
            
            # Add to data and metadata
            self.data.append(sequences)
            self.metadata.append((subject_id, timepoint))
        
        # Build next timepoint map for train_predictor_bool
        self.next_timepoint_map = {}
        
        # Group timepoints by patient
        patient_timepoints = {}
        for subject_id, timepoint in self.metadata:
            if subject_id not in patient_timepoints:
                patient_timepoints[subject_id] = []
            patient_timepoints[subject_id].append(timepoint)
        
        # For each patient, sort timepoints and create mapping
        for subject_id, timepoints in patient_timepoints.items():
            sorted_timepoints = sorted(timepoints)
            self.next_timepoint_map[subject_id] = {}
            for i in range(len(sorted_timepoints) - 1):
                current_time = sorted_timepoints[i]
                next_time = sorted_timepoints[i + 1]
                self.next_timepoint_map[subject_id][current_time] = next_time
    
    
    
    def get_train_predictor_bool(self, source_metadata, target_metadata):
        source_subject, source_time = source_metadata
        target_subject, target_time = target_metadata
        
        # Must be from the same patient
        if source_subject != target_subject:
            return False
        
        # Check if target is the next consecutive timepoint for this patient
        if source_subject in self.next_timepoint_map:
            expected_next_time = self.next_timepoint_map[source_subject].get(source_time)
            return expected_next_time == target_time
        
        return False
                
    def __len__(self):
        n = len(self.metadata)
        return n * (n - 1)
    
    # TODO: need to implement distance function in dataset class for Nic's ot coupling I think.
    def __getitem__(self, idx):
        n = len(self.metadata)
        i = idx // (n - 1)
        j = idx % (n - 1)
        if j >= i:
            j += 1
        source_idx, target_idx = i, j
        
        source_samples = self.data[source_idx]
        target_samples = self.data[target_idx]
        
        source_metadata = self.metadata[source_idx]
        target_metadata = self.metadata[target_idx]
        
        subset_indices = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
        
        # TODO: hmmm, need to check again how to incoorporate this with Nic's ot, because this is not the right way.
        source_samples = [source_samples[subset_idx] for subset_idx in subset_indices]
        target_samples = [target_samples[subset_idx] for subset_idx in subset_indices]
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            'source_metadata': source_metadata,
            'target_metadata': target_metadata,
            'source_idx': source_idx,
            'target_idx': target_idx,                
            'train_predictor_bool': self.get_train_predictor_bool(source_idx, target_idx),
        }