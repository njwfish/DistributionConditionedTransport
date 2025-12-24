import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
import ot
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
    
    
    
    def get_train_predictor_bool(self, source_idx, target_idx):
        pass        
    def __len__(self):
        pass
    def __getitem__(self, idx):
        pass