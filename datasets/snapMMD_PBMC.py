import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp
import os
import hydra

class SnapMMDPBMC(Dataset):
    """Dataset for multivariate normal distributions."""
    
    # TODO: eventually remove data_shape, it is not necessary.
    def __init__(
            self, 
            testing_method: str = "forecast",
            data_dir: str = "data/realdata",
            data_name: str = "processed_pbmc_data_sub500_every_2_until20.npz",
            data_shape: Optional[List[int]] = None,
            seed: Optional[int] = None,
            ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            seed: Random seed for reproducibility
        """
        
        if seed is not None:
            np.random.seed(seed)
        
        self.testing_method = testing_method
        # Use the original working directory before Hydra changed it
        original_cwd = hydra.utils.get_original_cwd()
        
        self.full_dataset = np.load(f"{os.path.join(original_cwd, data_dir, data_name)}")
        self.data = self.full_dataset['Xs']
        self.initial_conditions = self.full_dataset['y0']
        self.time_steps = self.full_dataset['dts']
        self.time_scale = self.full_dataset['time_scale']
        self.N_steps = self.full_dataset['N_steps']
        self.X_scaling = self.full_dataset['X_scaling']
        
        #print("!!!!!!",self.time_steps)
        #print("????",self.X_scaling)   
        #print("????",self.data.shape)
        #print("EEEEEENNNNNDDDDD")

    # TODO: make really really sure that you are not training on the test data.
    def __len__(self):
        # TODO: should you do self.data.shape[0]-1 because the final point is going to be held out to compute the forecasting performance?
        if self.testing_method == "forecast":
            return self.data.shape[0]-1
        else:
            return self.data.shape[0]
    
    #def __getitem__(self, idx):
    #    shape = self.data[idx].shape
    #    return {
    #        'samples': torch.full(shape, idx, dtype=torch.float),
    #        'idx': idx
    #    }
    
    def __getitem__(self, idx):
        if self.testing_method == "forecast":
            return {
                'samples': torch.tensor(self.data[:-1][idx], dtype=torch.float),
                'idx': idx
            }
        else:
            return {
                'samples': torch.tensor(self.data[idx], dtype=torch.float),
            'idx': idx
        }