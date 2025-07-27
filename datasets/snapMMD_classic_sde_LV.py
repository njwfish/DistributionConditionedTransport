import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp
import os
import hydra

class SnapMMDClassicSDELV(Dataset):
    """Dataset for multivariate normal distributions."""
    
    # TODO: eventually remove data_shape, it is not necessary.
    def __init__(
            self, 
            testing_method: str = "forecast",
            data_dir: str = "data/classic",
            data_name: str = "LV_data.npz",
            data_shape: Optional[List[int]] = None,
            seed: Optional[int] = None,
            bidirectional: bool = False,
            set_size: int = 32,
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
        # Available keys in dataset: ['N_steps', 'Xs', 'y0', 'time_scale', 'dts']
        #print("Available keys in dataset:", list(self.full_dataset.keys()))
        # NOTE: indexing [:-1] is to remove the last time point, which is the target for forecasting benchmarks.
        self.data = self.full_dataset['Xs'][:-1]
        self.initial_conditions = self.full_dataset['y0']
        self.time_steps = self.full_dataset['dts']
        self.time_scale = self.full_dataset['time_scale']
        self.N_steps = self.full_dataset['N_steps']
        self.set_size = set_size
        self.bidirectional = bidirectional
        
        if self.bidirectional:
            self.index_pairs = np.array([(i, j) for i in range(self.data.shape[0]) for j in range(self.data.shape[0]) if i != j])
        else:
            self.index_pairs = np.array([(i, j) for i in range(self.data.shape[0] - 1) for j in range(i+1, self.data.shape[0])])
        
        print("_data_shape:", self.data.shape)
        print("data:", self.data)
        print("initial_conditions:", self.initial_conditions)
        print("time_steps:", self.time_steps)
        print("time_scale:", self.time_scale)
        print("N_steps:", self.N_steps)

    # TODO: this is for everything below: for now, I am just making it such that it takes in the entire population every time. Might want to modify to just sample both time-points and the individual unit.
    
    # TODO: make really really sure that you are not training on the test data.
    def __len__(self):
        # TODO: should you do self.data.shape[0]-1 because the final point is going to be held out to compute the forecasting performance?
        if self.testing_method == "forecast":
            num = self.data.shape[0]
            if self.bidirectional:
                return num**2 - num
            else:
                # TODO: make sure this is correct.
                return ((num-1)*num) // 2
        # TODO: implement interpolation as an alternative task to forecasting.
        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")

    
    def __getitem__(self, idx):
        if self.testing_method == "forecast":
            source_idx, target_idx = self.index_pairs[idx]
            source_samples = torch.tensor(self.data[source_idx], dtype=torch.float)
            target_samples = torch.tensor(self.data[target_idx], dtype=torch.float)
            
            
            subset_indices = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
            
            source_samples = source_samples[subset_indices]
            target_samples = target_samples[subset_indices]
            
            # TODO: make sure shape here will be consistent with what it was before.
            return {
                'source_samples': source_samples,
                'target_samples': target_samples,
                'dt': target_idx - source_idx,
                'idx': idx
            }
        # TODO: implement interpolation as an alternative task to forecasting.
        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")