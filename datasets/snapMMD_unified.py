import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp
import os
import hydra

class SnapMMDUnified(Dataset):
    """Unified dataset for all SnapMMD datasets (GoM, LV, PBMC, Repressilator)."""
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'GoM': {
            'data_dir': 'data/realdata',
            'data_name': 'GoM_data.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'LV': {
            'data_dir': 'data/classic',
            'data_name': 'LV_data.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'PBMC': {
            'data_dir': 'data/realdata',
            'data_name': 'processed_pbmc_data_sub500_every_2_until20.npz',
            'data_shape': [30],
            'has_x_scaling': True
        },
        'Repressilator': {
            'data_dir': 'data/classic',
            'data_name': 'Repressilator_data.npz',
            'data_shape': [3],
            'has_x_scaling': False
        }
    }
    
    def __init__(
            self,
            dataset_name: str,
            testing_method: str = "forecast",
            seed: Optional[int] = None,
            set_size: int = 32,
            **kwargs,  # absorb any extra keyword args without failing
            ):
        """
        Args:
            dataset_name: Name of the dataset to load ('GoM', 'LV', 'PBMC', 'Repressilator')
            testing_method: Testing method to use
            seed: Random seed for reproducibility
            set_size: Number of samples per parameter set
            **kwargs: Absorb additional keyword arguments that may be supplied by Hydra. These are ignored to ensure
                robustness when the class is referenced in Hydra configs for purposes other than direct instantiation.
        """
        if dataset_name not in self.DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available datasets: {list(self.DATASET_CONFIGS.keys())}")
        
        self.dataset_name = dataset_name
        self.config = self.DATASET_CONFIGS[dataset_name]
        
        if seed is not None:
            np.random.seed(seed)
        
        self.testing_method = testing_method
        self.set_size = set_size
        
        # Use the original working directory before Hydra changed it
        original_cwd = hydra.utils.get_original_cwd()
        data_path = os.path.join(original_cwd, self.config['data_dir'], self.config['data_name'])
        self.full_dataset = np.load(data_path)
        
        # NOTE: indexing [:-1] is to remove the last time point, which is the target for forecasting benchmarks.
        self.data = self.full_dataset['Xs'][:-1]
        self.initial_conditions = self.full_dataset['y0']
        self.time_steps = self.full_dataset['dts']
        self.time_scale = self.full_dataset['time_scale']
        self.N_steps = self.full_dataset['N_steps']
        
        # Handle PBMC-specific X_scaling field
        if self.config['has_x_scaling']:
            self.X_scaling = self.full_dataset['X_scaling']
        
        self.index_pairs = np.array([(i, j) for i in range(self.data.shape[0]) for j in range(self.data.shape[0]) if i != j])

    # TODO: this is for everything below: for now, I am just making it such that it takes in the entire population every time. Might want to modify to just sample both time-points and the individual unit.
    
    # TODO: make really really sure that you are not training on the test data.
    def __len__(self):
        # TODO: should you do self.data.shape[0]-1 because the final point is going to be held out to compute the forecasting performance?
        if self.testing_method == "forecast":
            num = self.data.shape[0]
            return num**2 - num

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
    
    # TODO: remove this? I don't remember putting it in.
    @property
    def data_shape(self):
        """Get the data shape for this dataset."""
        return self.config['data_shape'] 