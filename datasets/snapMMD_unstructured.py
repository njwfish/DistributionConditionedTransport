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

class SnapMMDUnified(Dataset):
    """Unified dataset for all SnapMMD datasets (GoM, LV, PBMC, Repressilator)."""
    
    # Dataset-specific configurations
    DATASET_CONFIGS = {
        'GoM': {
            'data_dir': 'data/realdata',
            'data_name': 'GoM_data.npz',
            'unstructured_data_name': 'GoM_data_interp_val.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'LV': {
            'data_dir': 'data/classic',
            'data_name': 'LV_data.npz',
            'unstructured_data_name': 'LV_data_interp_val.npz',
            'data_shape': [2],
            'has_x_scaling': False
        },
        'PBMC': {
            'data_dir': 'data/realdata',
            'data_name': 'processed_pbmc_data_sub500_every_2_until20.npz',
            'unstructured_data_name': 'processed_pbmc_data_sub500_every_2_until20_interp_val.npz',
            'data_shape': [30],
            'has_x_scaling': True
        },
        'Repressilator': {
            'data_dir': 'data/classic',
            'data_name': 'Repressilator_data.npz',
            'unstructured_data_name': 'Repressilator_data_interp_val.npz',
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
            ot_coupling: bool = False,
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
        self.ot_coupling = ot_coupling
        # TODO: hmmm, maybe I interpreted set_size slightly wrong. Is it supposed to be a subset of the whole population at a given time point or just the size of the population?
        self.set_size = set_size
        
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            base_dir = hydra.utils.get_original_cwd()
        else:
            base_dir = os.getcwd()
        data_path = os.path.join(base_dir, self.config['data_dir'], self.config['data_name'])
        self.full_dataset = np.load(data_path)
        
        # NOTE: indexing [:-1] is to remove the last time point, which is the target for forecasting benchmarks.
        self.structured_data = self.full_dataset['Xs'][:-1]

        unstructured_data_path = os.path.join(base_dir, self.config['data_dir'], self.config['unstructured_data_name'])
        unstructured_data = np.load(unstructured_data_path)
        self.unstructured_data = unstructured_data['Xs']
        
        self.data = np.concatenate([self.structured_data, self.unstructured_data], axis=0)
        
        self.structured_data_size = self.structured_data.shape[0] # NOTE: this is the number of time points in the structured data.
    
    def get_train_predictor_bool(self, source_idx, target_idx):
        return (target_idx - source_idx) == 1 and source_idx < self.structured_data_size and target_idx < self.structured_data_size
        
    def __len__(self):
        if self.testing_method == "forecast":
            num = self.data.shape[0]
            return num**2

        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")

    
    def __getitem__(self, idx):
        if self.testing_method == "forecast":
            n = self.data.shape[0]
            i = idx // n
            j = idx % n

            source_idx, target_idx = i, j
            
            source_samples = torch.tensor(self.data[source_idx], dtype=torch.float)
            target_samples = torch.tensor(self.data[target_idx], dtype=torch.float)
            
            subset_indices = np.random.choice(source_samples.shape[0], size=self.set_size, replace=False)
            
            source_samples = source_samples[subset_indices]
            target_samples = target_samples[subset_indices]

            if self.ot_coupling:
                # NOTE: converted to numpy to avoid CUDA issues.  
                # Compute OT coupling using POT with NumPy backend to avoid CUDA init in DataLoader workers
                source_np = source_samples.cpu().numpy()
                target_np = target_samples.cpu().numpy()
                cost = ot.dist(source_np, target_np, metric="sqeuclidean")
                G = ot.emd([], [], cost)
                # G = ot.sinkhorn([], [], cost, 1e-1)
                # G = ot.bregman.empirical_sinkhorn(src, tgt, 1e-1)

                # use all elements from ot plan
                # TODO: is random shuffling needed here?
                choices = np.arange(G.shape[0] * G.shape[1])
                idx0, idx1 = np.divmod(choices, G.shape[1])

                # OT paired samples
                source_samples = source_samples[idx0]
                target_samples = target_samples[idx1]
            
            
            return {
                'source_samples': source_samples,
                'target_samples': target_samples,
                'source_idx': source_idx,
                'target_idx': target_idx,                
                'train_predictor_bool': self.get_train_predictor_bool(source_idx, target_idx),
            }

        else:
            raise NotImplementedError(f"Testing method '{self.testing_method}' not implemented")
    
    # TODO: remove this? I don't remember putting it in.
    @property
    def data_shape(self):
        """Get the data shape for this dataset."""
        return self.config['data_shape'] 