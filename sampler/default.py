import torch
import math
from torch.utils.data import WeightedRandomSampler
from typing import Optional, Union, List, Tuple

class Sampler(WeightedRandomSampler):

    def __init__(
        self, 
        dataset=None,
        num_samples: Optional[int] = None,
        selective_pairing_mode: Optional[str] = None,
    ):


        self.dataset = dataset
        self.selective_pairing_mode = selective_pairing_mode
        
        # Compute weights
        weights = self._compute_weights()
        
        # Set default num_samples if not provided
        if num_samples is None:
            num_samples = len(dataset)
            
        # Initialize parent WeightedRandomSampler
        super().__init__(
            weights=weights,
            num_samples=num_samples,
        )

    def _compute_weights(self) -> torch.Tensor:
        """Compute weights for each sample based on the sampling mode."""
        weights = torch.ones(len(self.dataset), dtype=torch.float)
                                      
        if self.selective_pairing_mode == "unidirectional":
            # set all weights with d < 0 to 0
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                d = item['d']
                if d < 0:
                    weights[idx] = 0.0
                    
        if self.selective_pairing_mode == "single_step":
            # Compute valid indices mathematically instead of iterating through entire dataset
            # This is MUCH faster for large datasets where len(dataset) = total_x_populations^2
            #
            # train_predictor_bool is True when:
            #   - source is x0 from train (source_flat_idx is even, in [0, 2*n_train))
            #   - target is x1 from same sample (target_flat_idx = source_flat_idx + 1)
            #
            # Given idx = source_flat_idx * N + target_flat_idx, valid indices are:
            #   idx = (2*k) * N + (2*k + 1) for k in range(n_train)
            #   idx = 2*k * (N + 1) + 1
            
            # Start with all weights = 0, then set valid indices to 1
            weights.zero_()
            
            if hasattr(self.dataset, 'n_train') and hasattr(self.dataset, 'total_x_populations'):
                n_train = self.dataset.n_train
                N = self.dataset.total_x_populations
                
                # Compute valid indices: for each train sample k, the true pair is at
                # idx = source_flat_idx * N + target_flat_idx
                # where source_flat_idx = 2*k (x0) and target_flat_idx = 2*k + 1 (x1)
                for k in range(n_train):
                    source_flat_idx = 2 * k  # x0 of train sample k
                    target_flat_idx = 2 * k + 1  # x1 of train sample k
                    idx = source_flat_idx * N + target_flat_idx
                    weights[idx] = 1.0
            else:
                # Fallback to slow method if dataset doesn't have required attributes
                for idx in range(len(self.dataset)):
                    item = self.dataset[idx]
                    if item['train_predictor_bool']:
                        weights[idx] = 1.0

        return weights