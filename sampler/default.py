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
            # only pair t and t+N with N > 0
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                source_idx = item['source_idx']
                target_idx = item['target_idx']
                if target_idx < source_idx:
                    weights[idx] = 0.0
                    
        if self.selective_pairing_mode == "single_step":
            # only pair t and t+1
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                source_idx = item['source_idx']
                target_idx = item['target_idx']
                if source_idx != target_idx - 1:
                    weights[idx] = 0.0  
                           
        return weights