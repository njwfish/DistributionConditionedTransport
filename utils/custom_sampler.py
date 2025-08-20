import torch
import math
from torch.utils.data import WeightedRandomSampler
from typing import Optional, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)


class CustomWeightedSampler(WeightedRandomSampler):
    """
    Custom weighted sampler that supports different sampling modes based on 'dt' values in dataset items.
    
    Supported weight modes:
    - "uniform": Equal weights for all samples
    - "exponential": Weight = exp(-|dt| / exponential_weight_scale)
    
    The 'unidirectional' flag can be used in conjunction with any weight mode to zero out samples with dt < 0.
    """
    # TODO: do we really want to have replacement=True?
    def __init__(
        self, 
        dataset=None,
        weight_mode: str = "uniform",
        num_samples: Optional[int] = None,
        replacement: bool = True,
        const_weight: float = 1.0,
        unidirectional: bool = False,
        specific_pairing: Optional[List[Tuple[int, int]]] = None,
        exponential_weight_scale: Optional[float] = 1.0,
    ):
        """
        Initialize the custom weighted sampler.
        
        Args:
            dataset: The dataset to sample from
            weight_mode: One of ["uniform", "exponential"]
            num_samples: Number of samples to draw. If None, uses len(dataset)
            replacement: Whether to sample with replacement
            const_weight: Constant weight
        """

        self.dataset = dataset
        self.weight_mode = weight_mode
        self.const_weight = const_weight
        self.unidirectional = unidirectional
        self.specific_pairing = specific_pairing
        self.exponential_weight_scale = exponential_weight_scale
        
        # Compute weights
        weights = self._compute_weights()
        
        # Set default num_samples if not provided
        if num_samples is None:
            num_samples = len(dataset)
            
        # Initialize parent WeightedRandomSampler
        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=replacement
        )
        
        logger.info(f"CustomWeightedSampler initialized with weight_mode='{weight_mode}', "
                   f"unidirectional={unidirectional}, "
                   f"specific_pairing={specific_pairing}, "
                   f"exponential_weight_scale={exponential_weight_scale}, "
                   f"num_samples={num_samples}, replacement={replacement}")
        logger.info(f"Weight statistics - Min: {weights.min():.6f}, Max: {weights.max():.6f}, "
                   f"Mean: {weights.mean():.6f}, Non-zero: {(weights > 0).sum()}/{len(weights)}")
    
    def _compute_weights(self) -> torch.Tensor:
        """Compute weights for each sample based on the sampling mode."""
        weights = torch.zeros(len(self.dataset), dtype=torch.float)
        
        if self.weight_mode == "uniform":
            # All samples get equal weight
            weights.fill_(self.const_weight)
        
        elif self.weight_mode == "exponential":
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                d = item['d']
                weights[idx] = math.exp(-abs(d) / self.exponential_weight_scale)

        else:
            raise NotImplementedError(f"Weight mode '{self.weight_mode}' is not implemented.")
                                      
                                      
        if self.unidirectional:
            # set all weights with d < 0 to 0
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                d = item['d']
                if d < 0:
                    weights[idx] = 0.0
                           

        if self.specific_pairing:
            # set everything outside a specific list of pairs to 0
            for idx in range(len(self.dataset)):
                item = self.dataset[idx]
                source_idx = item['source_idx']
                target_idx = item['target_idx']
                if (source_idx, target_idx) not in self.specific_pairing:
                    weights[idx] = 0.0  
                            
        return weights
    
    def get_weight_statistics(self) -> dict:
        """Return statistics about the computed weights."""
        weights = torch.tensor(self.weights)
        return {
            'min_weight': weights.min().item(),
            'max_weight': weights.max().item(),
            'mean_weight': weights.mean().item(),
            'std_weight': weights.std().item(),
            'num_nonzero': (weights > 0).sum().item(),
            'total_samples': len(weights),
            'weight_mode': self.weight_mode,
            'unidirectional': self.unidirectional,
            'specific_pairing': self.specific_pairing,
            'exponential_weight_scale': self.exponential_weight_scale,
            'const_weight': self.const_weight,
        }