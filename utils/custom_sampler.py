import torch
import math
from torch.utils.data import WeightedRandomSampler
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


class CustomWeightedSampler(WeightedRandomSampler):
    """
    Custom weighted sampler that supports different sampling modes based on 'dt' values in dataset items.
    
    Sampling modes:
    - "bidirectional": Equal weights for all samples (ignores dt values)
    - "unidirectional": Positive weight only for samples with dt > 0, zero weight otherwise
    - "exponential": Weight = exp(|dt|) / ln(2) for each sample
    """
    # TODO: do we really want to have replacement=True?
    def __init__(
        self, 
        dataset,
        sampling_mode: str = "bidirectional",
        num_samples: Optional[int] = None,
        replacement: bool = True,
        const_weight: float = 1.0
    ):
        """
        Initialize the custom weighted sampler.
        
        Args:
            dataset: The dataset to sample from
            sampling_mode: One of ["bidirectional", "unidirectional", "exponential"]
            num_samples: Number of samples to draw. If None, uses len(dataset)
            replacement: Whether to sample with replacement
            const_weight: Constant weight for unidirectional mode when dt > 0
        """
        if sampling_mode not in ["bidirectional", "unidirectional", "exponential"]:
            raise NotImplementedError(f"Sampling mode '{sampling_mode}' is not implemented. "
                                    f"Must be one of: ['bidirectional', 'unidirectional', 'exponential']")
        
        self.dataset = dataset
        self.sampling_mode = sampling_mode
        self.const_weight = const_weight
        
        # Compute weights based on sampling mode
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
        
        logger.info(f"CustomWeightedSampler initialized with mode='{sampling_mode}', "
                   f"num_samples={num_samples}, replacement={replacement}")
        logger.info(f"Weight statistics - Min: {weights.min():.6f}, Max: {weights.max():.6f}, "
                   f"Mean: {weights.mean():.6f}, Non-zero: {(weights > 0).sum()}/{len(weights)}")
    
    def _compute_weights(self) -> torch.Tensor:
        """Compute weights for each sample based on the sampling mode."""
        weights = torch.zeros(len(self.dataset), dtype=torch.float)
        
        if self.sampling_mode == "bidirectional":
            # All samples get equal weight
            weights.fill_(self.const_weight)
            
        elif self.sampling_mode == "unidirectional":
            # Only samples with dt > 0 get positive weight
            for idx in range(len(self.dataset)):
                try:
                    item = self.dataset[idx]
                    if isinstance(item, dict) and 'dt' in item:
                        dt = item['dt']
                        if dt > 0:
                            weights[idx] = self.const_weight
                        else:
                            weights[idx] = 0.0
                    else:
                        logger.warning(f"Item at index {idx} is not a dict or doesn't contain 'dt' key. "
                                     f"Setting weight to 0.")
                        weights[idx] = 0.0
                except Exception as e:
                    logger.warning(f"Error accessing item at index {idx}: {e}. Setting weight to 0.")
                    weights[idx] = 0.0
                    
        elif self.sampling_mode == "exponential":
            # Weight = exp(|dt|) / ln(2)
            ln_2 = math.log(2)
            for idx in range(len(self.dataset)):
                try:
                    item = self.dataset[idx]
                    if isinstance(item, dict) and 'dt' in item:
                        dt = item['dt']
                        weights[idx] = math.exp(abs(dt)) / ln_2
                    else:
                        logger.warning(f"Item at index {idx} is not a dict or doesn't contain 'dt' key. "
                                     f"Setting weight to 1/ln(2).")
                        weights[idx] = 1.0 / ln_2  # exp(0) / ln(2)
                except Exception as e:
                    logger.warning(f"Error accessing item at index {idx}: {e}. Setting weight to 1/ln(2).")
                    weights[idx] = 1.0 / ln_2
        
        # Ensure we have at least some positive weights
        if weights.sum() == 0:
            logger.error(f"All weights are zero for sampling mode '{self.sampling_mode}'. "
                        f"This will cause sampling to fail.")
            raise ValueError(f"All weights are zero for sampling mode '{self.sampling_mode}'")
        
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
            'sampling_mode': self.sampling_mode
        }