import torch
import math
import numpy as np
from torch.utils.data import WeightedRandomSampler, Subset
from typing import Optional, Union, Any
import os
import logging

logger = logging.getLogger(__name__)


class CustomWeightedSampler(WeightedRandomSampler):
    """
    Custom weighted sampler that supports different sampling modes based on 'dt' values in dataset items.
    
    Sampling modes:
    - "bidirectional": Equal weights for all samples (ignores dt values)
    - "unidirectional": Positive weight only for samples with dt > 0, zero weight otherwise
    - "exponential": Weight = exp(|dt|) / ln(2) for each sample
    - "dt_equals_one": Weight = 1 for samples where dt == 1, zero weight otherwise
    """
    # TODO: do we really want to have replacement=True?
    def __init__(
        self, 
        dataset,
        sampling_mode: str = "bidirectional",
        num_samples: Optional[int] = None,
        replacement: bool = True,
        const_weight: float = 1.0,
        time_index_path: Optional[str] = None,
        time_scale: float = 1.0,
        cfg: Optional[Any] = None,
    ):
        """
        Initialize the custom weighted sampler.
        
        Args:
            dataset: The dataset to sample from
            sampling_mode: One of ["bidirectional", "unidirectional", "exponential", "dt_equals_one"]
            num_samples: Number of samples to draw. If None, uses len(dataset)
            replacement: Whether to sample with replacement
            const_weight: Constant weight for unidirectional/dt_equals_one modes
        """
        if sampling_mode not in ["bidirectional", "unidirectional", "exponential", "dt_equals_one"]:
            raise NotImplementedError(f"Sampling mode '{sampling_mode}' is not implemented. "
                                    f"Must be one of: ['bidirectional', 'unidirectional', 'exponential', 'dt_equals_one']")
        
        self.dataset = dataset
        self.sampling_mode = sampling_mode
        self.const_weight = const_weight
        self.time_index_path = time_index_path
        self.time_scale = float(time_scale) if time_scale is not None else 1.0
        self.cfg = cfg
        # Optionally load precomputed year-month indices for the base dataset elements
        self._time_indices: Optional[np.ndarray] = None
        self._base_n: Optional[int] = None
        if self.time_index_path is not None:
            npz = np.load(self.time_index_path)
            self._time_indices = np.array(npz['time_indices'], dtype=np.int64)
            # Determine base dataset object and base element count n
            self._base_n = len(self._time_indices)

        
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
                dt = self._get_dt(idx)
                if dt > 0:
                    weights[idx] = self.const_weight
                else:
                    weights[idx] = 0.0

                    
        elif self.sampling_mode == "exponential":
            # Weight = exp(|dt|) / ln(2)
            for idx in range(len(self.dataset)):
                dt = self._get_dt(idx)
                weights[idx] = math.exp(-abs(dt)) / self.cfg.experiment.sampling_exponential_weight_scale
                    
        elif self.sampling_mode == "dt_equals_one":
            # Only samples with dt == 1 get positive weight
            for idx in range(len(self.dataset)):
                dt = self._get_dt(idx)
                if dt == 1:
                    weights[idx] = self.const_weight
                else:
                    weights[idx] = 0.0
        
        # Ensure we have at least some positive weights
        if weights.sum() == 0:
            logger.error(f"All weights are zero for sampling mode '{self.sampling_mode}'. "
                        f"This will cause sampling to fail.")
            raise ValueError(f"All weights are zero for sampling mode '{self.sampling_mode}'")
        
        return weights
    
    def _get_dt(self, pair_idx: int) -> float:
        """Return dt using fast mapping if available; otherwise access dataset item."""
        if self._time_indices is not None and self._base_n is not None:
            # Map local index to global underlying pair index if we're sampling a Subset

            n = self._base_n
            # Map linear pair index to (i, j) with skipped diagonal, as in datasets
            i = pair_idx // (n - 1)
            j = pair_idx % (n - 1)
            if j >= i:
                j += 1
            dt_months = int(self._time_indices[j]) - int(self._time_indices[i])
            return float(dt_months) / self.time_scale if self.time_scale != 0 else float(dt_months)
        
        # Fallback: access dataset (may be slower and more memory intensive)
        item = self.dataset[pair_idx]
        if isinstance(item, dict) and 'dt' in item:
            return float(item['dt'])
        raise KeyError("Dataset item does not contain 'dt'")
    
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