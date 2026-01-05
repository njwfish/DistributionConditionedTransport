import torch
import numpy as np
from torch.utils.data import Sampler as BaseSampler
from typing import Optional, Iterator, List


class StratifiedBatchSampler(BaseSampler):
    """
    A batch sampler that ensures each batch contains a mix of consecutive 
    (predictor-training) pairs and non-consecutive pairs.
    
    This addresses the sparse signal problem where the predictor only receives
    gradients from consecutive pairs (~10% of data), by guaranteeing each batch
    contains at least `min_consecutive_per_batch` consecutive pairs.
    
    Args:
        dataset: The dataset to sample from. Must return items with 'source_idx' 
                 and 'target_idx' keys.
        batch_size: Total number of samples per batch.
        min_consecutive_per_batch: Minimum number of consecutive pairs (t→t+1) 
                                   per batch. If None, uses consecutive_ratio.
        consecutive_ratio: Target ratio of consecutive pairs per batch (0.0 to 1.0).
                          Only used if min_consecutive_per_batch is None.
                          Default 0.25 means ~25% consecutive pairs per batch.
        drop_last: If True, drop the last incomplete batch.
        shuffle: If True, shuffle indices within each group before sampling.
        seed: Random seed for reproducibility.
        forward_only: If True, only include forward pairs (target_idx > source_idx)
                     for non-consecutive sampling. Default True.
    """
    
    def __init__(
        self,
        dataset=None,
        batch_size: int = 32,
        min_consecutive_per_batch: Optional[int] = None,
        consecutive_ratio: float = 0.25,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
        forward_only: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.forward_only = forward_only
        
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()
        
        # Separate indices into consecutive and non-consecutive groups
        self.consecutive_indices = []
        self.non_consecutive_indices = []
        
        self._categorize_indices()
        
        # Determine how many consecutive pairs per batch
        if min_consecutive_per_batch is not None:
            self.min_consecutive = min(min_consecutive_per_batch, batch_size)
        else:
            self.min_consecutive = max(1, int(batch_size * consecutive_ratio))
        
        # Ensure we don't request more consecutive than available or than batch size
        self.min_consecutive = min(
            self.min_consecutive, 
            len(self.consecutive_indices),
            batch_size
        )
        
        # Non-consecutive count per batch
        self.non_consecutive_per_batch = batch_size - self.min_consecutive
        
    def _categorize_indices(self):
        """Separate dataset indices into consecutive and non-consecutive groups."""
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            source_idx = item['source_idx']
            target_idx = item['target_idx']
            
            is_consecutive = (target_idx - source_idx) == 1
            is_forward = target_idx > source_idx
            
            if is_consecutive:
                self.consecutive_indices.append(idx)
            elif self.forward_only:
                if is_forward:
                    self.non_consecutive_indices.append(idx)
            else:
                self.non_consecutive_indices.append(idx)
        
        self.consecutive_indices = np.array(self.consecutive_indices)
        self.non_consecutive_indices = np.array(self.non_consecutive_indices)
        
    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle indices if requested
        if self.shuffle:
            consecutive_perm = self.rng.permutation(len(self.consecutive_indices))
            non_consecutive_perm = self.rng.permutation(len(self.non_consecutive_indices))
            consecutive_indices = self.consecutive_indices[consecutive_perm]
            non_consecutive_indices = self.non_consecutive_indices[non_consecutive_perm]
        else:
            consecutive_indices = self.consecutive_indices.copy()
            non_consecutive_indices = self.non_consecutive_indices.copy()
        
        # Pointers for cycling through indices
        cons_ptr = 0
        non_cons_ptr = 0
        
        # Calculate number of batches based on the limiting factor
        # We cycle through indices, so we base num_batches on total samples desired
        total_samples = len(self.dataset)
        num_batches = total_samples // self.batch_size
        if not self.drop_last and (total_samples % self.batch_size != 0):
            num_batches += 1
        
        for batch_idx in range(num_batches):
            batch = []
            
            # Add consecutive pairs (with wraparound)
            for _ in range(self.min_consecutive):
                if cons_ptr >= len(consecutive_indices):
                    # Reshuffle and reset pointer
                    if self.shuffle:
                        consecutive_perm = self.rng.permutation(len(self.consecutive_indices))
                        consecutive_indices = self.consecutive_indices[consecutive_perm]
                    cons_ptr = 0
                batch.append(int(consecutive_indices[cons_ptr]))
                cons_ptr += 1
            
            # Add non-consecutive pairs (with wraparound)
            num_non_cons_to_add = min(
                self.non_consecutive_per_batch,
                self.batch_size - len(batch)  # Don't exceed batch size
            )
            
            for _ in range(num_non_cons_to_add):
                if non_cons_ptr >= len(non_consecutive_indices):
                    # Reshuffle and reset pointer
                    if self.shuffle:
                        non_consecutive_perm = self.rng.permutation(len(self.non_consecutive_indices))
                        non_consecutive_indices = self.non_consecutive_indices[non_consecutive_perm]
                    non_cons_ptr = 0
                batch.append(int(non_consecutive_indices[non_cons_ptr]))
                non_cons_ptr += 1
            
            # Handle edge case: not enough non-consecutive, fill with more consecutive
            while len(batch) < self.batch_size and len(consecutive_indices) > 0:
                if cons_ptr >= len(consecutive_indices):
                    if self.shuffle:
                        consecutive_perm = self.rng.permutation(len(self.consecutive_indices))
                        consecutive_indices = self.consecutive_indices[consecutive_perm]
                    cons_ptr = 0
                batch.append(int(consecutive_indices[cons_ptr]))
                cons_ptr += 1
            
            # Drop incomplete last batch if requested
            if self.drop_last and len(batch) < self.batch_size:
                break
                
            # Shuffle the batch so consecutive/non-consecutive aren't grouped
            if self.shuffle:
                self.rng.shuffle(batch)
            
            yield batch
    
    def __len__(self) -> int:
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


class StratifiedWeightedSampler(BaseSampler):
    """
    An index sampler (not batch sampler) that oversamples consecutive pairs
    to ensure they appear more frequently during training.
    
    Use this with a regular DataLoader (not as batch_sampler) when you want
    to control the sampling distribution rather than batch composition.
    
    Args:
        dataset: The dataset to sample from.
        consecutive_weight: Weight multiplier for consecutive pairs.
                           Higher values = more frequent sampling of consecutive pairs.
                           Default 5.0 means consecutive pairs are 5x more likely to be sampled.
        num_samples: Total number of samples per epoch. If None, uses len(dataset).
        seed: Random seed for reproducibility.
        forward_only: If True, set weight=0 for backward pairs (target_idx < source_idx).
    """
    
    def __init__(
        self,
        dataset=None,
        consecutive_weight: float = 5.0,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
        forward_only: bool = False,
    ):
        self.dataset = dataset
        self.consecutive_weight = consecutive_weight
        self.forward_only = forward_only
        
        if num_samples is None:
            self.num_samples = len(dataset)
        else:
            self.num_samples = num_samples
            
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None
        
        # Compute weights
        self.weights = self._compute_weights()
        
    def _compute_weights(self) -> torch.Tensor:
        """Compute sampling weights for each index."""
        weights = torch.ones(len(self.dataset), dtype=torch.float64)
        
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            source_idx = item['source_idx']
            target_idx = item['target_idx']
            
            is_consecutive = (target_idx - source_idx) == 1
            is_forward = target_idx > source_idx
            
            if is_consecutive:
                weights[idx] = self.consecutive_weight
            elif self.forward_only and not is_forward:
                weights[idx] = 0.0
        
        return weights
    
    def __iter__(self) -> Iterator[int]:
        # Use torch's multinomial for weighted sampling
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=True,
            generator=self.generator
        )
        yield from indices.tolist()
    
    def __len__(self) -> int:
        return self.num_samples

