"""
Optimized stratified samplers for Trellis datasets.

These samplers compute weights/indices analytically based on dataset structure,
avoiding the need to iterate through every item (which is extremely slow for 
large datasets like trellis_a2a with millions of items).
"""

import torch
import numpy as np
from torch.utils.data import Sampler as BaseSampler
from typing import Optional, Iterator, List


class StratifiedBatchSamplerTrellis(BaseSampler):
    """
    A batch sampler optimized for Trellis datasets that computes indices analytically.
    
    This addresses the performance issue where iterating through large datasets
    (e.g., trellis_a2a with millions of items) takes hours. Instead, it computes
    which indices are "consecutive" pairs directly from the dataset structure.
    
    Args:
        dataset: The dataset to sample from. Must have either:
                 - n_train and total_x_populations attributes (trellis_a2a style)
                 - num_samples attribute (regular trellis style)
        batch_size: Total number of samples per batch.
        num_samples: Total number of samples per epoch. If None, uses len(dataset).
        min_consecutive_per_batch: Minimum number of consecutive pairs per batch.
        consecutive_ratio: Target ratio of consecutive pairs per batch (0.0 to 1.0).
        drop_last: If True, drop the last incomplete batch.
        shuffle: If True, shuffle indices within each group before sampling.
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        dataset=None,
        batch_size: int = 32,
        num_samples: Optional[int] = None,
        min_consecutive_per_batch: Optional[int] = None,
        consecutive_ratio: float = 0.25,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        
        # Set num_samples (total samples per epoch)
        if num_samples is None:
            self.num_samples = len(dataset)
        else:
            self.num_samples = num_samples
        
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
        """
        Compute consecutive and non-consecutive indices analytically.
        
        This avoids iterating through the entire dataset (which is very slow
        for large datasets with millions of items).
        """
        dataset_len = len(self.dataset)
        
        # trellis_a2a style dataset
        if hasattr(self.dataset, 'n_train') and hasattr(self.dataset, 'total_x_populations'):
            N = self.dataset.total_x_populations
            n_train = self.dataset.n_train
            
            # Consecutive pairs: source is x0 (2*i), target is x1 (2*i+1) for same train sample
            # Flat index layout: [train_0_x0, train_0_x1, train_1_x0, train_1_x1, ..., test_0_x0, ...]
            for i in range(n_train):
                source_flat = 2 * i      # x0 of train sample i
                target_flat = 2 * i + 1  # x1 of train sample i
                idx = source_flat * N + target_flat
                self.consecutive_indices.append(idx)
            
            # Non-consecutive: all other indices
            consecutive_set = set(self.consecutive_indices)
            for idx in range(dataset_len):
                if idx not in consecutive_set:
                    self.non_consecutive_indices.append(idx)
            
            print(f"[StratifiedBatchSamplerTrellis] Computed indices analytically: "
                  f"{len(self.consecutive_indices)} consecutive, {len(self.non_consecutive_indices)} non-consecutive")
        

        else:
            raise ValueError("Dataset must have (n_train, total_x_populations) attributes for trellis_a2a style")
        
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
        
        # Calculate number of batches based on num_samples
        num_batches = self.num_samples // self.batch_size
        if not self.drop_last and (self.num_samples % self.batch_size != 0):
            num_batches += 1
        
        for batch_idx in range(num_batches):
            batch = []
            
            # Add consecutive pairs (with wraparound)
            for _ in range(self.min_consecutive):
                if cons_ptr >= len(consecutive_indices):
                    if self.shuffle:
                        consecutive_perm = self.rng.permutation(len(self.consecutive_indices))
                        consecutive_indices = self.consecutive_indices[consecutive_perm]
                    cons_ptr = 0
                batch.append(int(consecutive_indices[cons_ptr]))
                cons_ptr += 1
            
            # Add non-consecutive pairs (with wraparound)
            num_non_cons_to_add = min(
                self.non_consecutive_per_batch,
                self.batch_size - len(batch)
            )
            
            for _ in range(num_non_cons_to_add):
                if non_cons_ptr >= len(non_consecutive_indices):
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
        if self.drop_last:
            return self.num_samples // self.batch_size
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size


class StratifiedWeightedSamplerTrellis(BaseSampler):
    """
    An optimized weighted sampler for Trellis datasets that computes weights analytically.
    
    This addresses the performance issue where iterating through large datasets
    (e.g., trellis_a2a with millions of items) takes hours. Instead, it computes
    sampling weights directly from the dataset structure without calling __getitem__.
    
    Args:
        dataset: The dataset to sample from. Must have either:
                 - n_train and total_x_populations attributes (trellis_a2a style)
                 - num_samples attribute (regular trellis style)
        consecutive_weight: Weight multiplier for consecutive pairs.
        num_samples: Total number of samples per epoch. If None, uses len(dataset).
        seed: Random seed for reproducibility.
    """
    
    def __init__(
        self,
        dataset=None,
        consecutive_weight: float = 1.0,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        self.dataset = dataset
        self.consecutive_weight = consecutive_weight
        
        if num_samples is None:
            self.num_samples = len(dataset)
        else:
            self.num_samples = num_samples
            
        if seed is not None:
            self.generator = torch.Generator().manual_seed(seed)
        else:
            self.generator = None
        
        # Compute weights analytically
        self.weights = self._compute_weights()
        
    def _compute_weights(self) -> torch.Tensor:
        """
        Compute sampling weights analytically based on dataset structure.
        
        This avoids calling __getitem__ for every index, which is extremely slow
        for large datasets (e.g., trellis_a2a with millions of items).
        """
        dataset_len = len(self.dataset)
        weights = torch.ones(dataset_len, dtype=torch.float64)
        
        # trellis_a2a style dataset
        if hasattr(self.dataset, 'n_train') and hasattr(self.dataset, 'total_x_populations'):
            N = self.dataset.total_x_populations
            n_train = self.dataset.n_train
            
            # Consecutive pairs are where source is x0 and target is x1 from the same train sample
            # Flat index layout: [train_0_x0, train_0_x1, train_1_x0, train_1_x1, ..., test_0_x0, ...]
            # For train sample i: source_flat = 2*i (x0), target_flat = 2*i+1 (x1)
            # idx = source_flat * N + target_flat
            for i in range(n_train):
                source_flat = 2 * i      # x0 of train sample i
                target_flat = 2 * i + 1  # x1 of train sample i
                idx = source_flat * N + target_flat
                weights[idx] = self.consecutive_weight
            
            print(f"[StratifiedWeightedSamplerTrellis] Computed weights analytically for {dataset_len} items "
                  f"({n_train} consecutive pairs with weight {self.consecutive_weight})")
        
        else:
            raise ValueError("Dataset must have (n_train, total_x_populations) attributes for trellis_a2a style")
        
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