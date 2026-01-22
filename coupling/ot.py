"""
Implement an OT coupling class that implements a call function so it can be used as a collate function for a DataLoader.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import default_collate
from typing import Optional, Callable
import warnings

import ot

ot_fns = {
    'sinkhorn': ot.sinkhorn,
    'emd': ot.emd,
    'bregman': ot.bregman.empirical_sinkhorn
}


class BaseOTCollate(nn.Module):
    """
    Base class for OT-based coupling in DataLoader collate functions.

    Subclasses should implement `compute_cost_matrix` to define how
    the cost matrix is computed from source and target samples.
    """

    def __init__(
        self,
        ot_type: str = 'sinkhorn',
        ot_params: Optional[dict] = None,
        replace: bool = False
    ):
        super(BaseOTCollate, self).__init__()
        if ot_params is None:
            ot_params = {}
        self.ot_type = ot_type
        self.ot_params = ot_params
        self.replace = replace

    def compute_cost_matrix(self, source_samples, target_samples) -> np.ndarray:
        """
        Compute the cost matrix between source and target samples.

        Args:
            source_samples: Source samples from the batch
            target_samples: Target samples from the batch

        Returns:
            np.ndarray of shape (n_source, n_target) with costs
        """
        raise NotImplementedError("Subclasses must implement compute_cost_matrix")

    def get_set_size(self, source_samples, target_samples) -> int:
        """Get the set size from samples. Override if needed."""
        if isinstance(source_samples, torch.Tensor):
            return source_samples.shape[0]
        elif isinstance(source_samples, dict):
            # Get size from first tensor in dict
            for v in source_samples.values():
                if isinstance(v, torch.Tensor):
                    return v.shape[0]
                elif isinstance(v, (list, tuple)):
                    return len(v)
        raise ValueError("Cannot determine set size from samples")

    def reorder_samples(self, sample, source_idx, target_idx):
        """
        Reorder source and target samples according to OT indices.
        Override for custom sample structures.
        """
        processed = {}
        for key, value in sample.items():
            if key == 'source_samples':
                processed[key] = self._reorder_data(value, source_idx)
            elif key == 'target_samples':
                processed[key] = self._reorder_data(value, target_idx)
            else:
                processed[key] = value
        return processed

    def _reorder_data(self, data, idx):
        """Reorder a dict of tensors/lists or a single tensor by indices."""
        if isinstance(data, dict):
            result = {}
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    result[k] = v[idx]
                elif isinstance(v, (list, tuple)):
                    result[k] = [v[i] for i in idx]
                else:
                    result[k] = v
            return result
        elif isinstance(data, torch.Tensor):
            return data[idx]
        else:
            return data

    def solve_ot(self, cost: np.ndarray, set_size: int, ot_type: str, ot_params: dict):
        """
        Solve OT problem and sample indices from the transport plan.

        Args:
            cost: Cost matrix of shape (n_source, n_target)
            set_size: Number of pairs to sample
            ot_type: Type of OT solver
            ot_params: Parameters for OT solver

        Returns:
            Tuple of (source_indices, target_indices)
        """
        # Create uniform marginal distributions
        a = ot.unif(cost.shape[0])
        b = ot.unif(cost.shape[1])

        # Compute OT plan
        G = ot_fns[ot_type](a, b, cost, **ot_params)

        # Check for numerical issues
        if not np.all(np.isfinite(G)):
            warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            G = np.ones_like(G) / G.size
        if np.abs(G.sum()) < 1e-8:
            warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            G = np.ones_like(G) / G.size

        # Sample from the OT plan
        G_flat = G.flatten()
        G_flat = G_flat / G_flat.sum()  # Normalize to ensure valid probability

        choices = np.random.choice(
            G.shape[0] * G.shape[1],
            p=G_flat,
            size=set_size,
            replace=self.replace
        )
        idx0, idx1 = np.divmod(choices, G.shape[1])

        return idx0, idx1

    def __call__(self, batch, ot_type: Optional[str] = None, ot_params: Optional[dict] = None):
        if ot_type is None:
            ot_type = self.ot_type
        if ot_params is None:
            ot_params = self.ot_params

        processed_samples = []

        for sample in batch:
            source_samples = sample['source_samples']
            target_samples = sample['target_samples']

            set_size = self.get_set_size(source_samples, target_samples)

            # Compute cost matrix (subclass-specific)
            cost = self.compute_cost_matrix(source_samples, target_samples)

            # Solve OT and get indices
            source_idx, target_idx = self.solve_ot(cost, set_size, ot_type, ot_params)

            # Reorder samples
            processed_sample = self.reorder_samples(sample, source_idx, target_idx)
            processed_samples.append(processed_sample)

        return default_collate(processed_samples)


class OTCollate(BaseOTCollate):
    """
    OT coupling using Euclidean distance on tensor features.
    """

    def __init__(
        self,
        ot_type: str = 'sinkhorn',
        ot_params: Optional[dict] = None,
        pairwise_dist_fn: Optional[Callable] = None,
        replace: bool = False
    ):
        super(OTCollate, self).__init__(ot_type, ot_params, replace)

        if pairwise_dist_fn is None:
            self.pairwise_dist_fn = lambda x, y: torch.cdist(x, y, p=2) ** 2
        else:
            self.pairwise_dist_fn = pairwise_dist_fn

    def compute_cost_matrix(self, source_samples, target_samples) -> np.ndarray:
        """Compute cost matrix using pairwise distance function on tensors."""
        # Flatten if needed (for multi-dimensional features)
        if source_samples.dim() > 2:
            source_flat = source_samples.reshape(source_samples.shape[0], -1)
        else:
            source_flat = source_samples
        if target_samples.dim() > 2:
            target_flat = target_samples.reshape(target_samples.shape[0], -1)
        else:
            target_flat = target_samples

        cost = self.pairwise_dist_fn(source_flat, target_flat)
        return cost.detach().cpu().numpy()

    def reorder_samples(self, sample, source_idx, target_idx):
        """Simple reordering for tensor-based samples."""
        processed = sample.copy()
        processed['source_samples'] = sample['source_samples'][source_idx]
        processed['target_samples'] = sample['target_samples'][target_idx]
        return processed