import numpy as np
import torch
from typing import Optional, Tuple

from datasets.distribution_datasets import (
    MultivariateNormalDistributionDataset,
    GaussianMixtureModelDistributionDataset
)


class SupervisedMVNDataset(MultivariateNormalDistributionDataset):
    """
    Supervised dataset for multivariate normal distributions.

    Inherits from MultivariateNormalDistributionDataset but overrides __getitem__
    to create a deterministic linear relationship: Y = A @ X + b, where A is a
    random linear transformation and b is a random bias vector. The target samples
    are then shuffled (permuted) to create an unpaired but distributionally
    related dataset.
    """

    def __init__(
            self,
            transform_scale: float = 1.0,
            bias_scale: float = 1.0,
            *args,
            **kwargs
            ):
        """
        Args:
            transform_scale: Scale for random linear transformation A
            bias_scale: Scale for random bias vector b
            *args, **kwargs: Passed to MultivariateNormalDistributionDataset
        """
        super().__init__(*args, **kwargs)

        self.transform_scale = transform_scale
        self.bias_scale = bias_scale

        # Get data dimensionality from the parent class
        data_dim = self.data.shape[-1]
        n_sets = self.n_sets

        # Generate linear transformation parameters for each unique set
        # Use a separate random state to not interfere with parent's data
        rng = np.random.RandomState(kwargs.get('seed', 42) + 1000)

        if self.n_unique_sets is not None:
            n_unique = self.n_unique_sets
        else:
            n_unique = n_sets

        self.A = transform_scale * rng.randn(n_unique, data_dim, data_dim)
        self.b = bias_scale * rng.randn(n_unique, data_dim)

    def __getitem__(self, idx):
        # Get the unique set index for transformation lookup
        unique_idx = self.set_idx[idx]

        # Source samples from parent
        source_samples = self.data[idx].copy()

        # Apply linear transformation: Y = X @ A^T + b
        A = self.A[unique_idx]
        b = self.b[unique_idx]
        target_samples = source_samples @ A.T + b

        # Shuffle target samples
        np.random.shuffle(target_samples)

        return {
            'source_samples': torch.tensor(source_samples, dtype=torch.float),
            'target_samples': torch.tensor(target_samples, dtype=torch.float),
            'source_idx': unique_idx,
            'target_idx': unique_idx,  # Same idx since target is derived from source
        }


class SupervisedGMMDataset(GaussianMixtureModelDistributionDataset):
    """
    Supervised dataset for Gaussian Mixture Model distributions.

    Inherits from GaussianMixtureModelDistributionDataset but overrides __getitem__
    to create a deterministic linear relationship: Y = A @ X + b, where A is a
    random linear transformation and b is a random bias vector. The target samples
    are then shuffled (permuted) to create an unpaired but distributionally
    related dataset.
    """

    def __init__(
            self,
            transform_scale: float = 1.0,
            bias_scale: float = 1.0,
            *args,
            **kwargs
            ):
        """
        Args:
            transform_scale: Scale for random linear transformation A
            bias_scale: Scale for random bias vector b
            *args, **kwargs: Passed to GaussianMixtureModelDistributionDataset
        """
        super().__init__(*args, **kwargs)

        self.transform_scale = transform_scale
        self.bias_scale = bias_scale

        # Get data dimensionality from the parent class
        data_dim = self.data.shape[-1]
        n_sets = self.n_sets

        # Generate linear transformation parameters for each unique set
        rng = np.random.RandomState(kwargs.get('seed', 42) + 1000)

        if self.n_unique_sets is not None:
            n_unique = self.n_unique_sets
        else:
            n_unique = n_sets

        self.A = transform_scale * rng.randn(n_unique, data_dim, data_dim)
        self.b = bias_scale * rng.randn(n_unique, data_dim)

    def __getitem__(self, idx):
        # Get the unique set index for transformation lookup
        unique_idx = self.set_idx[idx]

        # Source samples from parent
        source_samples = self.data[idx].copy()

        # Apply linear transformation: Y = X @ A^T + b
        A = self.A[unique_idx]
        b = self.b[unique_idx]
        target_samples = source_samples @ A.T + b

        # Shuffle target samples
        np.random.shuffle(target_samples)

        return {
            'source_samples': torch.tensor(source_samples, dtype=torch.float),
            'target_samples': torch.tensor(target_samples, dtype=torch.float),
            'source_idx': unique_idx,
            'target_idx': unique_idx,  # Same idx since target is derived from source
        }
