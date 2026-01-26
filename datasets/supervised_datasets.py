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
    to create a deterministic linear relationship: Y = X + b (shift only).
    The SAME b is used for ALL distributions, making the relationship learnable.
    The target samples are then shuffled (permuted) to create an unpaired but
    distributionally related dataset.
    """

    def __init__(
            self,
            shift: Tuple[float, ...] = (1.0, 1.0),
            *args,
            **kwargs
            ):
        """
        Args:
            shift: Fixed shift vector b for transformation Y = X + b
            *args, **kwargs: Passed to MultivariateNormalDistributionDataset
        """
        super().__init__(*args, **kwargs)

        # Get data dimensionality from the parent class
        data_dim = self.data.shape[-1]

        # Fixed shift transformation: A = I, b = shift
        self.A = np.eye(data_dim)
        self.b = np.array(shift)

    def __getitem__(self, idx):
        # Get the unique set index for tracking
        unique_idx = self.set_idx[idx]

        # Source samples from parent
        source_samples = self.data[idx].copy()

        # Apply linear transformation: Y = X @ A^T + b (same A, b for all)
        target_samples = source_samples @ self.A.T + self.b

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
    to create a deterministic linear relationship: Y = X + b (shift only).
    The SAME b is used for ALL distributions, making the relationship learnable.
    The target samples are then shuffled (permuted) to create an unpaired but
    distributionally related dataset.
    """

    def __init__(
            self,
            shift: Tuple[float, ...] = (0.5, 0.5),
            off_axis: float = 0.1,
            *args,
            **kwargs
            ):
        """
        Args:
            shift: Fixed shift vector b for transformation Y = X + b
            *args, **kwargs: Passed to GaussianMixtureModelDistributionDataset
        """
        super().__init__(*args, **kwargs)

        # Get data dimensionality from the parent class
        data_dim = self.data.shape[-1]

        # Fixed shift transformation: A = I, b = shift
        self.A = np.eye(data_dim)
        self.b = np.array(shift)
        self.off_axis_shift = off_axis * np.array([-1, 1])

    def __getitem__(self, idx):
        # Get the unique set index for tracking
        unique_idx = self.set_idx[idx]

        # Source samples from parent
        source_samples = self.data[idx].copy()

        # Apply linear transformation: Y = X @ A^T + b (same A, b for all)
        target_samples_left = source_samples @ self.A.T + self.b + self.off_axis_shift
        target_samples_right = source_samples @ self.A.T + self.b - self.off_axis_shift
        target_samples = np.vstack([target_samples_left, target_samples_right])


        # Shuffle target samples
        np.random.shuffle(target_samples)

        return {
            'source_samples': torch.tensor(source_samples, dtype=torch.float),
            'target_samples': torch.tensor(target_samples[:source_samples.shape[0]], dtype=torch.float),
            'source_idx': unique_idx,
            'target_idx': unique_idx,  # Same idx since target is derived from source
        }
