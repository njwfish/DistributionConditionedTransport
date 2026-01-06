import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, List, Any
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import scipy as sp

class MultivariateNormalDistributionDataset(Dataset):
    """Dataset for multivariate normal distributions."""
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            data_shape: Tuple[int, ...] = (2,),     
            prior_mu: Tuple[float, float] = (0, 1),
            prior_cov_df: int = 10,
            prior_cov_scale: float = 1.,
            seed: Optional[int] = None,
            n_unique_sets: Optional[int] = None,
            ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            seed: Random seed for reproducibility
        """
        prior_cov_df = prior_cov_df * (data_shape[0] - 1)
        self.n_sets = n_sets
        self.set_size = set_size
        self.n_unique_sets = n_unique_sets

        if seed is not None:
            np.random.seed(seed)

        self.delta = prior_mu[1] - prior_mu[0]
            
        self.mu = np.random.uniform(prior_mu[0], prior_mu[1], (n_sets, data_shape[0]))
        # sample covariance matrix from inverse wishart distribution
        self.cov = np.linalg.inv(sp.stats.wishart.rvs(
            df=prior_cov_df, scale=prior_cov_scale*np.eye(data_shape[0]), size=n_sets
        ))

        self.set_idx = np.arange(n_sets)

        if n_unique_sets is not None:
            self.mu = np.repeat(self.mu[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            self.cov = np.repeat(self.cov[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            self.set_idx = np.repeat(self.set_idx[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            
        
        self.data = self.sample(self.mu, self.cov, n_sets, set_size, data_shape)

        
    def sample(self, mu, cov, n_sets, set_size, data_shape):
        """
        Vectorized sampling from a multivariate normal distribution.
        
        Args:
            mu: Mean of the distributions (n_sets, n_features)
            cov: Covariance matrix of the distributions (n_sets, n_features, n_features)
            n_sets: Number of sets to sample from
            set_size: Number of samples per set
            data_shape: Shape of each sample
            
        Returns:
            z: Samples from the multivariate normal distribution (n_sets, set_size, n_features)
        """
        z = np.random.randn(n_sets, set_size, data_shape[0])

        L_cov = np.linalg.cholesky(cov)
        # transform the normal to a multivariate normal
        # z = np.einsum('ijk,ikl->ijl', z, L_cov)
        z = np.matmul(z, L_cov.transpose(0, 2, 1))
        z = z + mu[:, None, :]
        return z
    
    def fisher_rao_distance(self, mu, cov):
        raise NotImplementedError("Fisher-Rao distance not implemented for multivariate normal distribution")
    
    def wasserstein_distance(self, mu, cov):
        mean_dist = np.linalg.norm(mu[:, None, :] - mu[None, :, :], axis=2)
        var_dist = np.zeros((mu.shape[0], mu.shape[0]))
        for i, cov_i in enumerate(cov):
            for j, cov_j in enumerate(cov):
                var_dist[i, j] = np.trace(cov_i + cov_j - 2 * sp.linalg.sqrtm(sp.linalg.sqrtm(cov_i) @ cov_j @ sp.linalg.sqrtm(cov_i)))
        return mean_dist + var_dist

    def __len__(self):
        return self.n_sets
    
    def __getitem__(self, idx):

        source_idx = idx
        target_idx = np.random.choice(self.n_sets, size=1, replace=False)[0]
        source_samples = torch.tensor(self.data[source_idx], dtype=torch.float)
        target_samples = torch.tensor(self.data[target_idx], dtype=torch.float)
        
        return {
            'source_samples': source_samples,
            'target_samples': target_samples,
            'source_idx': self.set_idx[source_idx],
            'target_idx': self.set_idx[target_idx],
        }

class ShiftedMultivariateNormalDistributionDataset(MultivariateNormalDistributionDataset):
    def __init__(self, *args, shift_scale: float = 1., **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_scale = shift_scale

    def __getitem__(self, idx):
        source_idx = idx
        target_idx = np.random.choice(self.n_sets, size=1, replace=False)[0]
        source_samples = self.data[source_idx]
        # target_samples = torch.tensor(self.data[target_idx], dtype=torch.float)
        
        # Compute Wasserstein distance between the two distributions using their parameters
        source_mu = self.mu[source_idx]
        source_cov = self.cov[source_idx]

        # new sample 
        new_sample = np.random.multivariate_normal(source_mu, source_cov, size=self.set_size)
        shift_mu = self.shift_scale * source_mu @ source_cov
        target_samples = new_sample + shift_mu 
        
        return {
            'source_samples': torch.tensor(source_samples, dtype=torch.float),
            'target_samples': torch.tensor(target_samples, dtype=torch.float),
            'source_idx': source_idx,
            'target_idx': target_idx,
        }

    def __len__(self):
        return self.data.shape[0]


class LowRankMultivariateNormalDistributionDataset(Dataset):
    def __init__(
        self,
        n_sets: int = 10_000,
        set_size: int = 100,
        data_shape: Tuple[int, ...] = (1000,),
        rank: int = 5,
        prior_mu: Tuple[float, float] = (0, 1),
        prior_cov_df: int = 10,
        prior_cov_scale: float = 1.,
        noise_scale: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.n_sets = n_sets
        self.set_size = set_size
        self.dim = data_shape[0]
        
        if seed is not None:
            np.random.seed(seed)

        self.mu = np.random.uniform(prior_mu[0], prior_mu[1], (n_sets, self.dim))
        self.cov = []
        self.data = []

        for i in range(n_sets):
            # low-rank latent cov
            latent_cov = sp.stats.wishart.rvs(df=prior_cov_df, scale=prior_cov_scale*np.eye(rank))
            A = np.random.randn(self.dim, rank)
            full_cov = A @ latent_cov @ A.T + noise_scale * np.eye(self.dim)

            x = np.random.multivariate_normal(self.mu[i], full_cov, size=set_size)

            self.cov.append(full_cov)
            self.data.append(x)

        self.cov = np.stack(self.cov)
        self.data = np.stack(self.data)

    def __len__(self):
        return self.n_sets

    def __getitem__(self, idx):
        return {
            'samples': torch.tensor(self.data[idx], dtype=torch.float),
            'mean': torch.tensor(self.mu[idx], dtype=torch.float),
            'cov': torch.tensor(self.cov[idx], dtype=torch.float)
        }
    
    def sample(self, mu, cov, n_sets, set_size, data_shape):
        """
        Vectorized sampling with high-dimensional noise but low-rank covariance.
        
        Args:
            mu: Mean vectors in low-rank space (n_sets, rank)
            cov: Covariance matrices in low-rank space (n_sets, rank, rank)
            n_sets: Number of sets to sample from
            set_size: Number of samples per set
            data_shape: Shape of each sample (high-dimensional)
        """
        # Generate high-dimensional standard normal noise
        z = np.random.randn(20, set_size, data_shape[0])
        
        # Compute Cholesky decomposition in low-rank space
        L_cov = np.linalg.cholesky(cov)  # shape: (n_sets, rank, rank)
        
        # Project the Cholesky factor to high-dimensional space
        # projection_matrix shape: (rank, high_dim)
        # L_cov_high_dim will be shape: (n_sets, rank, high_dim)
        L_cov_high_dim = np.matmul(L_cov, self.projection_matrix)
        
        # Project the mean to high-dimensional space
        # mu_high_dim will be shape: (n_sets, high_dim)
        mu_high_dim = np.matmul(mu, self.projection_matrix)
        
        # Apply the projected Cholesky factor to the noise
        # We need to reshape z for proper matrix multiplication
        # result shape: (n_sets, set_size, high_dim)
        z = np.matmul(z, L_cov_high_dim.transpose(0, 2, 1))
        
        # Add the projected mean
        z = z + mu_high_dim[:, None, :]
        
        return z

        
class GaussianMixtureModelDistributionDataset(Dataset):
    """Dataset for Gaussian Mixture Models."""
    
    def __init__(
            self, 
            n_sets: int = 10_000, 
            set_size: int = 100, 
            n_unique_sets: Optional[int] = None,
            data_shape: Tuple[int, ...] = (2,),
            n_components: int = 3,
            prior_mu: Tuple[float, float] = (0, 1),
            prior_cov_df: int = 10,
            prior_cov_scale: float = 1.,
            dirichlet_alpha: float = 1.,
            seed: Optional[int] = None,
            ):
        """
        Args:
            n_sets: Number of parameter sets to generate
            set_size: Number of samples per parameter set
            data_shape: Shape of each sample
            n_components: Number of mixture components
            prior_mu: Range for uniform prior on component means
            prior_cov_df: Degrees of freedom for Wishart prior on precision matrices
            prior_cov_scale: Scale matrix for Wishart prior
            dirichlet_alpha: Concentration parameter for Dirichlet prior on mixture weights
            seed: Random seed for reproducibility
        """
        self.n_sets = n_sets
        self.n_components = n_components
        self.n_unique_sets = n_unique_sets
        
        if seed is not None:
            np.random.seed(seed)
            
        # Sample mixture weights from Dirichlet distribution
        self.weights = np.random.dirichlet(
            alpha=[dirichlet_alpha] * n_components, 
            size=n_sets
        )  # shape: (n_sets, n_components)
        
        # Sample component means
        self.mu = np.random.uniform(
            prior_mu[0], 
            prior_mu[1], 
            (n_sets, n_components, data_shape[0])
        )  # shape: (n_sets, n_components, data_dim)
        
        # Sample component covariances from inverse Wishart
        self.cov = np.zeros((n_sets, n_components, data_shape[0], data_shape[0]))
        for i in range(n_sets):
            for j in range(n_components):
                self.cov[i, j] = np.linalg.inv(sp.stats.wishart.rvs(
                    df=prior_cov_df, 
                    scale=prior_cov_scale*np.eye(data_shape[0])
                ))
        
        if n_unique_sets is not None:
            self.weights = np.repeat(self.weights[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            self.mu = np.repeat(self.mu[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            self.cov = np.repeat(self.cov[:n_unique_sets], n_sets // n_unique_sets, axis=0)
            # Create mapping from data index to unique set index (for EmbeddingEncoder)
            self.set_idx = np.tile(np.arange(n_unique_sets), n_sets // n_unique_sets)
        else:
            # If no unique sets specified, each set is unique
            self.set_idx = np.arange(n_sets)
        
        self.data = self.sample(
            self.weights, self.mu, self.cov, 
            n_sets, set_size, data_shape
        )
        
    def sample(self, weights, mu, cov, n_sets, set_size, data_shape):
        """
        Vectorized sampling from Gaussian Mixture Models.
        
        Args:
            weights: Mixture weights (n_sets, n_components)
            mu: Component means (n_sets, n_components, n_features)
            cov: Component covariances (n_sets, n_components, n_features, n_features)
            n_sets: Number of sets to sample from
            set_size: Number of samples per set
            data_shape: Shape of each sample
            
        Returns:
            z: Samples from the GMM (n_sets, set_size, n_features)
        """
        # Generate component assignments using cumsum trick
        # Shape: (n_sets, set_size)
        rand_uniform = np.random.rand(n_sets, set_size)  # shape: (n_sets, set_size)
        cumsum_weights = np.cumsum(weights, axis=1)  # shape: (n_sets, n_components)
        
        # Expand dimensions for broadcasting
        rand_uniform = rand_uniform[..., None]  # shape: (n_sets, set_size, 1)
        cumsum_weights = cumsum_weights[:, None, :]  # shape: (n_sets, 1, n_components)
        
        # Compare and sum to get assignments
        assignments = np.sum(rand_uniform > cumsum_weights[..., :-1], axis=2)
        
        # Generate standard normal samples for all components at once
        # Shape: (n_sets, n_components, set_size, data_dim)
        z = np.random.randn(n_sets, self.n_components, set_size, data_shape[0])
        
        # Compute Cholesky decomposition for all covariance matrices
        # Shape: (n_sets, n_components, data_dim, data_dim)
        L_cov = np.linalg.cholesky(cov)
        
        # Transform standard normal to multivariate normal for all components
        # Shape after matmul: (n_sets, n_components, set_size, data_dim)
        z = np.matmul(z, L_cov.transpose(0, 1, 3, 2))
        
        # Add means for all components
        # Shape: (n_sets, n_components, set_size, data_dim)
        z = z + mu[:, :, None, :]
        
        # Create indices for selecting samples
        batch_idx = np.arange(n_sets)[:, None]
        sample_idx = np.arange(set_size)[None, :]
        
        # Select samples according to assignments
        # Shape: (n_sets, set_size, data_dim)
        result = z[batch_idx, assignments, sample_idx]
        
        return result
    
    def wasserstein_distance(self, weights1, mu1, cov1, weights2, mu2, cov2):
        """
        Approximate Wasserstein distance between two GMMs.
        This is a simplified version that treats each component independently.
        """
        raise NotImplementedError(
            "Wasserstein distance not implemented for GMM. "
            "Computing exact Wasserstein between GMMs is challenging "
            "and typically requires numerical approximations."
        )

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        source_idx = idx
        target_idx = np.random.choice(self.n_sets, size=1, replace=False)[0]

        return {
            'source_samples': torch.tensor(self.data[source_idx], dtype=torch.float),
            'target_samples': torch.tensor(self.data[target_idx], dtype=torch.float),
            'source_idx': self.set_idx[source_idx],
            'target_idx': self.set_idx[target_idx],
        }