import torch
import torch.nn as nn
from generator.losses import sliced_wasserstein_distance, mmd
from geomloss import SamplesLoss

class DirectGenerator(nn.Module):
    def __init__(self, model, loss_type='swd', loss_params=None, noise_dim=100):
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.loss_params = loss_params

        def loss_fn(x, y):
            if self.loss_type == 'swd':
                return torch.vmap(sliced_wasserstein_distance, randomness='different')(x, y, **self.loss_params).mean()
            elif self.loss_type == 'mmd':
                return torch.vmap(mmd, randomness='different')(x, y).mean()
            elif self.loss_type == 'sinkhorn':
                # return torch.vmap(sinkhorn, randomness='different')(x, y, **self.loss_params).mean()
                sinkhorn = SamplesLoss("sinkhorn", p=2, scaling=0.9)
                return sinkhorn(x, y).mean()
            
        self.loss_fn = loss_fn

    def forward(self, source_samples, source_latent, target_latent):
        """
        Transport source samples to target distribution using latent embeddings
        Args:
            source_samples: [batch_size, source_dim] - samples from source distribution
            source_latent: [batch_size, latent_dim] - source distribution embedding  
            target_latent: [batch_size, latent_dim] - target distribution embedding
            num_samples: number of samples to generate per set
        """
        return self.model(source_samples, source_latent, target_latent)

    def loss(self, source_samples, target_samples, source_latent, target_latent):
        """
        Compute transport loss between source and target distributions
        Args:
            source_samples: samples from source distribution
            target_samples: samples from target distribution  
            source_latent: source distribution embedding
            target_latent: target distribution embedding
        """
        samples_per_set = target_samples.shape[0] // target_latent.shape[0]

        source_samples = source_samples.view(source_latent.shape[0], samples_per_set, -1)
        source_latent = source_latent.unsqueeze(1).repeat(1, samples_per_set, 1)
        target_latent = target_latent.unsqueeze(1).repeat(1, samples_per_set, 1)
        
        # Generate transported samples
        transported = self.forward(
            source_samples,  
            source_latent, 
            target_latent, 
        )
        
        # Reshape for comparison
        target_samples = target_samples.reshape(transported.shape)
        
        return self.loss_fn(transported, target_samples)
    
    def sample(self, source_samples, source_latent, target_latent):
        """Generate samples by transporting from source to target distribution"""
        num_samples = source_samples.shape[0]
        samples_per_set = num_samples // source_latent.shape[0]

        source_samples = source_samples.view(source_latent.shape[0], samples_per_set, -1)
        source_latent = source_latent.unsqueeze(1).repeat(1, samples_per_set, 1)
        target_latent = target_latent.unsqueeze(1).repeat(1, samples_per_set, 1)

        generated = self.forward(source_samples, source_latent, target_latent)
        return generated.reshape(num_samples, samples_per_set, *source_samples.shape[1:])
    
    