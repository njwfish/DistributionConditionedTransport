import torch
import torch.nn as nn
from generator.losses import sliced_wasserstein_distance, mmd
from geomloss import SamplesLoss

class DirectGenerator(nn.Module):
    def __init__(self, model, loss_type='swd', loss_params=None, noise_dim=100, flatten_for_model=True):
        """
        Args:
            model: Neural network model
            loss_type: 'swd', 'mmd', or 'sinkhorn'
            loss_params: Parameters for loss function
            noise_dim: Not used but kept for compatibility
            flatten_for_model: If True, flatten spatial dims before model (for MLP).
                              If False, preserve shape (for UNet/CNN).
        """
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.flatten_for_model = flatten_for_model

        def loss_fn(x, y):
            # Flatten spatial dimensions for loss computation: [batch, samples, ...] -> [batch, samples, -1]
            x_flat = x.reshape(x.shape[0], x.shape[1], -1)
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)

            if self.loss_type == 'swd':
                return torch.vmap(sliced_wasserstein_distance, randomness='different')(x_flat, y_flat, **self.loss_params).mean()
            elif self.loss_type == 'mmd':
                # Default to energy kernel (parameter-free), but allow overriding via loss_params
                return torch.vmap(mmd, randomness='different')(x_flat, y_flat, **self.loss_params).mean()
            elif self.loss_type == 'sinkhorn':
                sinkhorn = SamplesLoss("sinkhorn", p=2, scaling=0.9)
                return sinkhorn(x_flat, y_flat).mean()

        self.loss_fn = loss_fn

    def forward(self, source_samples, source_latent, target_latent, treat_cond):
        """
        Transport source samples to target distribution using latent embeddings
        Args:
            source_samples: [batch, samples, ...] - samples from source distribution (can be images)
            source_latent: [batch, samples, latent_dim] - source distribution embedding  
            target_latent: [batch, samples, latent_dim] - target distribution embedding
            treat_cond: [batch, samples, treat_dim] - treatment conditioning
        """
        return self.model(source_samples, source_latent, target_latent, treat_cond)

    def loss(self, source_samples, target_samples, source_latent, target_latent, treat_cond):
        """
        Compute transport loss between source and target distributions
        Args:
            source_samples: [N, ...] samples from source distribution
            target_samples: [N, ...] samples from target distribution  
            source_latent: [batch, latent_dim] source distribution embedding
            target_latent: [batch, latent_dim] target distribution embedding
            treat_cond: [batch, treat_dim] treatment conditioning
        """
        samples_per_set = target_samples.shape[0] // target_latent.shape[0]
        data_shape = source_samples.shape[1:]  # Preserve original shape (e.g., C, H, W for images)

        # Reshape: [N, ...] -> [batch, samples_per_set, ...]
        source_samples = source_samples.view(source_latent.shape[0], samples_per_set, *data_shape)
        target_samples = target_samples.view(target_latent.shape[0], samples_per_set, *data_shape)
        
        # Expand latents: [batch, latent_dim] -> [batch, samples_per_set, latent_dim]
        source_latent = source_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        target_latent = target_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        
        # Expand treat_cond: [batch, treat_dim] -> [batch, samples_per_set, treat_dim]
        treat_cond = treat_cond.unsqueeze(1).expand(-1, samples_per_set, -1)
        
        if self.flatten_for_model:
            # For MLP: flatten spatial dims
            source_samples_model = source_samples.reshape(source_samples.shape[0], samples_per_set, -1)
        else:
            # For UNet/CNN: merge batch and samples dims, preserve spatial
            batch_size = source_samples.shape[0]
            source_samples_model = source_samples.reshape(batch_size * samples_per_set, *data_shape)
            source_latent = source_latent.reshape(batch_size * samples_per_set, -1)
            target_latent = target_latent.reshape(batch_size * samples_per_set, -1)
            treat_cond = treat_cond.reshape(batch_size * samples_per_set, -1)
        
        # Generate transported samples
        transported = self.forward(source_samples_model, source_latent, target_latent, treat_cond)
        
        if not self.flatten_for_model:
            # Reshape back: [batch * samples, ...] -> [batch, samples, ...]
            transported = transported.reshape(batch_size, samples_per_set, *data_shape)
        
        return self.loss_fn(transported, target_samples)
    
    def sample(self, source_samples, source_latent, target_latent, treat_cond):
        """Generate samples by transporting from source to target distribution
        
        Args:
            source_samples: [N, ...] samples from source distribution
            source_latent: [num_sets, latent_dim] source distribution embedding
            target_latent: [num_sets, latent_dim] target distribution embedding
            treat_cond: [num_sets, treat_dim] treatment conditioning
        """
        num_samples, *data_shape = source_samples.shape
        num_sets = source_latent.shape[0]
        samples_per_set = num_samples // num_sets

        # Reshape source samples
        source_samples = source_samples.view(num_sets, samples_per_set, *data_shape)
        source_latent = source_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        target_latent = target_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        
        # Expand treat_cond: [num_sets, treat_dim] -> [num_sets, samples_per_set, treat_dim]
        treat_cond = treat_cond.unsqueeze(1).expand(-1, samples_per_set, -1)

        if self.flatten_for_model:
            source_samples_model = source_samples.reshape(num_sets, samples_per_set, -1)
        else:
            source_samples_model = source_samples.reshape(num_sets * samples_per_set, *data_shape)
            source_latent = source_latent.reshape(num_sets * samples_per_set, -1)
            target_latent = target_latent.reshape(num_sets * samples_per_set, -1)
            treat_cond = treat_cond.reshape(num_sets * samples_per_set, -1)

        generated = self.forward(source_samples_model, source_latent, target_latent, treat_cond)
        
        if not self.flatten_for_model:
            generated = generated.reshape(num_sets, samples_per_set, *data_shape)
        else:
            generated = generated.reshape(num_sets, samples_per_set, *data_shape)
            
        return generated
    
    