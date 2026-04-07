import math
import torch
import torch.nn as nn
from generator.losses import sliced_wasserstein_distance, mmd
from geomloss import SamplesLoss

ANNEAL_SCHEDULES = {
    'linear': lambda p: 1.0 - p,
    'cosine': lambda p: 0.5 * (1.0 + math.cos(math.pi * p)),
    'step': lambda p: 1.0 if p < 0.5 else 2.0 * (1.0 - p),
}

class DirectGenerator(nn.Module):
    def __init__(self, model, loss_type='swd', loss_params=None, noise_dim=100, flatten_for_model=True,
                 coupling_weight=0.0, coupling_anneal_steps=0, coupling_anneal_schedule='linear'):
        """
        Args:
            model: Neural network model
            loss_type: 'swd', 'mmd', or 'sinkhorn'
            loss_params: Parameters for loss function
            noise_dim: Not used but kept for compatibility
            flatten_for_model: If True, flatten spatial dims before model (for MLP).
                              If False, preserve shape (for UNet/CNN).
            coupling_weight: Weight for coupling penalty ||x_src - x_pred||^2.
            coupling_anneal_steps: If > 0, anneal coupling_weight to 0
                                   over this many loss() calls. 0 = constant weight.
            coupling_anneal_schedule: 'linear', 'cosine', or 'step'.
        """
        super().__init__()
        self.model = model
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.flatten_for_model = flatten_for_model
        self.coupling_weight = coupling_weight
        self.coupling_anneal_steps = coupling_anneal_steps
        self.coupling_anneal_schedule = coupling_anneal_schedule
        # self.register_buffer('_step', torch.tensor(0, dtype=torch.long))

        if coupling_weight <= 0:
            self._coupling_schedule = lambda: 0.0
        elif coupling_anneal_steps <= 0:
            self._coupling_schedule = lambda: coupling_weight
        else:
            anneal_fn = ANNEAL_SCHEDULES[coupling_anneal_schedule]
            self._coupling_schedule = lambda: coupling_weight * anneal_fn(
                min(self._step.item() / self.coupling_anneal_steps, 1.0)
            )

        def loss_fn(x, y):
            # Flatten spatial dimensions for loss computation: [batch, samples, ...] -> [batch, samples, -1]
            x_flat = x.reshape(x.shape[0], x.shape[1], -1)
            y_flat = y.reshape(y.shape[0], y.shape[1], -1)

            if self.loss_type == 'swd':
                return torch.vmap(sliced_wasserstein_distance, randomness='different')(x_flat, y_flat, **self.loss_params).mean()
            elif self.loss_type == 'mmd':
                return torch.vmap(mmd, randomness='different')(x_flat, y_flat, **self.loss_params).mean()
            elif self.loss_type == 'sinkhorn':
                sinkhorn = SamplesLoss("sinkhorn", p=2, scaling=0.9)
                return sinkhorn(x_flat, y_flat).mean()

        self.loss_fn = loss_fn


    def forward(self, source_samples, source_latent, target_latent):
        """
        Transport source samples to target distribution using latent embeddings
        Args:
            source_samples: [batch, samples, ...] - samples from source distribution (can be images)
            source_latent: [batch, samples, latent_dim] - source distribution embedding
            target_latent: [batch, samples, latent_dim] - target distribution embedding
        """
        return self.model(source_samples, source_latent, target_latent)

    def loss(self, source_samples, target_samples, source_latent, target_latent):
        """
        Compute transport loss between source and target distributions
        Args:
            source_samples: [N, ...] samples from source distribution
            target_samples: [N, ...] samples from target distribution
            source_latent: [batch, latent_dim] source distribution embedding
            target_latent: [batch, latent_dim] target distribution embedding
        """
        samples_per_set = target_samples.shape[0] // target_latent.shape[0]
        data_shape = source_samples.shape[1:]  # Preserve original shape (e.g., C, H, W for images)

        # Reshape: [N, ...] -> [batch, samples_per_set, ...]
        source_samples = source_samples.view(source_latent.shape[0], samples_per_set, *data_shape)
        target_samples = target_samples.view(target_latent.shape[0], samples_per_set, *data_shape)

        # Expand latents: [batch, latent_dim] -> [batch, samples_per_set, latent_dim]
        source_latent = source_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        target_latent = target_latent.unsqueeze(1).expand(-1, samples_per_set, -1)

        if self.flatten_for_model:
            # For MLP: flatten spatial dims
            source_samples_model = source_samples.reshape(source_samples.shape[0], samples_per_set, -1)
        else:
            # For UNet/CNN: merge batch and samples dims, preserve spatial
            batch_size = source_samples.shape[0]
            source_samples_model = source_samples.reshape(batch_size * samples_per_set, *data_shape)
            source_latent = source_latent.reshape(batch_size * samples_per_set, -1)
            target_latent = target_latent.reshape(batch_size * samples_per_set, -1)

        # Generate transported samples
        transported = self.forward(source_samples_model, source_latent, target_latent)

        if not self.flatten_for_model:
            # Reshape back: [batch * samples, ...] -> [batch, samples, ...]
            transported = transported.reshape(batch_size, samples_per_set, *data_shape)

        loss = self.loss_fn(transported, target_samples)

        w = self._coupling_schedule()
        if w > 0:
            src_flat = source_samples.reshape(source_samples.shape[0], source_samples.shape[1], -1)
            trn_flat = transported.reshape(transported.shape[0], transported.shape[1], -1)
            loss = loss + w * (src_flat - trn_flat).pow(2).mean()

        if self.training:
            self._step += 1  # persisted via register_buffer → state_dict

        return loss

    def sample(self, source_samples, source_latent, target_latent):
        """Generate samples by transporting from source to target distribution"""
        num_samples, *data_shape = source_samples.shape
        num_sets = source_latent.shape[0]
        samples_per_set = num_samples // num_sets

        # Reshape source samples
        source_samples = source_samples.view(num_sets, samples_per_set, *data_shape)
        source_latent = source_latent.unsqueeze(1).expand(-1, samples_per_set, -1)
        target_latent = target_latent.unsqueeze(1).expand(-1, samples_per_set, -1)

        if self.flatten_for_model:
            source_samples_model = source_samples.reshape(num_sets, samples_per_set, -1)
        else:
            source_samples_model = source_samples.reshape(num_sets * samples_per_set, *data_shape)
            source_latent = source_latent.reshape(num_sets * samples_per_set, -1)
            target_latent = target_latent.reshape(num_sets * samples_per_set, -1)

        generated = self.forward(source_samples_model, source_latent, target_latent)

        if not self.flatten_for_model:
            generated = generated.reshape(num_sets, samples_per_set, *data_shape)
        else:
            generated = generated.reshape(num_sets, samples_per_set, *data_shape)

        return generated
