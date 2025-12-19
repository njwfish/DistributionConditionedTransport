import torch
import torch.nn as nn
from torchdyn.core import NeuralODE
from utils.latents import expand_latent_to_batch

class TorchWrapper(nn.Module):
    """Wrapper to make the model compatible with NeuralODE"""
    def __init__(self, model, source_latent, target_latent):
        super().__init__()
        self.model = model
        self.source_latent = source_latent
        self.target_latent = target_latent
    
    def forward(self, t, x, args={}):
        """NeuralODE expects (t, x) -> dx/dt"""
        # Expand time to match batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0], 1)
        elif t.dim() == 1 and t.shape[0] == 1:
            t = t.expand(x.shape[0], 1)
        else:
            t = t.unsqueeze(-1) if t.dim() == 1 else t
            
        # Pass target_latent only if available
        if self.target_latent is None:
            return self.model(x, t, self.source_latent)
        else:
            return self.model(x, t, self.source_latent, self.target_latent)


class FlowMatchingGenerator(nn.Module):
    def __init__(
        self, 
        model, 
        sigma=0.3
    ):
        """
        Flow Matching Generator for coupled distribution embeddings.
        
        Args:
            model: Neural network that predicts velocity field v_t(x_t, t, source_latent, target_latent)
            sigma: Noise level for numerical stability
        """
        super().__init__()
        self.model = model
        self.sigma = sigma

    def sample_time(self, batch_size, device):
        """Sample random times uniformly from [0, 1]"""
        return torch.rand(batch_size, device=device)

    def compute_sigma_t(self, t):
        return self.sigma

    def interpolant(self, source_samples, target_samples, t):
        """
        Linear interpolant between source and target samples
        x_t = (1 - t) * x_0 + t * x_1 + sigma_t * epsilon
        
        Args:
            source_samples: [batch_size, ...] samples from source distribution
            target_samples: [batch_size, ...] samples from target distribution  
            t: [batch_size] time values in [0, 1]
        """
        # Reshape t to broadcast properly
        t = t.view(-1, *([1] * (source_samples.ndim - 1)))
        
        # Linear interpolation
        x_t = (1 - t) * source_samples + t * target_samples
        
        epsilon = torch.randn_like(x_t)
        sigma_t = self.compute_sigma_t(t)
        x_t = x_t + sigma_t * epsilon
        
        return x_t

    def compute_conditional_flow(self, source_samples, target_samples, t):
        """
        Compute the target velocity field v_t = d/dt x_t
        For linear interpolant: v_t = x_1 - x_0 - sigma_min * epsilon (approximately)
        """
        velocity = target_samples - source_samples
        return velocity

    def forward(self, source_samples, source_latent, target_latent, num_steps=100, return_trajectory=False):
        """
        Generate samples by integrating the learned flow from source to target using NeuralODE
        
        Args:
            source_samples: [batch_size, ...] starting points
            source_latent: [batch_size, latent_dim] source distribution embedding
            target_latent: [batch_size, latent_dim] target distribution embedding
            num_steps: number of time steps for integration
            return_trajectory: whether to return full trajectory or just final point
        """
        batch_size = source_samples.shape[0]
        
        # Expand latents to match the number of source samples
        source_latent = expand_latent_to_batch(source_latent, source_samples)
        target_latent = expand_latent_to_batch(target_latent, source_samples)
        
        # Create wrapped model for NeuralODE
        wrapped_model = TorchWrapper(self.model, source_latent, target_latent)
        
        # Create NeuralODE node
        node = NeuralODE(
            wrapped_model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4
        )
        
        with torch.no_grad():
            # Return only final point
            traj = node.trajectory(
                source_samples,
                t_span=torch.linspace(0, 1, num_steps, device=source_samples.device),  # Just start and end
            )
            if return_trajectory:
                # Return full trajectory
                return traj
            else:
                return traj[-1]  # Return the last point

    def loss(self, source_samples, target_samples, source_latent, target_latent):
        """
        Flow matching loss: ||v_theta(x_t, t) - v_t||^2
        
        Args:
            source_samples: samples from source distribution
            target_samples: samples from target distribution
            source_latent: source distribution embedding  
            target_latent: target distribution embedding
        """
        
        batch_size, set_size = source_samples.shape[:2]

        source_latent = expand_latent_to_batch(source_latent, source_samples)
        target_latent = expand_latent_to_batch(target_latent, target_samples)

        # Sample random time
        t = self.sample_time(batch_size, source_samples.device)
        
        # Create interpolated samples
        x_t = self.interpolant(source_samples, target_samples, t)
        
        # True velocity field
        v_true = self.compute_conditional_flow(source_samples, target_samples, t)
        
        # Predicted velocity field
        if target_latent is None:
            v_pred = self.model(x_t, t.unsqueeze(-1), source_latent)
        else:
            v_pred = self.model(x_t, t.unsqueeze(-1), source_latent, target_latent)
        
        # MSE loss
        loss = torch.mean((v_pred - v_true) ** 2)
        
        return loss

    def sample(self, source_samples, source_latent, target_latent, num_steps=100, return_trajectory=False):
        """
        Generate samples by integrating the flow
        
        Args:
            source_samples: [batch_size, ...] starting points
            source_latent: [batch_size, latent_dim] source embedding
            target_latent: [batch_size, latent_dim] target embedding
            num_steps: number of integration steps
            return_trajectory: whether to return full trajectory
        """
        
        num_samples = source_samples.shape[0] // source_latent.shape[0]
        generated = self.forward(source_samples, source_latent, target_latent, num_steps, return_trajectory)
        
        if return_trajectory:
            # Reshape trajectory: [time_steps, batch_size, ...] -> [time_steps, num_sets, num_samples, ...]
            return generated.reshape(generated.shape[0], source_latent.shape[0], num_samples, *source_samples.shape[1:])
        else:
            # Reshape final result: [batch_size, ...] -> [num_sets, num_samples, ...]
            return generated.reshape(source_latent.shape[0], num_samples, *source_samples.shape[1:])