import torch
import torch.nn as nn
from torchdyn.core import NeuralODE

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
            
        return self.model(x, t, self.source_latent, self.target_latent)

class NeuralNetworkMapping(nn.Module):
    """Multi-layer perceptron for latent mapping"""
    def __init__(self, latent_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def regularization_loss(self):
        """No additional regularization for neural network"""
        return 0.0

class RidgeRegressionMapping(nn.Module):
    """Ridge regression (linear mapping with L2 regularization)"""
    def __init__(self, latent_dim, ridge_alpha=1e-3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.ridge_alpha = ridge_alpha
        
    def forward(self, x):
        return self.linear(x)
    
    def regularization_loss(self):
        """L2 regularization on weights"""
        return self.ridge_alpha * torch.sum(self.linear.weight ** 2)

class LinearMapping(nn.Module):
    """Simple linear transformation without regularization"""
    def __init__(self, latent_dim):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        
    def forward(self, x):
        return self.linear(x)
    
    def regularization_loss(self):
        """No regularization for simple linear mapping"""
        return 0.0

class FlowMatchingGenerator(nn.Module):
    def __init__(self, model, sigma_min=1e-4, learn_target_mapping=False, latent_dim=None, 
                 hidden_dim=128, mapping_method="neural_network", ridge_alpha=1e-3):
        """
        Flow Matching Generator for coupled distribution embeddings.
        
        Args:
            model: Neural network that predicts velocity field v_t(x_t, t, source_latent, target_latent)
            sigma_min: Minimum noise level for numerical stability
            learn_target_mapping: If True, learn mapping from source_latent to target_latent
            latent_dim: Dimension of latent embeddings (required if learn_target_mapping=True and mapping_method != "identity")
            hidden_dim: Hidden dimension for the neural network mapping
            mapping_method: Method for latent mapping ("neural_network", "ridge", "linear", "identity")
            ridge_alpha: Regularization strength for ridge regression
        """
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min
        self.learn_target_mapping = learn_target_mapping
        self.mapping_method = mapping_method
        
        if self.learn_target_mapping:
            # Choose the appropriate mapping method
            if mapping_method == "identity":
                # No target mapping network needed - we'll just use source_latent as target_latent
                self.target_mapping_net = None
            elif mapping_method == "neural_network":
                if latent_dim is None:
                    raise ValueError("latent_dim must be provided when using neural_network mapping")
                self.target_mapping_net = NeuralNetworkMapping(latent_dim, hidden_dim)
            elif mapping_method == "ridge":
                if latent_dim is None:
                    raise ValueError("latent_dim must be provided when using ridge mapping")
                self.target_mapping_net = RidgeRegressionMapping(latent_dim, ridge_alpha)
            elif mapping_method == "linear":
                if latent_dim is None:
                    raise ValueError("latent_dim must be provided when using linear mapping")
                self.target_mapping_net = LinearMapping(latent_dim)
            else:
                raise ValueError(f"Unknown mapping_method: {mapping_method}. "
                               "Choose from 'neural_network', 'ridge', 'linear', or 'identity'")
        else:
            self.target_mapping_net = None

    def sample_time(self, batch_size, device):
        """Sample random times uniformly from [0, 1]"""
        return torch.rand(batch_size, device=device)

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
        
        # Add small amount of noise for numerical stability
        if self.sigma_min > 0:
            epsilon = torch.randn_like(x_t)
            sigma_t = self.sigma_min * (1 - t)
            x_t = x_t + sigma_t * epsilon
            
        return x_t

    def velocity_field(self, source_samples, target_samples, t):
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
            target_latent: [batch_size, latent_dim] target distribution embedding (ignored if learn_target_mapping=True)
            num_steps: number of time steps for integration
            return_trajectory: whether to return full trajectory or just final point
        """
        batch_size = source_samples.shape[0]
        
        # Generate target_latent from source_latent if learn_target_mapping is enabled
        if self.learn_target_mapping:
            if self.target_mapping_net is None:
                target_latent = source_latent # Use source_latent as target_latent for identity mapping
            else:
                target_latent = self.target_mapping_net(source_latent)
        
        # Expand latents to match the number of source samples
        source_latent = source_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // source_latent.shape[0], 1).view(-1, source_latent.shape[-1])
        target_latent = target_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // target_latent.shape[0], 1).view(-1, target_latent.shape[-1])
        
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
                t_span=torch.linspace(0, 1, 2, device=source_samples.device),  # Just start and end
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
            target_latent: target distribution embedding (ignored if learn_target_mapping=True)
        """
        batch_size, set_size = source_samples.shape

        # Generate target_latent from source_latent if learn_target_mapping is enabled
        if self.learn_target_mapping:
            if self.target_mapping_net is None:
                target_latent = source_latent # Use source_latent as target_latent for identity mapping
            else:
                target_latent = self.target_mapping_net(source_latent)

        source_latent = source_latent.unsqueeze(1).repeat(1, source_samples.shape[0] // source_latent.shape[0], 1).view(-1, source_latent.shape[-1]) 
        target_latent = target_latent.unsqueeze(1).repeat(1, target_samples.shape[0] // target_latent.shape[0], 1).view(-1, target_latent.shape[-1]) 
        
        # Sample random time
        t = self.sample_time(batch_size, source_samples.device)
        
        # Create interpolated samples
        x_t = self.interpolant(source_samples, target_samples, t)
        
        # True velocity field
        v_true = self.velocity_field(source_samples, target_samples, t)
        
        # Predicted velocity field
        v_pred = self.model(x_t, t.unsqueeze(-1), source_latent, target_latent)
        
        # MSE loss
        loss = torch.mean((v_pred - v_true) ** 2)
        
        # Add regularization loss from target mapping network if applicable
        if self.learn_target_mapping and hasattr(self.target_mapping_net, 'regularization_loss'):
            loss = loss + self.target_mapping_net.regularization_loss()
        
        return loss

    def sample(self, source_samples, source_latent, target_latent, num_steps=100, return_trajectory=False):
        """
        Generate samples by integrating the flow
        
        Args:
            source_samples: [batch_size, ...] starting points
            source_latent: [batch_size, latent_dim] source embedding
            target_latent: [batch_size, latent_dim] target embedding (ignored if learn_target_mapping=True)
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