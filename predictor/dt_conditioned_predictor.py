import torch
import torch.nn as nn
import math
from layers import MLP


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for numerical values (like dt)."""
    
    def __init__(self, d_model, max_freq=10000):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
        # Create frequency coefficients
        half_dim = d_model // 2
        freqs = torch.exp(-math.log(max_freq) * torch.arange(half_dim) / half_dim)
        self.register_buffer('freqs', freqs)
        
    def forward(self, dt):
        """
        Args:
            dt: Tensor of shape [batch_size] or [batch_size, 1]
        Returns:
            Positional encodings of shape [batch_size, d_model]
        """
        if dt.dim() == 1:
            dt = dt.unsqueeze(-1)  # [batch_size, 1]
            
        # Apply sinusoidal encoding
        args = dt * self.freqs[None, :]  # [batch_size, half_dim]
        
        # Concatenate sin and cos
        if self.d_model % 2 == 0:
            encodings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        else:
            # Handle odd d_model by adding one more sin
            encodings = torch.cat([
                torch.sin(args), 
                torch.cos(args), 
                torch.sin(args[:, :1])
            ], dim=-1)
            
        return encodings


class DTConditionedMLPPredictor(nn.Module):
    """MLP predictor conditioned on dt using various flexible approaches."""
    
    # Class attribute to indicate this predictor requires dt
    requires_dt = True
    
    def __init__(
        self, 
        latent_dim, 
        hidden_dim=128, 
        num_layers=2,
        conditioning_mode="sinusoidal",  # "sinusoidal", "concat", "film"
        dt_embed_dim=16,
        normalize_dt=True
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.conditioning_mode = conditioning_mode
        self.normalize_dt = normalize_dt
        self.dt_embed_dim = dt_embed_dim
        
        # Running statistics for dt normalization (will be updated dynamically)
        self.register_buffer('dt_mean', torch.tensor(0.0))
        self.register_buffer('dt_std', torch.tensor(1.0))
        self.register_buffer('dt_count', torch.tensor(0))
        
        if conditioning_mode == "sinusoidal":
            # Sinusoidal positional encoding for dt
            self.dt_encoder = SinusoidalPositionalEncoding(dt_embed_dim)
            input_dim = latent_dim + dt_embed_dim
            
            self.net = MLP(
                in_dims=input_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers
            )
            
        elif conditioning_mode == "concat":
            # Simple concatenation (with optional normalization)
            input_dim = latent_dim + 1  # +1 for dt value
            
            self.net = MLP(
                in_dims=input_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers
            )
            
        elif conditioning_mode == "film":
            # FiLM conditioning: dt generates scale and shift parameters
            self.dt_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim * 2)  # *2 for scale and shift
            )
            
            self.net = MLP(
                in_dims=latent_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers
            )
            
        else:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")
        
        self.latent_act = nn.SELU()
        self.similarity = nn.CosineSimilarity()
    
    def _update_dt_stats(self, dt_batch):
        """Update running statistics for dt normalization."""
        if not self.training:
            return
            
        batch_mean = dt_batch.float().mean()
        batch_var = dt_batch.float().var(unbiased=False)
        batch_count = dt_batch.numel()
        
        # Online update of running statistics
        new_count = self.dt_count + batch_count
        delta = batch_mean - self.dt_mean
        
        new_mean = self.dt_mean + delta * batch_count / new_count
        m_a = self.dt_std ** 2 * self.dt_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.dt_count * batch_count / new_count
        new_std = torch.sqrt(M2 / new_count)
        
        self.dt_mean.copy_(new_mean)
        self.dt_std.copy_(torch.clamp(new_std, min=1e-6))  # Avoid division by zero
        self.dt_count.copy_(new_count)
    
    def _normalize_dt(self, dt):
        """Normalize dt using running statistics."""
        if self.normalize_dt:
            return (dt.float() - self.dt_mean) / self.dt_std
        return dt.float()
    
    def forward(self, x, dt):
        """
        Args:
            x: Source latent [batch_size, latent_dim]
            dt: Time difference [batch_size] or [batch_size, 1]
        """
        # Update dt statistics during training
        self._update_dt_stats(dt)
        
        # Normalize dt
        dt_norm = self._normalize_dt(dt)
        
        if self.conditioning_mode == "sinusoidal":
            # Encode dt using sinusoidal encoding
            dt_encoded = self.dt_encoder(dt_norm)  # [batch_size, dt_embed_dim]
            # Concatenate with latent
            x_conditioned = torch.cat([x, dt_encoded], dim=-1)
            output = self.net(x_conditioned)
            
        elif self.conditioning_mode == "concat":
            # Simple concatenation
            if dt_norm.dim() == 1:
                dt_norm = dt_norm.unsqueeze(-1)
            x_conditioned = torch.cat([x, dt_norm], dim=-1)
            output = self.net(x_conditioned)
            
        elif self.conditioning_mode == "film":
            # FiLM conditioning
            if dt_norm.dim() == 1:
                dt_norm = dt_norm.unsqueeze(-1)
            film_params = self.dt_encoder(dt_norm)  # [batch_size, latent_dim * 2]
            scale, shift = film_params.chunk(2, dim=-1)  # Each [batch_size, latent_dim]
            
            # Apply FiLM modulation to input
            x_modulated = x * (1 + scale) + shift
            output = self.net(x_modulated)
        
        return self.latent_act(output)
    
    def loss(self, source_latent, target_latent, dt):
        """Loss function that takes dt into account."""
        pred_target_latent = self.forward(source_latent, dt)
        return (1 - self.similarity(pred_target_latent, target_latent)).mean(), pred_target_latent


class DTConditionedRidgePredictor(nn.Module):
    """Ridge regression predictor conditioned on dt."""
    
    # Class attribute to indicate this predictor requires dt
    requires_dt = True
    
    def __init__(
        self, 
        latent_dim, 
        ridge_alpha=1e-3,
        conditioning_mode="sinusoidal",  # "sinusoidal", "concat", "film"
        dt_embed_dim=16,
        normalize_dt=True,
        film_hidden_dim=64
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.ridge_alpha = ridge_alpha
        self.conditioning_mode = conditioning_mode
        self.normalize_dt = normalize_dt
        
        # Running statistics for dt normalization
        self.register_buffer('dt_mean', torch.tensor(0.0))
        self.register_buffer('dt_std', torch.tensor(1.0))
        self.register_buffer('dt_count', torch.tensor(0))
        
        if conditioning_mode == "sinusoidal":
            self.dt_encoder = SinusoidalPositionalEncoding(dt_embed_dim)
            input_dim = latent_dim + dt_embed_dim
            self.linear = nn.Linear(input_dim, latent_dim)
            
        elif conditioning_mode == "concat":
            input_dim = latent_dim + 1
            self.linear = nn.Linear(input_dim, latent_dim)
            
        elif conditioning_mode == "film":
            # Use dt to generate bias terms for the linear layer
            self.dt_encoder = nn.Sequential(
                nn.Linear(1, film_hidden_dim),
                nn.ReLU(),
                nn.Linear(film_hidden_dim, latent_dim)  # Generate bias terms
            )
            self.linear = nn.Linear(latent_dim, latent_dim)
            
        else:
            raise ValueError(f"Unknown conditioning_mode: {conditioning_mode}")
        
        self.latent_act = nn.SELU()
        self.similarity = nn.CosineSimilarity()
    
    def _update_dt_stats(self, dt_batch):
        """Update running statistics for dt normalization."""
        if not self.training:
            return
            
        batch_mean = dt_batch.float().mean()
        batch_var = dt_batch.float().var(unbiased=False)
        batch_count = dt_batch.numel()
        
        new_count = self.dt_count + batch_count
        delta = batch_mean - self.dt_mean
        
        new_mean = self.dt_mean + delta * batch_count / new_count
        m_a = self.dt_std ** 2 * self.dt_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.dt_count * batch_count / new_count
        new_std = torch.sqrt(M2 / new_count)
        
        self.dt_mean.copy_(new_mean)
        self.dt_std.copy_(torch.clamp(new_std, min=1e-6))
        self.dt_count.copy_(new_count)
    
    def _normalize_dt(self, dt):
        """Normalize dt using running statistics."""
        if self.normalize_dt:
            return (dt.float() - self.dt_mean) / self.dt_std
        return dt.float()
    
    def forward(self, x, dt):
        """
        Args:
            x: Source latent [batch_size, latent_dim]
            dt: Time difference [batch_size] or [batch_size, 1]
        """
        # Update dt statistics during training
        self._update_dt_stats(dt)
        
        # Normalize dt
        dt_norm = self._normalize_dt(dt)
        
        if self.conditioning_mode == "sinusoidal":
            dt_encoded = self.dt_encoder(dt_norm)
            x_conditioned = torch.cat([x, dt_encoded], dim=-1)
            output = self.linear(x_conditioned)
            
        elif self.conditioning_mode == "concat":
            if dt_norm.dim() == 1:
                dt_norm = dt_norm.unsqueeze(-1)
            x_conditioned = torch.cat([x, dt_norm], dim=-1)
            output = self.linear(x_conditioned)
            
        elif self.conditioning_mode == "film":
            if dt_norm.dim() == 1:
                dt_norm = dt_norm.unsqueeze(-1)
            dt_bias = self.dt_encoder(dt_norm)  # [batch_size, latent_dim]
            output = self.linear(x) + dt_bias
        
        return self.latent_act(output)
    
    def loss(self, source_latent, target_latent, dt):
        """Loss function with L2 regularization."""
        pred_target_latent = self.forward(source_latent, dt)
        loss = (1 - self.similarity(pred_target_latent, target_latent)).mean()
        
        if self.ridge_alpha > 0:
            # Add L2 regularization on linear layer weights
            loss += self.ridge_alpha * torch.sum(self.linear.weight ** 2)
            
        return loss, pred_target_latent 