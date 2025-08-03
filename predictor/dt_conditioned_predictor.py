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
        # Convert to float to avoid dtype mismatch
        dt = dt.float()
        
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
        self.dt_embed_dim = dt_embed_dim
        
        
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
    
    
    def forward(self, x, dt):
        """
        Args:
            x: Source latent [batch_size, latent_dim]
            dt: Time difference [batch_size] or [batch_size, 1]
        """
        # Convert to float to avoid dtype mismatch
        dt = dt.float()
        
        if self.conditioning_mode == "sinusoidal":
            # Encode dt using sinusoidal encoding
            dt_encoded = self.dt_encoder(dt)  # [batch_size, dt_embed_dim]
            # Concatenate with latent
            x_conditioned = torch.cat([x, dt_encoded], dim=-1)
            output = self.net(x_conditioned)
            
        elif self.conditioning_mode == "concat":
            # Simple concatenation
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            x_conditioned = torch.cat([x, dt], dim=-1)
            output = self.net(x_conditioned)
            
        elif self.conditioning_mode == "film":
            # FiLM conditioning
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            film_params = self.dt_encoder(dt)  # [batch_size, latent_dim * 2]
            scale, shift = film_params.chunk(2, dim=-1)  # Each [batch_size, latent_dim]
            
            # Apply FiLM modulation to input
            x_modulated = x * (1 + scale) + shift
            output = self.net(x_modulated)
        
        return self.latent_act(output)
    
    def loss(self, source_latent, target_latent, dt):
        """Loss function that takes dt into account."""
        pred_target_latent = self.forward(source_latent, dt)
        # TODO: should one include an absolute value to make sure this is never negative?
        return (1 - self.similarity(pred_target_latent, target_latent)).mean(), pred_target_latent


class DTConditionedRidgePredictor(nn.Module):
    """Ridge regression predictor conditioned on dt."""
    
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
    
    
    def forward(self, x, dt):
        """
        Args:
            x: Source latent [batch_size, latent_dim]
            dt: Time difference [batch_size] or [batch_size, 1]
        """
        # Convert to float to avoid dtype mismatch
        dt = dt.float()
        
        if self.conditioning_mode == "sinusoidal":
            dt_encoded = self.dt_encoder(dt)
            x_conditioned = torch.cat([x, dt_encoded], dim=-1)
            output = self.linear(x_conditioned)
            
        elif self.conditioning_mode == "concat":
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            x_conditioned = torch.cat([x, dt], dim=-1)
            output = self.linear(x_conditioned)
            
        elif self.conditioning_mode == "film":
            if dt.dim() == 1:
                dt = dt.unsqueeze(-1)
            dt_bias = self.dt_encoder(dt)  # [batch_size, latent_dim]
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