import torch
import torch.nn as nn
import math
from layers import MLP


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for numerical values (like d)."""
    
    def __init__(self, d_embed_dim, max_freq=10000):
        super().__init__()
        self.d_embed_dim = d_embed_dim
        self.max_freq = max_freq
        
        # Create frequency coefficients
        half_dim = self.d_embed_dim // 2
        freqs = torch.exp(-math.log(max_freq) * torch.arange(half_dim) / half_dim)
        self.register_buffer('freqs', freqs)
        
    def forward(self, d):
        """
        Args:
            d: Tensor of shape [batch_size] or [batch_size, 1]
        Returns:
            Positional encodings of shape [batch_size, d_embed_dim]
        """
        # Convert to float to avoid dtype mismatch
        d = d.float()
        
        if d.dim() == 1:
            d = d.unsqueeze(-1)  # [batch_size, 1]
            
        # Apply sinusoidal encoding
        args = d * self.freqs[None, :]  # [batch_size, half_dim]
        
        # Concatenate sin and cos
        if self.d_embed_dim % 2 == 0:
            encodings = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        else:
            # Handle odd d_embed_dim by adding one more sin
            encodings = torch.cat([
                torch.sin(args), 
                torch.cos(args), 
                torch.sin(args[:, :1])
            ], dim=-1)
            
        return encodings


class DTConditionedPredictor(nn.Module):
    """Unified predictor that supports MLP and Linear (ridge) backends with d conditioning."""

    def __init__(
        self,
        latent_dim,
        model_type="mlp",  # "mlp" or "linear"
        hidden_dim=128,
        num_layers=2,
        conditioning_mode="sinusoidal",  # "sinusoidal", "concat"
        d_embed_dim=16,
        ridge_alpha=1e-3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.conditioning_mode = conditioning_mode
        self.ridge_alpha = ridge_alpha

        if self.conditioning_mode == "sinusoidal":
            self.d_encoder = SinusoidalPositionalEncoding(d_embed_dim)
            input_dim = latent_dim + d_embed_dim
        elif self.conditioning_mode == "concat":
            input_dim = latent_dim + 1
        else:
            raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")

        if self.model_type == "mlp":
            self.model = MLP(
                in_dims=input_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers,
            )
        elif self.model_type == "linear":
            self.model = nn.Linear(input_dim, latent_dim)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
        
        self.latent_act = nn.SELU()
        self.similarity = nn.CosineSimilarity()

    def forward(self, x, d):
        d = d.float()

        if self.conditioning_mode == "sinusoidal":
            d_encoded = self.d_encoder(d)
            x_conditioned = torch.cat([x, d_encoded], dim=-1)
            
        elif self.conditioning_mode == "concat":
            if d.dim() == 1:
                d = d.unsqueeze(-1)
            x_conditioned = torch.cat([x, d], dim=-1)
            
        else:
            raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")
        
        output = self.model(x_conditioned)

        return self.latent_act(output)

    def loss(self, source_latent, target_latent, d):
        pred_target_latent = self.forward(source_latent, d)
        loss = (1 - self.similarity(pred_target_latent, target_latent)).mean()
        if self.model_type == "linear" and self.ridge_alpha > 0:
            loss += self.ridge_alpha * torch.sum(self.model.weight ** 2)
        return loss, pred_target_latent