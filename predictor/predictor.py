import torch
import torch.nn as nn
import math


class SimpleMLP(nn.Module):
    """Lightweight MLP defined locally to avoid external dependencies."""

    def __init__(self, in_dims, hidden_dim, out_dim, layers):
        super().__init__()
        layers_list = []

        current_dim = in_dims
        if layers <= 1:
            layers_list.append(nn.Linear(current_dim, out_dim))
        else:
            for _ in range(layers - 1):
                layers_list.append(nn.Linear(current_dim, hidden_dim))
                layers_list.append(nn.ReLU())
                current_dim = hidden_dim
            layers_list.append(nn.Linear(current_dim, out_dim))

        self.network = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.network(x)


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
            d: Tensor of shape [batch_size], [batch_size, 1] or [batch_size, K] with K>=1 scalars
        Returns:
            Positional encodings of shape [batch_size, K * d_embed_dim]
        """
        # Convert to float to avoid dtype mismatch
        d = d.float()
        
        if d.dim() == 1:
            d = d.unsqueeze(-1)  # [batch_size, 1]
        
        # d shape: [batch_size, K]
        # Apply sinusoidal encoding per scalar and concatenate along feature dim
        # args shape: [batch_size, K, half_dim]
        args = d.unsqueeze(-1) * self.freqs.view(1, 1, -1)
        
        if self.d_embed_dim % 2 == 0:
            # [batch_size, K, d_embed_dim]
            enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        else:
            # Handle odd d_embed_dim by adding one more sin per scalar
            enc = torch.cat([
                torch.sin(args),
                torch.cos(args),
                torch.sin(args[:, :, :1])
            ], dim=-1)
        
        # Flatten K scalars into feature dimension -> [batch_size, K * d_embed_dim]
        batch_size, num_scalars, feat_dim = enc.shape
        return enc.view(batch_size, num_scalars * feat_dim)


class Predictor(nn.Module):
    """Unified predictor supporting optional conditioning with MLP or Ridge."""

    def __init__(
        self,
        latent_dim,
        model_type="mlp",  # "mlp" or "ridge"
        model_args=dict(
            hidden_dim=128,
            num_layers=2,
        ),
        conditioning_mode="sinusoidal",  # "sinusoidal", "concat", None
        d_embed_dim=16,
        num_condition_scalars=2,
        loss_type="cosine",  # "cosine" or "MSE"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_type = model_type
        self.model_args = model_args
        self.conditioning_mode = conditioning_mode
        self.num_condition_scalars = num_condition_scalars
        self.requires_condition = conditioning_mode is not None
        self.model_args = model_args
        self.d_embed_dim = d_embed_dim
        self.loss_type = loss_type
        
        if self.conditioning_mode == "sinusoidal":
            self.d_encoder = SinusoidalPositionalEncoding(d_embed_dim)
            input_dim = latent_dim + d_embed_dim * self.num_condition_scalars
        elif self.conditioning_mode == "concat":
            input_dim = latent_dim + self.num_condition_scalars
        elif self.conditioning_mode is None:
            input_dim = latent_dim
        else:
            raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")

        if self.model_type == "mlp":
            hidden_dim = self.model_args.get("hidden_dim", 128)
            num_layers = self.model_args.get("num_layers", 2)
            self.model = SimpleMLP(
                in_dims=input_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers,
            )
        elif self.model_type == "ridge":
            self.model = nn.Linear(input_dim, latent_dim)

        
        self.latent_act = nn.SELU()
        self.similarity = nn.CosineSimilarity()
        

    def forward(self, x, condition_scalars=None):
        # condition_scalars should be a tuple of scalars
        if condition_scalars:
            # TODO: make sure the condition_scalars are handled correctly here (especially for batch sizes larger than 1)
            d1, d2 = condition_scalars
            d1 = d1.float()
            d2 = d2.float()

            if d1.dim() == 1:
                d1 = d1.unsqueeze(-1)
            if d2.dim() == 1:
                d2 = d2.unsqueeze(-1)
            
            d_tensor = torch.cat([d1, d2], dim=-1)
            

        if self.conditioning_mode is None:
            x_conditioned = x
            
        elif self.conditioning_mode == "sinusoidal":
            d_encoded = self.d_encoder(d_tensor)
            x_conditioned = torch.cat([x, d_encoded], dim=-1)
            
        elif self.conditioning_mode == "concat":
            x_conditioned = torch.cat([x, d_tensor], dim=-1)
            
        else:
            raise ValueError(f"Unknown conditioning_mode: {self.conditioning_mode}")
        
        output = self.model(x_conditioned)

        return self.latent_act(output)

    def loss(self, source_latent, target_latent, condition_scalars=None):
        pred_target_latent = self.forward(source_latent, condition_scalars=condition_scalars)

        if self.loss_type == "MSE":
            loss = (pred_target_latent - target_latent).pow(2).mean()
        elif self.loss_type == "cosine":
            loss = (1 - self.similarity(pred_target_latent, target_latent)).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        if self.model_type == "ridge":
            loss += self.model_args.get("ridge_alpha", 0) * torch.sum(self.model.weight ** 2)
        return loss, pred_target_latent