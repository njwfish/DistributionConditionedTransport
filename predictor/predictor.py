import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    """Lightweight MLP defined locally to avoid external dependencies."""

    def __init__(self, in_dims, hidden_dim, out_dim, layers, activation):
        super().__init__()
        layers_list = []

        current_dim = in_dims
        self.activation = activation

        for _ in range(layers - 1):
            layers_list.append(nn.Linear(current_dim, hidden_dim))
            layers_list.append(self.activation)
            current_dim = hidden_dim
        layers_list.append(nn.Linear(current_dim, out_dim))

        self.network = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.network(x)


class Predictor(nn.Module):
    """Unified predictor with MLP or Ridge."""

    def __init__(
        self,
        latent_dim,
        model_type="mlp",  # "mlp" or "ridge"
        model_args=dict(
            hidden_dim=128,
            num_layers=2,
        ),
        loss_type="cosine",  # "cosine" or "MSE"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_type = model_type

        # Backward-compat: store string label as requested
        self.model_args = model_args
        self.loss_type = loss_type
        
        # Define activation before using it
        self.latent_act = nn.SELU()
        self.similarity = nn.CosineSimilarity()

        if self.model_type == "mlp":
            hidden_dim = self.model_args.get("hidden_dim", 128)
            num_layers = self.model_args.get("num_layers", 2)
            self.model = SimpleMLP(
                in_dims=latent_dim,
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers,
                activation=self.latent_act
            )
        elif self.model_type == "ridge":
            self.model = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        return self.model(x)

    def loss(self, source_latent, target_latent):
        """
        Compute predictor loss between source and target latents.
        
        Args:
            source_latent: Source latent vectors [batch, latent_dim]
            target_latent: Target latent vectors [batch, latent_dim]
            
        Returns:
            Loss scalar tensor
        """
        pred_target_latent = self.forward(source_latent)
        
        if self.loss_type.lower() == "mse":
            loss = (pred_target_latent - target_latent).pow(2).mean()
        elif self.loss_type.lower() == "cosine":
            loss = (1 - self.similarity(pred_target_latent, target_latent)).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if self.model_type == "ridge":
            loss += self.model_args.ridge_alpha * torch.sum(self.model.weight ** 2)
        
        return loss
        