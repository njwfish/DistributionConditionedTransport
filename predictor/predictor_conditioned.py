"""
Conditioned predictor module.

This is a modified version of predictor.py that conditions the predictor on treatment information.
The predictor receives the source latent concatenated with the treatment condition one-hot encoding
(11-dimensional) to predict the target latent.

Usage:
    - Input: source_latent (batch_size, latent_dim) + treat_cond (batch_size, treat_dim)
    - Output: predicted_target_latent (batch_size, latent_dim)
"""

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


class PredictorConditioned(nn.Module):
    """
    Conditioned predictor that takes source latent + treatment condition as input.
    
    The treatment condition (treat_cond) is an 11-dimensional one-hot encoding
    that is concatenated with the source latent before being passed through the
    prediction network.
    """

    def __init__(
        self,
        latent_dim,
        treat_dim=11,  # dimension of treatment one-hot encoding
        model_type="mlp",  # "mlp" or "ridge"
        model_args=dict(
            hidden_dim=128,
            num_layers=2,
        ),
        loss_type="cosine",  # "cosine" or "MSE"
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.treat_dim = treat_dim
        self.model_type = model_type
        self.input_dim = latent_dim + treat_dim  # concatenated input dimension

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
                in_dims=self.input_dim,  # latent_dim + treat_dim
                hidden_dim=hidden_dim,
                out_dim=latent_dim,
                layers=num_layers,
                activation=self.latent_act
            )
        elif self.model_type == "ridge":
            self.model = nn.Linear(self.input_dim, latent_dim)

    def forward(self, source_latent, treat_cond):
        """
        Forward pass with conditioning.
        
        Args:
            source_latent: (batch_size, latent_dim) source embeddings
            treat_cond: (batch_size, treat_dim) treatment condition one-hot encoding
            
        Returns:
            predicted_target_latent: (batch_size, latent_dim)
        """
        # Concatenate source latent with treatment condition
        x = torch.cat([source_latent, treat_cond], dim=-1)
        return self.model(x)

    def loss(self, source_latent, target_latent, treat_cond, train_predictor_bool):
        """
        Compute predictor loss with treatment conditioning.
        
        Args:
            source_latent: (batch_size, latent_dim) source embeddings
            target_latent: (batch_size, latent_dim) target embeddings  
            treat_cond: (batch_size, treat_dim) treatment condition one-hot encoding
            train_predictor_bool: boolean mask for which samples to use for training
            
        Returns:
            loss: scalar loss value
        """
        # Convert to tensor if needed
        if not isinstance(train_predictor_bool, torch.Tensor):
            train_predictor_bool = torch.tensor(train_predictor_bool, device=source_latent.device)
        
        # Check if any samples should train the predictor
        if not train_predictor_bool.any():
            return torch.tensor(0.0, device=source_latent.device)
        
        # Mask to only include samples where train_predictor_bool is True
        mask = train_predictor_bool.bool()
        source_masked = source_latent[mask]
        target_masked = target_latent[mask]
        treat_cond_masked = treat_cond[mask]
        
        # Forward with conditioning
        pred_target_latent = self.forward(source_masked, treat_cond_masked)
        
        if self.loss_type == "MSE":
            loss = (pred_target_latent - target_masked).pow(2).mean()
        elif self.loss_type == "cosine":
            loss = (1 - self.similarity(pred_target_latent, target_masked)).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        if self.model_type == "ridge":
            loss += self.model_args.ridge_alpha * torch.sum(self.model.weight ** 2)
        
        return loss
