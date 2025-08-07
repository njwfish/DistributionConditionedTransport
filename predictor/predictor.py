import torch
import torch.nn as nn

from layers import MLP

class MLPPredictor(nn.Module):
    """Multi-layer perceptron for latent mapping"""
    
    # Class attribute to indicate this predictor does not require dt
    requires_dt = False
    
    def __init__(self, latent_dim, hidden_dim=128, num_layers=2):
        super().__init__()
        self.net = MLP(
            in_dims=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=latent_dim,
            layers=num_layers
        )
        self.latent_act = nn.SELU()

        self.similarity = nn.CosineSimilarity()

    def forward(self, x):
        return self.latent_act(self.net(x))
        
    def loss(self, source_latent, target_latent):
        """MSE loss between predicted and target latent"""
        pred_target_latent = self.forward(source_latent)
        return (1 - self.similarity(pred_target_latent, target_latent)).mean(), pred_target_latent

class RidgePredictor(nn.Module):
    """Ridge regression (linear mapping with L2 regularization). Set ridge_alpha to 0 to disable regularization."""
    
    # Class attribute to indicate this predictor does not require dt
    requires_dt = False
    
    def __init__(self, latent_dim, ridge_alpha=1e-3):
        super().__init__()
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.latent_act = nn.SELU()
        self.ridge_alpha = ridge_alpha
        self.similarity = nn.CosineSimilarity()
        
    def forward(self, x):
        return self.latent_act(self.linear(x))
    
    def loss(self, source_latent, target_latent):
        """L2 regularization on weights"""
        pred_target_latent = self.forward(source_latent)
        loss = (1 - self.similarity(pred_target_latent, target_latent)).mean()
        if self.ridge_alpha > 0:
            loss += self.ridge_alpha * torch.sum(self.linear.weight ** 2)
        return loss, pred_target_latent
