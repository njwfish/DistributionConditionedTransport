from layers import MLP
import torch.nn as nn

class MultiMarginalMLP(nn.Module):
    def __init__(self, data_dim, latent_dim, hidden_dim, out_dim, layers=4):
        super().__init__()
        self.mlp = MLP([data_dim, latent_dim], hidden_dim, out_dim, layers)

    def forward(self, x, t, source_latent, target_latent):
        latent = t * source_latent + (1 - t) * target_latent
        return self.mlp(x, latent)