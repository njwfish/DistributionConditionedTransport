
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv

from encoder.encoders import DistributionEncoder


class kNNEncoder(nn.Module):
    """
    Graph Neural Network encoder that builds k-NN graphs from input samples.
    
    This encoder:
    1. Builds a k-NN graph from input samples
    2. Applies GCN layers to learn representations
    3. Returns the graph-processed representations for pooling
    """
    
    def __init__(self, in_dim, hidden_dim, layers=3, knn_k=100):
        super().__init__()
        
        self.knn_k = knn_k
        
        # GCN layers
        self.gcn_convs = nn.ModuleList()
        self.gcn_convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(layers - 1):
            self.gcn_convs.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x):
        """
        Forward pass through the geometric GNN encoder.
        
        Args:
            x: Input tensor of shape [batch_size, set_size, in_dim]
            
        Returns:
            Encoded representations of shape [batch_size, set_size, hidden_dim]
        """
        batch_size, set_size, feature_dim = x.shape
        
        # Flatten batch and set dimensions for knn_graph
        x_flat = x.view(-1, feature_dim)  # [batch_size * set_size, feature_dim]
        
        # Create batch vector for knn_graph
        batch_vector = torch.arange(batch_size, device='cuda').repeat_interleave(set_size)
        
        # Build k-NN graphs for all samples at once using batch argument
        edge_index = torch_geometric.nn.pool.knn_graph(
            x_flat.cpu(), k=self.knn_k, batch=batch_vector.cpu()
        )
        edge_index = edge_index.to(x.device)
        
        # Apply GCN layers
        z = x_flat
        for i, conv in enumerate(self.gcn_convs):
            z = conv(z, edge_index)
            if i < len(self.gcn_convs) - 1:  # No activation on last layer
                z = F.relu(z)
        
        # Reshape back to batch format
        z = z.view(batch_size, set_size, -1)
        
        return z


class DistributionEncoderkNN(DistributionEncoder):
    """
    Distribution encoder using Graph Neural Networks with geometric nearest neighbor connections.
    
    Inherits from DistributionEncoder and follows the standard interface pattern.
    Preserves the original normalization behavior from the embed_source method.
    """
    
    def __init__(self, in_dim, latent_dim, hidden_dim, set_size, layers=3, knn_k=100):
        super().__init__(in_dim, latent_dim, hidden_dim, set_size)
        
        self.encoder = kNNEncoder(in_dim, hidden_dim, layers, knn_k)
    
    def forward(self, x):
        """
        Forward pass with original normalization behavior preserved.
        """
        # Get representations from the geometric GNN encoder
        enc = self.encoder(x)
        
        # Mean pooling across the set dimension (same as base class)
        enc_mean = torch.mean(enc, dim=1)
        
        # Apply normalization as in original embed_source implementation
        z_norm = torch.norm(enc_mean, dim=1, keepdim=True)
        enc_mean_normalized = enc_mean / (z_norm + 1e-8)  # Add epsilon for numerical stability
        
        # Project to latent space with normalization applied
        lat = self.latent_act(self.latent_proj(enc_mean_normalized))
        
        return lat