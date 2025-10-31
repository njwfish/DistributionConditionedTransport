import torch
import torch.nn as nn
import math
from utils.latents import normalize_latent


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with multi-head attention and feed-forward."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                             key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward block
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class SequenceTransformerEncoder(nn.Module):
    """BERT-like transformer encoder for discrete sequences.
    
    This encoder follows the DistributionEncoder API:
    - Takes input of shape (batch_size, set_size, seq_len) 
    - Returns latent of shape (batch_size, latent_dim)
    """
    
    def __init__(
        self, 
        vocab_size, 
        latent_dim, 
        hidden_dim, 
        set_size, 
        seq_len,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        normalize_latent_flag=False
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.set_size = set_size
        self.seq_len = seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=seq_len)
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling and projection to latent space
        self.pooling_type = 'mean'  # Could be 'mean', 'cls', or 'max'
        
        # Final projection layers
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        self.latent_act = nn.SELU()
        
        # Normalization
        self.normalize_latent_fn = normalize_latent if normalize_latent_flag else nn.Identity()
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, set_size, seq_len) or (set_size, seq_len)
               Contains discrete token indices
               
        Returns:
            latent: Tensor of shape (batch_size, latent_dim)
        """
        # Handle both batched and unbatched input
        if x.dim() == 2:
            # Add batch dimension: (set_size, seq_len) -> (1, set_size, seq_len)
            x = x.unsqueeze(0)
            
        batch_size, set_size, seq_len = x.shape
        
        # Reshape to process all sequences together
        x_flat = x.view(batch_size * set_size, seq_len)  # (batch_size * set_size, seq_len)
        
        # Token embedding
        embedded = self.token_embedding(x_flat)  # (batch_size * set_size, seq_len, hidden_dim)
        
        # Add positional encoding
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size * set_size, hidden_dim)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size * set_size, seq_len, hidden_dim)
        
        # Apply transformer layers
        hidden = embedded
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Pool sequence representations
        if self.pooling_type == 'mean':
            pooled = torch.mean(hidden, dim=1)  # (batch_size * set_size, hidden_dim)
        elif self.pooling_type == 'max':
            pooled = torch.max(hidden, dim=1)[0]  # (batch_size * set_size, hidden_dim)
        elif self.pooling_type == 'cls':
            pooled = hidden[:, 0, :]  # Use first token as [CLS] token
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")
        
        # Reshape back to set structure
        pooled = pooled.view(batch_size, set_size, self.hidden_dim)
        
        # Pool across set dimension (following DistributionEncoder pattern)
        set_pooled = torch.mean(pooled, dim=1)  # (batch_size, hidden_dim)
        
        # Project to latent space
        latent = self.latent_act(self.latent_proj(set_pooled))
        latent = self.normalize_latent_fn(latent)
        
        return latent


class DistributionEncoderTransformer(SequenceTransformerEncoder):
    """Alias that follows the naming convention of other encoders."""
    
    def __init__(self, vocab_size, latent_dim, hidden_dim, set_size, seq_len, **kwargs):
        super().__init__(vocab_size, latent_dim, hidden_dim, set_size, seq_len, **kwargs)
