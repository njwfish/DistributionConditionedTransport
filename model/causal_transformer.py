import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
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
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class CausalSelfAttention(nn.Module):
    """Causal self-attention with masking for autoregressive generation."""
    
    def __init__(self, d_model, num_heads, dropout=0.1, max_seq_len=1024):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:, :, :seq_len, :seq_len]
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        out = self.out_proj(out)
        
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with causal attention and feed-forward."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.attention = CausalSelfAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Attention block with residual connection
        attn_out = self.attention(self.ln1(x))
        x = x + self.dropout(attn_out)
        
        # Feed-forward block with residual connection
        ff_out = self.feed_forward(self.ln2(x))
        x = x + ff_out
        
        return x


class CausalTransformer(nn.Module):
    """Causal transformer model for sequence generation.
    
    This model follows the expected API for the flow matching framework:
    - Takes (x, t, source_latent, target_latent) as input
    - Returns predictions for next tokens
    """
    
    def __init__(
        self,
        vocab_size,
        seq_len,
        latent_dim,
        d_model=512,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        # Time embedding (for flow matching)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Latent conditioning
        self.source_latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.target_latent_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.ln_final = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, t, source_latent, target_latent=None):
        """
        Args:
            x: Input token sequences of shape (batch_size, seq_len)
            t: Time values of shape (batch_size, 1) for flow matching
            source_latent: Source distribution embedding (batch_size, latent_dim)
            target_latent: Target distribution embedding (batch_size, latent_dim), optional
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = x.shape
        
        # Token embeddings
        token_emb = self.token_embedding(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x_emb = self.pos_encoding(token_emb)
        
        # Time conditioning
        time_emb = self.time_embedding(t)  # (batch_size, d_model)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Latent conditioning
        source_emb = self.source_latent_proj(source_latent)  # (batch_size, d_model)
        source_emb = source_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
        
        # Combine embeddings
        x_emb = x_emb + time_emb + source_emb
        
        # Add target latent if provided
        if target_latent is not None:
            target_emb = self.target_latent_proj(target_latent)  # (batch_size, d_model)
            target_emb = target_emb.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, d_model)
            x_emb = x_emb + target_emb
        
        # Apply transformer blocks
        hidden = x_emb
        for block in self.transformer_blocks:
            hidden = block(hidden)
        
        # Final layer norm and output projection
        hidden = self.ln_final(hidden)
        logits = self.output_proj(hidden)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def generate(self, source_latent, target_latent=None, max_length=None, temperature=1.0, top_k=None):
        """Generate sequences autoregressively.
        
        Args:
            source_latent: Source distribution embedding (batch_size, latent_dim)
            target_latent: Target distribution embedding (batch_size, latent_dim), optional
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            generated: Generated sequences (batch_size, max_length)
        """
        if max_length is None:
            max_length = self.seq_len
            
        batch_size = source_latent.shape[0]
        device = source_latent.device
        
        # Start with empty sequences (or start token)
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
        
        # Fixed time for generation (could be parameterized)
        t = torch.ones(batch_size, 1, device=device)
        
        for _ in range(max_length - 1):
            # Get logits for current sequence
            logits = self.forward(generated, t, source_latent, target_latent)
            
            # Get logits for next token (last position)
            next_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample next token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

