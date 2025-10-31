import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.latents import expand_latent_to_batch


class CausalTransformerGenerator(nn.Module):
    """Causal transformer generator for discrete sequence generation.
    
    This generator follows the same API as FlowMatchingGenerator but uses
    autoregressive generation instead of flow matching.
    """
    
    def __init__(
        self, 
        model, 
        temperature=1.0,
        top_k=None,
        max_new_tokens=None
    ):
        """
        Args:
            model: CausalTransformer model for sequence generation
            temperature: Sampling temperature for generation
            top_k: Top-k sampling parameter
            max_new_tokens: Maximum number of new tokens to generate
        """
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        
    def forward(self, source_samples, source_latent, target_latent, **kwargs):
        """Generate sequences using the causal transformer.
        
        Args:
            source_samples: Starting sequences [batch_size, seq_len] or [batch_size * set_size, seq_len]
            source_latent: Source distribution embedding
            target_latent: Target distribution embedding
            
        Returns:
            Generated sequences of same shape as source_samples
        """
        # Handle both single sequences and batched sets
        original_shape = source_samples.shape
        if source_samples.dim() == 3:
            # Reshape from [batch_size, set_size, seq_len] to [batch_size * set_size, seq_len]
            batch_size, set_size, seq_len = source_samples.shape
            source_samples = source_samples.view(batch_size * set_size, seq_len)
            
            # Expand latents to match
            source_latent = expand_latent_to_batch(source_latent, source_samples)
            target_latent = expand_latent_to_batch(target_latent, source_samples)
        
        # Generate sequences
        max_length = self.max_new_tokens or source_samples.shape[1]
        generated = self.model.generate(
            source_latent=source_latent,
            target_latent=target_latent,
            max_length=max_length,
            temperature=self.temperature,
            top_k=self.top_k
        )
        
        # Reshape back to original format if needed
        if len(original_shape) == 3:
            generated = generated.view(batch_size, set_size, -1)
            
        return generated
    
    def loss(self, source_samples, target_samples, source_latent, target_latent):
        """Compute autoregressive language modeling loss.
        
        Args:
            source_samples: Source sequences
            target_samples: Target sequences (used as training targets)
            source_latent: Source distribution embedding
            target_latent: Target distribution embedding
            
        Returns:
            Cross-entropy loss for next-token prediction
        """
        # The loss function reshapes (batch_size, set_size, seq_len) -> (batch_size * set_size, seq_len)
        # So we receive flattened samples and need to expand latents accordingly
        batch_size, seq_len = target_samples.shape
        
        # Infer original dimensions from latent batch size
        original_batch_size = source_latent.shape[0]
        set_size = batch_size // original_batch_size
        
        # Expand latents to match flattened samples
        if set_size > 1:
            source_latent = source_latent.unsqueeze(1).expand(-1, set_size, -1).contiguous().view(batch_size, -1)
            target_latent = target_latent.unsqueeze(1).expand(-1, set_size, -1).contiguous().view(batch_size, -1)
        
        # Create input and target sequences for autoregressive training
        # Input: [BOS, x1, x2, ..., x_{n-1}]
        # Target: [x1, x2, ..., x_n]
        
        # For simplicity, we'll use the target sequence shifted by one position
        input_seq = target_samples[:, :-1]  # Remove last token
        target_seq = target_samples[:, 1:]  # Remove first token
        
        # Create time values (fixed for training, could be randomized)
        t = torch.ones(batch_size, 1, device=target_samples.device)
        
        # Forward pass
        logits = self.model(input_seq, t, source_latent, target_latent)
        
        # Compute cross-entropy loss
        logits_flat = logits.reshape(-1, logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
        targets_flat = target_seq.reshape(-1)  # [batch_size * (seq_len-1)]
        
        loss = F.cross_entropy(logits_flat, targets_flat, reduction='mean')
        
        return loss
    
    def sample(self, source_samples, source_latent, target_latent, num_samples=None, **kwargs):
        """Generate samples using the causal transformer.
        
        Args:
            source_samples: Starting sequences (can be used to determine batch size)
            source_latent: Source distribution embedding
            target_latent: Target distribution embedding
            num_samples: Number of samples to generate per latent
            
        Returns:
            Generated sequences
        """
        if num_samples is None:
            # Infer from source_samples shape
            if source_samples.dim() == 3:
                num_samples = source_samples.shape[1]  # set_size
            else:
                num_samples = 1
        
        batch_size = source_latent.shape[0]
        
        # Expand latents for multiple samples
        if num_samples > 1:
            source_latent_expanded = source_latent.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
            source_latent_expanded = source_latent_expanded.view(batch_size * num_samples, -1)
            
            target_latent_expanded = target_latent.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
            target_latent_expanded = target_latent_expanded.view(batch_size * num_samples, -1)
        else:
            source_latent_expanded = source_latent
            target_latent_expanded = target_latent
        
        # Generate sequences
        max_length = self.max_new_tokens or (source_samples.shape[-1] if source_samples is not None else 20)
        generated = self.model.generate(
            source_latent=source_latent_expanded,
            target_latent=target_latent_expanded,
            max_length=max_length,
            temperature=self.temperature,
            top_k=self.top_k
        )
        
        # Reshape to [batch_size, num_samples, seq_len]
        if num_samples > 1:
            seq_len = generated.shape[1]
            generated = generated.view(batch_size, num_samples, seq_len)
        
        return generated


class DiscreteDiffusionGenerator(CausalTransformerGenerator):
    """Alternative name for the causal transformer generator."""
    pass
