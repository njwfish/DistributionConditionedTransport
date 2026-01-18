"""
Edit Flows for Protein Sequence Generation

This module implements Edit Flows (https://arxiv.org/abs/2506.09018) for discrete 
flow-matching between protein sequences of variable lengths. Unlike standard discrete 
flow matching that operates on fixed-length sequences with token-wise transitions, 
Edit Flows use edit operations (insertions, deletions, substitutions) to enable 
natural variable-length generation.

The model learns to flow from source protein sequences to target protein sequences,
conditioned on latent embeddings for both distributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

from transformers import EsmConfig, EsmModel, EsmTokenizer

from utils.hf_local import resolve_local_or_repo

import math


# ============================================================================
# Utility Functions
# ============================================================================

def transformer_timestep_embedding(
    timesteps: torch.Tensor, 
    embedding_dim: int, 
    max_positions: int = 10000
) -> torch.Tensor:
    """Sinusoidal time embedding for conditioning on continuous time t in [0, 1].

    Args:
        timesteps: shape (B,) values in [0, 1]
        embedding_dim: hidden size
        max_positions: frequency base
        
    Returns:
        Tensor of shape (B, embedding_dim)
    """
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / max(half_dim - 1, 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1), mode='constant')
    return emb


# ============================================================================
# Sequence Alignment for Edit Flows
# ============================================================================

@dataclass
class AlignmentResult:
    """Result of sequence alignment for Edit Flows training."""
    z_0: torch.Tensor       # Aligned source sequence with epsilon tokens (B, max_align_len)
    z_1: torch.Tensor       # Aligned target sequence with epsilon tokens (B, max_align_len)
    align_mask: torch.Tensor  # Valid positions in alignment (B, max_align_len)
    edit_types: torch.Tensor  # Type of edit at each position: 0=match, 1=sub, 2=ins, 3=del (B, max_align_len)


def compute_optimal_alignment(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    mask_0: torch.Tensor,
    mask_1: torch.Tensor,
    epsilon_id: int,
    pad_id: int,
) -> AlignmentResult:
    """Compute optimal alignment between source and target sequences using dynamic programming.
    
    This implements the Needleman-Wunsch algorithm for global sequence alignment,
    which minimizes edit distance between sequences.
    
    Args:
        x_0: Source sequences (B, L0) - token ids
        x_1: Target sequences (B, L1) - token ids  
        mask_0: Source attention mask (B, L0)
        mask_1: Target attention mask (B, L1)
        epsilon_id: Token ID to use for gaps/epsilon in alignment
        pad_id: Padding token ID
        
    Returns:
        AlignmentResult with aligned sequences and edit type information
    """
    B = x_0.shape[0]
    device = x_0.device
    
    # Get actual sequence lengths (excluding padding)
    len_0 = mask_0.sum(dim=1).long()  # (B,)
    len_1 = mask_1.sum(dim=1).long()  # (B,)
    
    max_align_len = (len_0.max() + len_1.max()).item()
    
    # Initialize output tensors
    z_0_list = []
    z_1_list = []
    edit_types_list = []
    
    for b in range(B):
        n0 = len_0[b].item()
        n1 = len_1[b].item()
        
        seq_0 = x_0[b, :n0].tolist()
        seq_1 = x_1[b, :n1].tolist()
        
        # Dynamic programming for optimal alignment
        # dp[i][j] = minimum edit distance to align seq_0[:i] with seq_1[:j]
        dp = [[0] * (n1 + 1) for _ in range(n0 + 1)]
        
        # Initialize base cases
        for i in range(n0 + 1):
            dp[i][0] = i  # Deletions
        for j in range(n1 + 1):
            dp[0][j] = j  # Insertions
            
        # Fill DP table
        for i in range(1, n0 + 1):
            for j in range(1, n1 + 1):
                if seq_0[i-1] == seq_1[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # Match
                else:
                    dp[i][j] = min(
                        dp[i-1][j-1] + 1,  # Substitution
                        dp[i-1][j] + 1,     # Deletion
                        dp[i][j-1] + 1      # Insertion
                    )
        
        # Traceback to get alignment
        aligned_0 = []
        aligned_1 = []
        edits = []
        
        i, j = n0, n1
        while i > 0 or j > 0:
            if i > 0 and j > 0 and seq_0[i-1] == seq_1[j-1]:
                # Match
                aligned_0.append(seq_0[i-1])
                aligned_1.append(seq_1[j-1])
                edits.append(0)  # match
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                # Substitution
                aligned_0.append(seq_0[i-1])
                aligned_1.append(seq_1[j-1])
                edits.append(1)  # substitution
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                # Deletion (token in source, epsilon in target)
                aligned_0.append(seq_0[i-1])
                aligned_1.append(epsilon_id)
                edits.append(3)  # deletion
                i -= 1
            else:
                # Insertion (epsilon in source, token in target)
                aligned_0.append(epsilon_id)
                aligned_1.append(seq_1[j-1])
                edits.append(2)  # insertion
                j -= 1
        
        # Reverse to get correct order
        aligned_0 = aligned_0[::-1]
        aligned_1 = aligned_1[::-1]
        edits = edits[::-1]
        
        z_0_list.append(aligned_0)
        z_1_list.append(aligned_1)
        edit_types_list.append(edits)
    
    # Pad alignments to same length
    max_len = max(len(z) for z in z_0_list)
    
    z_0_padded = torch.full((B, max_len), pad_id, dtype=x_0.dtype, device=device)
    z_1_padded = torch.full((B, max_len), pad_id, dtype=x_1.dtype, device=device)
    edit_types = torch.full((B, max_len), -1, dtype=torch.long, device=device)
    align_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)
    
    for b in range(B):
        length = len(z_0_list[b])
        z_0_padded[b, :length] = torch.tensor(z_0_list[b], dtype=x_0.dtype, device=device)
        z_1_padded[b, :length] = torch.tensor(z_1_list[b], dtype=x_1.dtype, device=device)
        edit_types[b, :length] = torch.tensor(edit_types_list[b], dtype=torch.long, device=device)
        align_mask[b, :length] = True
    
    return AlignmentResult(
        z_0=z_0_padded,
        z_1=z_1_padded,
        align_mask=align_mask,
        edit_types=edit_types
    )


def sample_z_t(
    z_0: torch.Tensor,
    z_1: torch.Tensor,
    align_mask: torch.Tensor,
    kappa: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Sample intermediate aligned sequence z_t from the probability path.
    
    For each position i:
        z_t^i = z_1^i with probability kappa_t
        z_t^i = z_0^i with probability 1 - kappa_t
    
    Args:
        z_0: Aligned source sequences (B, L)
        z_1: Aligned target sequences (B, L)
        align_mask: Valid alignment positions (B, L)
        kappa: Scheduler value at time t, shape (B,) or scalar
        pad_id: Padding token ID
        
    Returns:
        z_t: Intermediate aligned sequence (B, L)
    """
    B, L = z_0.shape
    device = z_0.device
    
    if kappa.dim() == 0:
        kappa = kappa.expand(B)
    
    # Sample which positions take target value
    rand = torch.rand((B, L), device=device)
    use_target = rand < kappa[:, None]
    
    # Select between z_0 and z_1
    z_t = torch.where(use_target, z_1, z_0)
    
    # Apply mask
    z_t = torch.where(align_mask, z_t, torch.full_like(z_t, pad_id))
    
    return z_t


def remove_epsilon_tokens(
    z_t: torch.Tensor,
    align_mask: torch.Tensor,
    epsilon_id: int,
    pad_id: int,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Remove epsilon tokens from aligned sequence to get actual sequence x_t.
    
    Args:
        z_t: Aligned sequence with epsilon tokens (B, L_align)
        align_mask: Valid alignment positions (B, L_align)
        epsilon_id: Epsilon token ID
        pad_id: Padding token ID
        max_length: Maximum output sequence length
        
    Returns:
        x_t: Sequence without epsilon tokens (B, max_length)
        x_t_mask: Attention mask for x_t (B, max_length)
    """
    B = z_t.shape[0]
    device = z_t.device
    
    x_t_list = []
    
    for b in range(B):
        # Get valid tokens (non-epsilon, within alignment)
        valid = align_mask[b] & (z_t[b] != epsilon_id)
        tokens = z_t[b][valid].tolist()
        x_t_list.append(tokens)
    
    # Pad to max_length
    x_t = torch.full((B, max_length), pad_id, dtype=z_t.dtype, device=device)
    x_t_mask = torch.zeros((B, max_length), dtype=torch.long, device=device)
    
    for b in range(B):
        length = min(len(x_t_list[b]), max_length)
        if length > 0:
            x_t[b, :length] = torch.tensor(x_t_list[b][:length], dtype=z_t.dtype, device=device)
            x_t_mask[b, :length] = 1
    
    return x_t, x_t_mask


# ============================================================================
# Edit Flow Model Architecture
# ============================================================================

class EditFlowHead(nn.Module):
    """Head for predicting edit operation rates and token distributions.
    
    For each position, predicts:
    - λ_ins: rate of insertion after this position
    - λ_del: rate of deletion at this position  
    - λ_sub: rate of substitution at this position
    - Q_ins: distribution over tokens for insertion
    - Q_sub: distribution over tokens for substitution
    """
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Rate prediction heads (output positive values via softplus)
        self.lambda_ins_head = nn.Linear(hidden_size, 1)
        self.lambda_del_head = nn.Linear(hidden_size, 1)
        self.lambda_sub_head = nn.Linear(hidden_size, 1)
        
        # Token distribution heads
        self.q_ins_head = nn.Linear(hidden_size, vocab_size)
        self.q_sub_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, L, D) transformer outputs
            
        Returns:
            lambda_ins: (B, L) insertion rates
            lambda_del: (B, L) deletion rates
            lambda_sub: (B, L) substitution rates
            q_ins: (B, L, V) insertion token distributions (log-softmax)
            q_sub: (B, L, V) substitution token distributions (log-softmax)
        """
        # Predict rates (softplus to ensure positive)
        lambda_ins = F.softplus(self.lambda_ins_head(hidden_states)).squeeze(-1)
        lambda_del = F.softplus(self.lambda_del_head(hidden_states)).squeeze(-1)
        lambda_sub = F.softplus(self.lambda_sub_head(hidden_states)).squeeze(-1)
        
        # Predict token distributions (log-softmax for numerical stability)
        q_ins = F.log_softmax(self.q_ins_head(hidden_states), dim=-1)
        q_sub = F.log_softmax(self.q_sub_head(hidden_states), dim=-1)
        
        return lambda_ins, lambda_del, lambda_sub, q_ins, q_sub


class EditFlowTransformer(nn.Module):
    """Transformer model for Edit Flows with conditioning on latents and time.
    
    Based on ESM2 architecture, adapted for edit rate prediction.
    """
    
    def __init__(
        self, 
        config: EsmConfig, 
        latent_dim: int = 32, 
        condition_dim: int = 256
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # ESM backbone (without LM head)
        self.esm = EsmModel(config)
        
        # Conditioning projection
        combined_latent_dim = latent_dim * 2
        self.condition_proj = nn.Sequential(
            nn.Linear(combined_latent_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, self.hidden_size)
        )
        
        # Edit flow prediction head
        self.edit_head = EditFlowHead(self.hidden_size, config.vocab_size)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        latent_source: torch.Tensor,
        latent_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Current sequence x_t (B, L)
            attention_mask: Attention mask (B, L)
            t: Time values (B,) in [0, 1]
            latent_source: Source distribution latent (B, D_lat)
            latent_target: Target distribution latent (B, D_lat)
            
        Returns:
            lambda_ins, lambda_del, lambda_sub: Edit rates (B, L)
            q_ins, q_sub: Token distributions (B, L, V)
        """
        # Get transformer outputs
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, D)
        
        # Add time embedding
        time_embed = transformer_timestep_embedding(t, self.hidden_size)
        time_embed = time_embed.to(hidden_states.dtype)[:, None, :]  # (B, 1, D)
        hidden_states = hidden_states + time_embed
        
        # Add latent conditioning
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)
        cond = self.condition_proj(combined_latent)  # (B, D)
        hidden_states = hidden_states + cond.unsqueeze(1)
        
        # Predict edit rates and distributions
        return self.edit_head(hidden_states)


# ============================================================================
# Edit Flow Generator
# ============================================================================

class EditFlowGenerator(nn.Module):
    """Discrete Flow-Matching generator using Edit Flows for variable-length protein sequences.
    
    This generator learns to flow from source protein sequences to target sequences,
    supporting variable-length input and output through edit operations (insertions,
    deletions, substitutions).
    
    The model is conditioned on:
    - The current sequence state x_t
    - Source distribution latent embedding
    - Target distribution latent embedding  
    - Time t in [0, 1]
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        latent_dim: int = 32,
        condition_dim: int = 256,
        freeze_backbone: bool = False,
        max_length: int = 512,
        scheduler: str = "cubic",  # "linear" or "cubic"
        sample_steps: int = 100,
        temperature: float = 1.0,
    ):
        """
        Args:
            model_name: Pretrained ESM2 model name
            latent_dim: Dimension of latent embeddings
            condition_dim: Dimension of conditioning projection
            freeze_backbone: Whether to freeze the ESM backbone
            max_length: Maximum sequence length
            scheduler: Scheduler type for kappa_t ("linear" or "cubic")
            sample_steps: Number of sampling steps
            temperature: Sampling temperature
        """
        super().__init__()
        
        self.max_length = max_length
        self.scheduler = scheduler
        self.sample_steps = sample_steps
        self.temperature = temperature
        
        # Load tokenizer and config
        resolved_name = resolve_local_or_repo(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(resolved_name)
        config = EsmConfig.from_pretrained(resolved_name)
        
        # Initialize model
        self.model = EditFlowTransformer(
            config=config,
            latent_dim=latent_dim,
            condition_dim=condition_dim
        )
        
        # Load pretrained weights into backbone
        pretrained = EsmModel.from_pretrained(resolved_name)
        self.model.esm.load_state_dict(pretrained.state_dict())
        del pretrained
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.model.esm.parameters():
                param.requires_grad = False
        
        # Special token IDs
        self.pad_id = self.tokenizer.pad_token_id
        self.cls_id = self.tokenizer.cls_token_id  # BOS
        self.eos_id = self.tokenizer.eos_token_id
        
        # Epsilon token for alignment (use mask token or add a new one)
        # We use an ID that won't appear in actual sequences
        self.epsilon_id = self.tokenizer.mask_token_id
        
        # Amino acid token IDs for constrained sampling
        aa_tokens = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
        ]
        self.aa_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(a) for a in aa_tokens])
        
    def _get_kappa(self, t: torch.Tensor) -> torch.Tensor:
        """Compute scheduler value kappa(t).
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            kappa_t: Scheduler values
        """
        if self.scheduler == "linear":
            return t
        elif self.scheduler == "cubic":
            return t ** 3
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    def _get_kappa_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """Compute derivative of scheduler d(kappa)/dt.
        
        Args:
            t: Time values in [0, 1]
            
        Returns:
            d_kappa: Scheduler derivatives
        """
        if self.scheduler == "linear":
            return torch.ones_like(t)
        elif self.scheduler == "cubic":
            return 3 * t ** 2
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
    
    def _extract_sequences(self, x: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract input_ids and attention_mask from input dictionary.
        
        Handles both 2D (B, L) and 3D (B, S, L) tensor shapes.
        """
        input_ids = x["esm_input_ids"]
        attention_mask = x["esm_attention_mask"]
        
        if input_ids.ndim == 3:
            B, S, L = input_ids.shape
            input_ids = input_ids.view(B * S, L)
            attention_mask = attention_mask.view(B * S, L)
            
        return input_ids, attention_mask
    
    def _expand_latents(
        self, 
        latent: torch.Tensor, 
        input_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Expand latents to match flattened batch dimension if needed."""
        if len(input_shape) == 3:
            B, S, L = input_shape
            return latent.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)
        return latent
    
    def loss(
        self, 
        x_source: Dict, 
        x_target: Dict, 
        latent_source: torch.Tensor, 
        latent_target: torch.Tensor
    ) -> torch.Tensor:
        """Compute Edit Flow training loss.
        
        The loss follows Equation (23) from the paper:
        L(θ) = E[ Σ_x≠x_t u_θ(x|x_t) - Σ_i 1[z_1^i ≠ z_t^i] * (dκ/dt)/(1-κ) * log u_θ(edit_i|x_t) ]
        
        Args:
            x_source: Source sequences dict with 'esm_input_ids' and 'esm_attention_mask'
            x_target: Target sequences dict
            latent_source: Source distribution latent (B, D)
            latent_target: Target distribution latent (B, D)
            
        Returns:
            Scalar loss value
        """
        # Get original shape for latent expansion
        original_shape = x_source["esm_input_ids"].shape
        
        # Extract sequences
        ids_source, mask_source = self._extract_sequences(x_source)
        ids_target, mask_target = self._extract_sequences(x_target)
        
        # Expand latents if needed
        latent_source = self._expand_latents(latent_source, original_shape)
        latent_target = self._expand_latents(latent_target, original_shape)
        
        device = ids_source.device
        B = ids_source.shape[0]
        
        # Sample time t uniformly from [0, 1)
        t = torch.rand((B,), device=device) * 0.999  # Avoid t=1 for numerical stability
        kappa = self._get_kappa(t)
        d_kappa = self._get_kappa_derivative(t)
        
        # Compute optimal alignment between source and target
        alignment = compute_optimal_alignment(
            ids_source, ids_target,
            mask_source, mask_target,
            self.epsilon_id, self.pad_id
        )
        
        # Sample intermediate aligned state z_t
        z_t = sample_z_t(
            alignment.z_0, alignment.z_1,
            alignment.align_mask, kappa, self.pad_id
        )
        
        # Remove epsilon tokens to get x_t
        x_t, x_t_mask = remove_epsilon_tokens(
            z_t, alignment.align_mask,
            self.epsilon_id, self.pad_id, self.max_length
        )
        
        # Handle empty sequences (can happen if all tokens are epsilon)
        empty_mask = x_t_mask.sum(dim=1) == 0
        if empty_mask.any():
            # For empty sequences, use a single CLS token
            x_t[empty_mask, 0] = self.cls_id
            x_t_mask[empty_mask, 0] = 1
        
        # Forward pass through model
        lambda_ins, lambda_del, lambda_sub, q_ins, q_sub = self.model(
            input_ids=x_t,
            attention_mask=x_t_mask,
            t=t,
            latent_source=latent_source,
            latent_target=latent_target
        )
        
        # Compute loss term 1: sum of all rates (for valid positions)
        # This term minimizes total edit rate
        rate_sum = (lambda_ins + lambda_del + lambda_sub) * x_t_mask.float()
        loss_term_1 = rate_sum.sum(dim=1).mean()
        
        # Compute loss term 2: weighted cross-entropy for required edits
        # We need to map alignment positions to x_t positions
        loss_term_2 = self._compute_edit_loss(
            z_t, alignment.z_1, alignment.align_mask,
            x_t, x_t_mask,
            lambda_ins, lambda_del, lambda_sub,
            q_ins, q_sub,
            kappa, d_kappa
        )
        
        total_loss = loss_term_1 + loss_term_2
        return total_loss
    
    def _compute_edit_loss(
        self,
        z_t: torch.Tensor,
        z_1: torch.Tensor,
        align_mask: torch.Tensor,
        x_t: torch.Tensor,
        x_t_mask: torch.Tensor,
        lambda_ins: torch.Tensor,
        lambda_del: torch.Tensor,
        lambda_sub: torch.Tensor,
        q_ins: torch.Tensor,
        q_sub: torch.Tensor,
        kappa: torch.Tensor,
        d_kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss term for required edits.
        
        For each position where z_t != z_1, we need to perform an edit to reach z_1.
        The loss encourages high rate and correct token distribution for that edit.
        """
        B = z_t.shape[0]
        device = z_t.device
        
        # Weight for cross-entropy: d_kappa / (1 - kappa)
        weight = d_kappa / (1 - kappa + 1e-8)  # (B,)
        
        total_loss = torch.zeros(B, device=device)
        
        for b in range(B):
            x_t_idx = -1  # Current position in x_t (tracks non-epsilon tokens)
            
            for align_idx in range(align_mask.shape[1]):
                if not align_mask[b, align_idx]:
                    continue
                    
                z_t_token = z_t[b, align_idx].item()
                z_1_token = z_1[b, align_idx].item()
                
                # Track position in x_t
                if z_t_token != self.epsilon_id:
                    x_t_idx += 1
                
                # Skip if already at target
                if z_t_token == z_1_token:
                    continue
                
                # Determine edit type and compute loss
                if z_t_token == self.epsilon_id and z_1_token != self.epsilon_id:
                    # Need insertion: token missing from x_t, should be in x_1
                    # Insert after current x_t position (or at beginning)
                    pos = max(0, x_t_idx)
                    if pos < x_t_mask.shape[1] and x_t_mask[b, pos]:
                        # Log probability of correct insertion
                        log_rate = torch.log(lambda_ins[b, pos] + 1e-8)
                        log_q = q_ins[b, pos, z_1_token]
                        total_loss[b] -= weight[b] * (log_rate + log_q)
                        
                elif z_t_token != self.epsilon_id and z_1_token == self.epsilon_id:
                    # Need deletion: token in x_t should be removed
                    pos = x_t_idx
                    if 0 <= pos < x_t_mask.shape[1] and x_t_mask[b, pos]:
                        log_rate = torch.log(lambda_del[b, pos] + 1e-8)
                        total_loss[b] -= weight[b] * log_rate
                        
                elif z_t_token != self.epsilon_id and z_1_token != self.epsilon_id:
                    # Need substitution: token in x_t should be replaced
                    pos = x_t_idx
                    if 0 <= pos < x_t_mask.shape[1] and x_t_mask[b, pos]:
                        log_rate = torch.log(lambda_sub[b, pos] + 1e-8)
                        log_q = q_sub[b, pos, z_1_token]
                        total_loss[b] -= weight[b] * (log_rate + log_q)
        
        return total_loss.mean()
    
    @torch.no_grad()
    def sample(
        self,
        x_source: Dict,
        latent_source: torch.Tensor,
        latent_target: torch.Tensor,
        num_samples: int = 1,
        return_texts: bool = False,
    ) -> torch.Tensor:
        """Sample target sequences using Edit Flow dynamics.
        
        Starting from source sequences, iteratively apply edit operations
        based on predicted rates to generate target sequences.
        
        Args:
            x_source: Source sequences dict
            latent_source: Source distribution latent (B, D)
            latent_target: Target distribution latent (B, D)
            num_samples: Number of samples per input
            return_texts: Whether to return decoded text sequences
            
        Returns:
            samples: Generated token IDs (B, num_samples, L)
            texts (optional): Decoded sequences
        """
        self.model.eval()
        device = latent_source.device
        
        # Get source sequences
        src_ids = x_source["esm_input_ids"]
        src_mask = x_source["esm_attention_mask"]
        
        # Handle 3D input (B, S, L)
        if src_ids.ndim == 3:
            B, S, L = src_ids.shape
        else:
            B = src_ids.shape[0]
            S = 1
            src_ids = src_ids.unsqueeze(1)
            src_mask = src_mask.unsqueeze(1)
        
        all_samples = []
        all_texts = [] if return_texts else None
        
        for sample_idx in range(num_samples):
            # Select source sequence for this sample
            set_idx = sample_idx % S
            x_t = src_ids[:, set_idx, :].clone().to(device)
            mask_t = src_mask[:, set_idx, :].clone().to(device)
            
            # Get actual sequence lengths
            seq_lens = mask_t.sum(dim=1)
            
            # Sampling loop
            dt = 1.0 / self.sample_steps
            t = 0.0
            
            for step in range(self.sample_steps):
                t_tensor = torch.full((B,), t, device=device)
                
                # Forward pass
                lambda_ins, lambda_del, lambda_sub, q_ins, q_sub = self.model(
                    input_ids=x_t,
                    attention_mask=mask_t,
                    t=t_tensor,
                    latent_source=latent_source,
                    latent_target=latent_target
                )
                
                # Apply temperature to distributions
                if self.temperature != 1.0:
                    q_ins = q_ins / self.temperature
                    q_sub = q_sub / self.temperature
                
                # Apply edits based on rates
                x_t, mask_t, seq_lens = self._apply_edits(
                    x_t, mask_t, seq_lens,
                    lambda_ins, lambda_del, lambda_sub,
                    q_ins, q_sub,
                    dt
                )
                
                t += dt
            
            all_samples.append(x_t)
            
            if return_texts:
                batch_texts = []
                for b in range(B):
                    seq_len = seq_lens[b].item()
                    ids = x_t[b, :seq_len].tolist()
                    text = self.tokenizer.decode(ids, skip_special_tokens=True).replace(" ", "")
                    batch_texts.append(text if text else "")
                all_texts.append(batch_texts)
        
        # Stack samples: (B, num_samples, L)
        max_len = max(s.shape[1] for s in all_samples)
        samples = torch.full((B, num_samples, max_len), self.pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(all_samples):
            samples[:, i, :s.shape[1]] = s
        
        if return_texts:
            # Transpose texts from [num_samples][B] to [B][num_samples]
            texts_per_batch = list(map(list, zip(*all_texts)))
            return samples, texts_per_batch
        
        return samples
    
    def _apply_edits(
        self,
        x_t: torch.Tensor,
        mask_t: torch.Tensor,
        seq_lens: torch.Tensor,
        lambda_ins: torch.Tensor,
        lambda_del: torch.Tensor,
        lambda_sub: torch.Tensor,
        q_ins: torch.Tensor,
        q_sub: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply edit operations based on predicted rates.
        
        For each position, with probability dt * lambda, apply the corresponding edit.
        """
        B, L = x_t.shape
        device = x_t.device
        
        # Clamp rates to reasonable values
        lambda_ins = lambda_ins.clamp(max=10.0)
        lambda_del = lambda_del.clamp(max=10.0)
        lambda_sub = lambda_sub.clamp(max=10.0)
        
        # Sample edit decisions
        do_ins = (torch.rand((B, L), device=device) < dt * lambda_ins) & (mask_t > 0)
        do_del = (torch.rand((B, L), device=device) < dt * lambda_del) & (mask_t > 0)
        do_sub = (torch.rand((B, L), device=device) < dt * lambda_sub) & (mask_t > 0)
        
        # Exclusivity: if multiple edits selected at same position, prioritize
        # substitution > deletion > insertion
        do_del = do_del & ~do_sub
        do_ins = do_ins & ~do_sub & ~do_del
        
        # Apply substitutions (in-place)
        if do_sub.any():
            sub_probs = F.softmax(q_sub, dim=-1)
            new_tokens = torch.multinomial(
                sub_probs.view(-1, sub_probs.shape[-1]), 
                num_samples=1
            ).view(B, L)
            x_t = torch.where(do_sub, new_tokens, x_t)
        
        # Apply deletions and insertions (requires rebuilding sequences)
        new_x_t_list = []
        new_mask_list = []
        new_lens = []
        
        for b in range(B):
            old_seq = x_t[b, :seq_lens[b]].tolist()
            new_seq = []
            
            ins_probs = F.softmax(q_ins[b], dim=-1)
            
            for pos, token in enumerate(old_seq):
                # Insertion before this position
                if pos < L and do_ins[b, pos]:
                    ins_token = torch.multinomial(ins_probs[pos], num_samples=1).item()
                    new_seq.append(ins_token)
                
                # Deletion: skip this token
                if pos < L and do_del[b, pos]:
                    continue
                    
                new_seq.append(token)
            
            # Ensure we have at least one token (CLS)
            if len(new_seq) == 0:
                new_seq = [self.cls_id]
            
            # Truncate if too long
            if len(new_seq) > self.max_length:
                new_seq = new_seq[:self.max_length]
            
            new_x_t_list.append(new_seq)
            new_lens.append(len(new_seq))
        
        # Rebuild tensors
        new_max_len = max(new_lens)
        new_x_t = torch.full((B, new_max_len), self.pad_id, dtype=x_t.dtype, device=device)
        new_mask_t = torch.zeros((B, new_max_len), dtype=mask_t.dtype, device=device)
        
        for b in range(B):
            length = new_lens[b]
            new_x_t[b, :length] = torch.tensor(new_x_t_list[b], dtype=x_t.dtype, device=device)
            new_mask_t[b, :length] = 1
        
        return new_x_t, new_mask_t, torch.tensor(new_lens, device=device)


# ============================================================================
# Backward Compatibility Alias
# ============================================================================

# Alias for backward compatibility with existing code
ESM2_DFM_Generator = EditFlowGenerator
