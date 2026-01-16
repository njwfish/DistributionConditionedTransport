import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from utils.hf_local import resolve_local_or_repo

import math


# TODO: cite source (the original DFM paper).
# TODO: now that you are sticking to t \in [0,1], rather than t \in [0, 1000], is this still a good way to embed the timesteps?
def transformer_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int, max_positions: int = 10000) -> torch.Tensor:
    """Sinusoidal time embedding used for conditioning on continuous time t in [0, 1].

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


class TimeAwareEsmForFlow(EsmForMaskedLM):
    """ESM2 with additive conditioning on (source_latent, target_latent) and time t.

    The model outputs logits over tokens at each position like standard MLM, but we
    feed current sequence xt (token ids), attention mask, continuous time t, and a
    projected conditioning vector derived from the two latents.
    """

    def __init__(self, config: EsmConfig, latent_dim: int = 32, condition_dim: int = 256):
        super().__init__(config)
        self.hidden_size = config.hidden_size

        # Combine source and target latents (concatenate) and project to hidden size
        combined_latent_dim = latent_dim * 2
        self.condition_proj = nn.Sequential(
            nn.Linear(combined_latent_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, self.hidden_size)
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        t: torch.Tensor,
        latent_source: torch.Tensor,
        latent_target: torch.Tensor,
    ) -> torch.Tensor:
        # Encode tokens
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (B, L, D)

        # Time embedding (broadcast to all positions)
        time_embed = transformer_timestep_embedding(t, self.hidden_size).to(hidden_states.dtype)[:, None, :]
        hidden_states = hidden_states + time_embed

        # Additive latent conditioning (broadcast to all positions)
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)
        cond = self.condition_proj(combined_latent)  # (B, D)
        hidden_states = hidden_states + cond.unsqueeze(1)

        # Project to vocabulary logits
        logits = self.lm_head(hidden_states)  # (B, L, V)
        return logits


class ESM2_DFM_Generator(nn.Module):
    """Discrete Flow-Matching generator based on ESM2, drop-in for Progen2Generator.

    Conceptual difference: rather than concatenating source and target and using AR decoding,
    train an ESM2 MLM with uniform corruption towards target and evolve sequences by a flow
    over tokens, starting from the source sequence (seq2seq mutation rather than demasking).
    """

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        latent_dim: int = 32,
        condition_dim: int = 256,
        freeze_esm2: bool = False,
        temperature: float = 1.0,
        seq_length: Optional[int] = 1000,
        sample_dt: float = 0.1,
    ):
        super().__init__()

        # Tokenizer and config/model
        resolved_name = resolve_local_or_repo(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(resolved_name)
        self.config = EsmConfig.from_pretrained(resolved_name)
        self.model = TimeAwareEsmForFlow.from_pretrained(
            resolved_name,
            config=self.config,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
        )

        # Optionally freeze ESM backbone; keep LM head and conditioning projection trainable
        if freeze_esm2:
            for param in self.model.esm.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            for param in self.model.condition_proj.parameters():
                param.requires_grad = True

        # Special tokens and AA vocabulary subset (for constraints)
        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id
        self.bos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id

        aa_tokens = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
        ]
        self.aa_ids = [self.tokenizer.convert_tokens_to_ids(a) for a in aa_tokens]

        self.temperature = temperature
        self.seq_length = seq_length
        self.sample_dt = sample_dt

        # TODO: make sure X token ID is really correct.
        # Token id for ambiguous residue 'X' to optionally ignore in loss
        try:
            self.x_id = self.tokenizer.convert_tokens_to_ids('X')
        except Exception:
            self.x_id = None
    def _get_seq_lengths(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the actual sequence lengths (number of valid tokens) from attention mask.
        
        Args:
            attention_mask: (B, L) binary mask with 1s for valid positions
            
        Returns:
            (B,) tensor of sequence lengths
        """
        return attention_mask.sum(dim=-1)
    
    def _get_eos_positions(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the EOS position for each sequence (last valid position).
        
        For ESM tokenization: [CLS, AA1, ..., AAN, EOS, PAD, ...]
        The EOS is at position seq_length - 1 (0-indexed).
        
        Args:
            attention_mask: (B, L) binary mask with 1s for valid positions
            
        Returns:
            (B,) tensor of EOS positions
        """
        seq_lengths = self._get_seq_lengths(attention_mask)
        return seq_lengths - 1  # EOS is at the last valid position

    def loss(self, x_source, x_target, latent_source: torch.Tensor, latent_target: torch.Tensor) -> torch.Tensor:

        input_ids_source = x_source["esm_input_ids"]  
        attention_mask_source = x_source["esm_attention_mask"] 
        input_ids_target = x_target["esm_input_ids"]  
        attention_mask_target = x_target["esm_attention_mask"] 

        if input_ids_source.ndim == 3:
            B, S, L = input_ids_source.shape
            input_ids_source = input_ids_source.view(B * S, L)
            attention_mask_source = attention_mask_source.view(B * S, L)
            input_ids_target = input_ids_target.view(B * S, L)
            attention_mask_target = attention_mask_target.view(B * S, L)
            latent_source = latent_source.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)
            latent_target = latent_target.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)

        device = input_ids_target.device
        B, L = input_ids_target.shape

        # Sample times and build x_t by mixing source/target across all valid positions
        # Following the discrete flow-matching interpolation: x_t = where(Bernoulli(t), x_1, x_0)
        t = torch.rand((B,), device=device)

        # Valid (non-padded) positions: intersection of source and target masks
        if attention_mask_source is not None and attention_mask_target is not None:
            valid_mask = (attention_mask_source > 0) & (attention_mask_target > 0)
        elif attention_mask_target is not None:
            valid_mask = (attention_mask_target > 0)
        elif attention_mask_source is not None:
            valid_mask = (attention_mask_source > 0)
        else:
            valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

        bern = torch.rand((B, L), device=device) < t[:, None]
        # For valid positions, interpolate between source and target
        # For invalid (padding) positions, use padding token from target (will be masked in loss)
        xt = torch.where(valid_mask, torch.where(bern, input_ids_target, input_ids_source), input_ids_target)

        # Enforce BOS at position 0 for all sequences
        if self.bos_id is not None:
            xt[:, 0] = self.bos_id
        
        # Enforce EOS at the actual sequence boundary (not fixed position -1)
        # Use attention_mask_target to find EOS positions since that's our generation target
        if self.eos_id is not None and attention_mask_target is not None:
            eos_positions = self._get_eos_positions(attention_mask_target)
            batch_indices = torch.arange(B, device=device)
            xt[batch_indices, eos_positions] = self.eos_id

        # Forward through conditioned ESM
        attn_mask_xt = valid_mask.to(dtype=torch.long)
        logits = self.model(input_ids=xt,
            attention_mask=attn_mask_xt,
            t=t,
            latent_source=latent_source,
            latent_target=latent_target)

        # Cross-entropy to target tokens x1; ignore padded target positions
        labels = input_ids_target.clone()
        if attention_mask_target is not None:
            labels[attention_mask_target == 0] = -100
            
        # Ignore ambiguous 'X' residues in target
        if self.x_id is not None:
            labels[labels == self.x_id] = -100
        
        # Transpose for cross_entropy: (B, V, L) expected, logits is (B, L, V)
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='mean')

        return loss

    def _build_vocab_constraint_mask(self, attention_mask: torch.Tensor, vocab_size: int, device: torch.device) -> torch.Tensor:
        """Build a mask for constraining vocabulary at each position (vectorized).
        
        For valid positions:
        - Position 0: only BOS allowed
        - Inner positions (1 to EOS-1): only AA tokens allowed
        - EOS position: only EOS allowed
        - Padding positions: allow all tokens (won't affect generation, avoids NaN)
        
        Args:
            attention_mask: (B, L) binary mask
            vocab_size: vocabulary size
            device: torch device
            
        Returns:
            (B, L, V) mask with 0.0 for allowed tokens and -inf for disallowed
        """
        B, L = attention_mask.shape
        
        # Start with all -inf
        mask_logits = torch.full((B, L, vocab_size), -float('inf'), device=device)
        
        # Get EOS positions for each sequence: (B,)
        eos_positions = self._get_eos_positions(attention_mask)
        
        # Create position indices: (1, L)
        pos_indices = torch.arange(L, device=device).unsqueeze(0)
        
        # Build boolean masks for each position type: (B, L)
        is_bos_pos = (pos_indices == 0)  # Position 0
        is_eos_pos = (pos_indices == eos_positions.unsqueeze(1))  # EOS position per sequence
        is_padding = (attention_mask == 0)  # Padding positions
        is_inner = ~is_bos_pos & ~is_eos_pos & ~is_padding  # Inner AA positions
        
        # Position 0: BOS only
        if self.bos_id is not None:
            mask_logits[:, 0, self.bos_id] = 0.0
        
        # Inner positions: AA tokens only (vectorized assignment)
        aa_ids_tensor = torch.tensor(self.aa_ids, device=device)
        # Expand is_inner to (B, L, 1) and index into mask_logits
        inner_expanded = is_inner.unsqueeze(-1)  # (B, L, 1)
        # Set AA tokens to 0.0 where is_inner is True
        mask_logits[:, :, aa_ids_tensor] = torch.where(
            inner_expanded.expand(-1, -1, len(self.aa_ids)),
            torch.zeros(1, device=device),
            mask_logits[:, :, aa_ids_tensor]
        )
        
        # EOS positions: EOS only
        if self.eos_id is not None:
            batch_indices = torch.arange(B, device=device)
            mask_logits[batch_indices, eos_positions, self.eos_id] = 0.0
        
        # Padding positions: allow all tokens to avoid NaN in softmax
        # (these positions are masked in attention anyway)
        padding_mask_3d = is_padding.unsqueeze(-1).expand(-1, -1, vocab_size)
        mask_logits = torch.where(padding_mask_3d, torch.zeros_like(mask_logits), mask_logits)
            
        return mask_logits

    @torch.no_grad()
    def sample(self, x_source, latent_source: torch.Tensor, latent_target: torch.Tensor, num_samples: int = 1, return_texts: bool = False):
        """Mutational sampling starting from source sequences with discrete flow steps.

        Mirrors Progen2Generator.sample signature and return shape:
        - Returns [batch_size, num_samples, seq_len] token ids for ESM vocabulary.
        - Optionally returns decoded strings.
        """

        self.model.eval()
        device = latent_source.device

        src_ids_all = x_source["esm_input_ids"]
        src_mask_all = x_source["esm_attention_mask"]

        # Flatten set dim if present for starting point selection per sample
        if src_ids_all.ndim == 3:
            B, S, L = src_ids_all.shape
        else:
            raise ValueError(f"src_ids_all.ndim == {src_ids_all.ndim}, expected 3")

        def select_source_for_sample(sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            set_idx = sample_idx % S
            return src_ids_all[:, set_idx, :].to(device), src_mask_all[:, set_idx, :].to(device)

        results = []
        texts_all = []

        for s in range(num_samples):
            xt, attention_mask = select_source_for_sample(s)
            xt = xt.clone()  # Clone to avoid modifying source data
            
            # Get EOS positions from attention mask (actual sequence boundaries)
            eos_positions = self._get_eos_positions(attention_mask)
            batch_indices = torch.arange(B, device=device)
            
            # Enforce BOS at position 0
            if self.bos_id is not None:
                xt[:, 0] = self.bos_id
            
            # Enforce EOS at actual sequence boundaries
            if self.eos_id is not None:
                xt[batch_indices, eos_positions] = self.eos_id

            # Enforce expected sequence length (padded length)
            expected_len = self.seq_length
            if xt.size(1) != expected_len:
                raise ValueError(f"Source sequence length {xt.size(1)} does not match expected seq_length {expected_len}")

            # Build vocabulary constraint mask once (assumes all sequences have same structure)
            vocab_size = self.tokenizer.vocab_size
            constraint_mask = self._build_vocab_constraint_mask(attention_mask, vocab_size, device)

            # Discrete flow stepping from t=0 to 1 using mixture update
            t = 0.0
            while t < 1.0 - 1e-6:
                logits = self.model(
                    input_ids=xt,
                    attention_mask=attention_mask,
                    t=torch.full((B,), t, device=device),
                    latent_source=latent_source,
                    latent_target=latent_target,
                )
                
                # Temperature scaling
                logits = logits / max(self.temperature, 1e-8)

                # Apply vocabulary constraints per position
                p1 = F.softmax(logits + constraint_mask, dim=-1)  # (B, L, V)

                # Mixture update: probs = one_hot(xt) + h * (p1 - one_hot(xt)) / (1 - t)
                # with h = min(self.sample_dt, 1 - t)
                h = min(self.sample_dt, 1.0 - t)
                denom = max(1.0 - t, 1e-8)

                one_hot_x_t = F.one_hot(xt, num_classes=p1.size(-1)).to(dtype=torch.float32)
                p1_f32 = p1.to(dtype=torch.float32)
                step_probs = one_hot_x_t + (h / denom) * (p1_f32 - one_hot_x_t)
                # Numerical safety: clamp small negatives and renormalize
                step_probs = step_probs.clamp(min=0.0)
                step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

                # Sample next xt
                xt = torch.distributions.Categorical(probs=step_probs).sample()

                # Enforce BOS at position 0 after sampling
                if self.bos_id is not None:
                    xt[:, 0] = self.bos_id
                
                # Enforce EOS at actual sequence boundaries after sampling
                if self.eos_id is not None:
                    xt[batch_indices, eos_positions] = self.eos_id

                t += h

            results.append(xt)

            if return_texts:
                batch_texts = []
                for b in range(B):
                    # Only decode valid tokens (up to and including EOS)
                    seq_len = (eos_positions[b] + 1).item()
                    ids_b = xt[b, :seq_len]
                    text = self.tokenizer.decode(ids_b.tolist(), skip_special_tokens=True).replace(" ", "")
                    if not text:
                        text = ""
                    batch_texts.append(text)
                texts_all.append(batch_texts)

        # Stack to [B, num_samples, L]
        samples = torch.stack(results, dim=1)
        if return_texts:
            # Convert to list-of-list per batch
            # current texts_all is [num_samples][B], transpose to [B][num_samples]
            texts_per_batch = list(map(list, zip(*texts_all))) if len(texts_all) > 0 else [[] for _ in range(B)]
            return samples, texts_per_batch
        return samples