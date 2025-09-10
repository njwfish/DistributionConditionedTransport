import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hf_local import resolve_local_or_repo
from utils.debug_memory_logger import get_debug_logger
import torch.nn.functional as F
from typing import Optional

# This module has been replaced by a more complex sequential network in ConditionedProgen2
# class ZToPrefix(nn.Module):
#     def __init__(self, input_dim, prefix_length, d_model):
#         super().__init__()
#         # A simple linear projection that outputs prefix_length * d_model values
#         self.fc = nn.Linear(input_dim, prefix_length * d_model)
#         self.prefix_length = prefix_length
#         self.d_model = d_model
#         
#     def forward(self, x):
#         # x shape: (batch_size, input_dim)
#         batch_size = x.size(0)
#         prefix = self.fc(x)            # Shape: (batch_size, prefix_length * d_model)
#         prefix = prefix.view(batch_size, self.prefix_length, self.d_model)  # Reshape to (batch_size, prefix_length, d_model)
#         return prefix

class ConditionedProgen2(nn.Module):
    """Progen2 model conditioned on a latent vector representing a distribution of protein sequences."""
    
    def __init__(
        self, 
        progen2_name='hugohrban/progen2-medium', 
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix",
        seq_length=1000,
    ):
        """
        Initialize a conditioned Progen2 model.
        
        Args:
            progen2_name: Name of the pretrained Progen2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_progen2: Whether to freeze the Progen2 parameters
            condition_method: How to condition the model ('prefix' or 'additive')
        """
        import logging
        import time
        import os
        
        super().__init__()
        
        # Initialize Progen2 model      
        self.progen2 = AutoModelForCausalLM.from_pretrained(resolve_local_or_repo(progen2_name), trust_remote_code=True)
        
        # TODO: make really sure that progen2 is not frozen for virus task.
        # Freeze Progen2 if specified
        if freeze_progen2:
            for param in self.progen2.parameters():
                param.requires_grad = False
        
        # Get the embedding dimension from the model config
        # Different models might use different attribute names
        if hasattr(self.progen2.config, 'hidden_size'):
            self.hidden_dim = self.progen2.config.hidden_size
        elif hasattr(self.progen2.config, 'n_embd'):
            self.hidden_dim = self.progen2.config.n_embd
        elif hasattr(self.progen2.config, 'embed_dim'):
            self.hidden_dim = self.progen2.config.embed_dim
        elif hasattr(self.progen2.config, 'd_model'):
            self.hidden_dim = self.progen2.config.d_model
        # TODO: remove this strange fallback.
        else:
            # Default value if none of the above attributes exist
            self.hidden_dim = 768
        
        self.condition_method = condition_method
        self.seq_length = seq_length
        
        # Project latent to correct dimension (same approach as in GPT-2)
        # Note: input dimension is doubled since we concatenate source and target latents
        combined_latent_dim = latent_dim * 2
        
        if self.condition_method == "prefix":
            # For prefix conditioning, project to 20 token embeddings inserted mid-sequence
            self.num_condition_tokens = 20
            self.condition_proj = nn.Sequential(
                nn.Linear(combined_latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim * self.num_condition_tokens)
            )
        elif self.condition_method == "additive":
            # For additive conditioning, project to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(combined_latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")

        # Align dtype of conditioning projection with model (default)

    def forward(self, input_ids, attention_mask, latent_source, latent_target):
        """
        Forward pass through the conditioned Progen2 model.
        Memory-optimized to process sequences individually when dealing with sets.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len] or [batch_size, set_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] or [batch_size, set_size, seq_len]
            latent_source: Source latent distribution embedding [batch_size, latent_dim]
            latent_target: Target latent distribution embedding [batch_size, latent_dim]
            
        Returns:
            Logits for next token prediction
        """
        batch_size = input_ids.shape[0]
        
        # Combine the two latents - concatenate them to create a richer conditioning signal
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)
        # Project combined latent to correct dimension
        condition = self.condition_proj(combined_latent)
        
        # Flatten set dimension if present and expand condition accordingly
        if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
            set_size, seq_len = attention_mask.shape[1:]
            cur_input_ids = input_ids.view(batch_size * set_size, seq_len)
            cur_attention_mask = attention_mask.view(batch_size * set_size, seq_len)
            cur_condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
        else:
            cur_input_ids = input_ids
            cur_attention_mask = attention_mask
            cur_condition = condition
        
        if self.condition_method == "prefix":
            # Insert 20 learned condition tokens after the source + sep token boundary
            bsz = cur_input_ids.shape[0]
            insertion_index = self.seq_length + 1  # after source (seq_length) and sep (1)

            # Get token embeddings for the input sequence
            if hasattr(self.progen2.transformer, 'wte'):
                token_embeds = self.progen2.transformer.wte(cur_input_ids)
            else:
                token_embeds = self.progen2.get_input_embeddings()(cur_input_ids)

            # Reshape condition into 20 token embeddings
            cond_token_embeds = cur_condition.view(bsz, self.num_condition_tokens, self.hidden_dim)

            # Split embeddings and attention at insertion point
            left_embeds = token_embeds[:, :insertion_index, :]
            right_embeds = token_embeds[:, insertion_index:, :]

            left_mask = cur_attention_mask[:, :insertion_index]
            right_mask = cur_attention_mask[:, insertion_index:]
            cond_mask = torch.ones(bsz, self.num_condition_tokens, dtype=cur_attention_mask.dtype, device=cur_attention_mask.device)

            combined_embeds = torch.cat([left_embeds, cond_token_embeds, right_embeds], dim=1)
            extended_attention_mask = torch.cat([left_mask, cond_mask, right_mask], dim=1)

            outputs = self.progen2(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True
            )
            # Remove logits at the inserted positions to keep alignment with original input length
            left_logits = outputs.logits[:, :insertion_index, :]
            right_logits = outputs.logits[:, insertion_index + self.num_condition_tokens:, :]
            logits = torch.cat([left_logits, right_logits], dim=1)
            # TODO: make sure that this is correct
            # Ensure the last timestep predicts the token after the inserted condition tokens
            logits[:, insertion_index - 1, :] = outputs.logits[:, insertion_index + self.num_condition_tokens - 1, :]
        elif self.condition_method == "additive":
            outputs = self.progen2(
                input_ids=cur_input_ids,
                attention_mask=cur_attention_mask,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            
            condition_broadcast = cur_condition.unsqueeze(1)  # [batch, 1, hidden_dim]
            hidden_states = hidden_states + condition_broadcast
            logits = self.progen2.lm_head(hidden_states)
        else:
            raise ValueError(f"Unknown conditioning method: {self.condition_method}")
        
        return logits

    


class Progen2Generator(nn.Module):
    """Generator class using conditioned Progen2 for the distribution embeddings framework."""
    
    def __init__(
        self,
        progen2_name="hugohrban/progen2-medium",
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="additive",
        temperature=1.0,
        seq_length=1000,
    ):
        """
        Initialize the Progen2 generator.
        
        Args:
            progen2_name: Name of the pretrained Progen2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_progen2: Whether to freeze the Progen2 parameters
            condition_method: How to condition Progen2
            temperature: Sampling temperature
            seq_length: sequence length (constant for all sequences)
        """
        super().__init__()

        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_progen2=freeze_progen2,
            # Force additive conditioning to match ESM2 additive approach
            condition_method="additive",
            seq_length=seq_length,
        )
        
        self.temperature = temperature
        self.seq_length = seq_length
        
        # Initialize tokenizer (for generation)
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_local_or_repo(progen2_name), trust_remote_code=True)
        
        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|pad|>'
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<|bos|>'
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '<|eos|>'
        
        # Resolve separator token id ('<|endoftext|>') and pad token id
        self.sep_token = '<|endoftext|>'
        # TODO: make sure that this doesn't default to some nonsensical value when the sep_token doesn't exist.
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        # TODO: is this correct?
        self.pad_token_id = self.tokenizer.pad_token_id if (hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None) else 0        
    
    # TODO: why does the ESM-DFM model do all this reshaping work, while the Progen2 code doesn't?
    # TODO: throughout this code, make sure the attention mask is actually both correct (as loaded from the dataset) and used correctly.
    def loss(self, x_source, x_target, latent_source, latent_target):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'progen_input_ids' and 'progen_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"Progen2Generator.loss expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        source_ids = x_source['progen_input_ids']
        source_attention_mask = x_source['progen_attention_mask']
        
        target_ids = x_target['progen_input_ids']
        target_attention_mask = x_target['progen_attention_mask']
        # Remove BOS from target for conditioning and loss; predict only amino acids
        target_ids_wo_bos = target_ids[..., 1:]
        target_attention_mask_wo_bos = target_attention_mask[..., 1:]
        
        # Concatenate source + <|endoftext|> + target for conditioning on source content
        sep_shape = list(source_ids.shape[:-1]) + [1]
        sep_ids = torch.full(sep_shape, fill_value=self.sep_token_id, dtype=source_ids.dtype, device=source_ids.device)
        sep_mask = torch.ones_like(sep_ids, dtype=source_attention_mask.dtype)
        
        concat_ids = torch.cat([source_ids, sep_ids, target_ids_wo_bos], dim=-1)
        concat_attention_mask = torch.cat([source_attention_mask, sep_mask, target_attention_mask_wo_bos], dim=-1)
        
        # Forward pass on concatenated input
        logits = self.model(concat_ids, concat_attention_mask, latent_source, latent_target)
        shift_logits = logits[:, :-1, :]
        
        # Prepare labels: only compute loss on the target portion
        shift_labels_pre = concat_ids[..., 1:]
        source_len = source_ids.shape[-1]
        target_len_wo_bos = target_ids_wo_bos.shape[-1]
        
        # Build a mask that selects only the target amino-acid tokens (target without BOS)
        # TODO: still, make sure this should be source_len + 1?
        labels_mask = torch.zeros_like(shift_labels_pre, dtype=target_attention_mask.dtype)
        labels_mask[..., -target_len_wo_bos:] = target_attention_mask_wo_bos
        
        # TODO: are we sure that -100 is the correct ignore index?
        # Apply mask using ignore_index=-100
        shift_labels = shift_labels_pre.masked_fill(labels_mask == 0, -100)
        
        # Additionally ignore positions where target token is 'X'
        # 'X' represents unknown amino acid; use tokenizer to resolve id
        x_token_id = self.tokenizer.convert_tokens_to_ids('X')
        x_mask = (target_ids_wo_bos == x_token_id)
        shift_labels[..., -target_len_wo_bos:] = shift_labels[..., -target_len_wo_bos:].masked_fill(x_mask, -100)
        
        # If model flattened set dimension, align labels accordingly
        if shift_labels.dim() == 3 and shift_logits.dim() == 3:
            if shift_labels.shape[0] * shift_labels.shape[1] == shift_logits.shape[0]:
                shift_labels = shift_labels.view(-1, shift_labels.shape[-1])
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        loss = loss_fct(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1)
        )
        
        return loss

    def conditional_log_likelihood(self, x_source, x_target, latent_source, latent_target):
        """
        Compute conditional log-likelihood log p(y|x) for target sequences y given source sequences x.

        Args:
            x_source: Dict with keys 'progen_input_ids' and 'progen_attention_mask' for source sequences
            x_target: Dict with keys 'progen_input_ids' and 'progen_attention_mask' for target sequences
            latent_source: Tensor [batch_size, latent_dim]
            latent_target: Tensor [batch_size, latent_dim]

        Returns:
            Tensor of log-likelihoods summed over target tokens.
            Shape is [batch_size] if no set dimension is present, otherwise [batch_size, set_size].
        """
        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"Progen2Generator.conditional_log_likelihood expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        source_ids = x_source['progen_input_ids']
        source_attention_mask = x_source['progen_attention_mask']

        target_ids = x_target['progen_input_ids']
        target_attention_mask = x_target['progen_attention_mask']

        # Remove BOS from target; we condition on source + sep and predict only amino acids
        target_ids_wo_bos = target_ids[..., 1:]
        target_attention_mask_wo_bos = target_attention_mask[..., 1:]

        # Concatenate source + <|endoftext|> + target (without BOS)
        sep_shape = list(source_ids.shape[:-1]) + [1]
        sep_ids = torch.full(sep_shape, fill_value=self.sep_token_id, dtype=source_ids.dtype, device=source_ids.device)
        sep_mask = torch.ones_like(sep_ids, dtype=source_attention_mask.dtype)

        concat_ids = torch.cat([source_ids, sep_ids, target_ids_wo_bos], dim=-1)
        concat_attention_mask = torch.cat([source_attention_mask, sep_mask, target_attention_mask_wo_bos], dim=-1)

        # Forward pass on concatenated input
        logits = self.model(concat_ids, concat_attention_mask, latent_source, latent_target)
        shift_logits = logits[:, :-1, :]  # [B_flat, L-1, V]

        # Labels for next-token prediction
        shift_labels_pre = concat_ids[..., 1:]  # [B or B,S, L-1]

        # Mask: only evaluate log-probs on the target portion (excluding BOS)
        source_len = source_ids.shape[-1]
        target_len_wo_bos = target_ids_wo_bos.shape[-1]

        labels_mask = torch.zeros_like(shift_labels_pre, dtype=target_attention_mask.dtype)
        labels_mask[..., -target_len_wo_bos:] = target_attention_mask_wo_bos

        # Exclude positions where target token is 'X'
        x_token_id = self.tokenizer.convert_tokens_to_ids('X')
        x_mask = (target_ids_wo_bos == x_token_id)  # [B or B,S, target_len_wo_bos]

        # If the model flattened set dimension internally, align labels/masks accordingly
        if shift_labels_pre.dim() == 3 and shift_logits.dim() == 3 and \
           (shift_labels_pre.shape[0] * shift_labels_pre.shape[1] == shift_logits.shape[0]):
            flat_shift_labels = shift_labels_pre.view(-1, shift_labels_pre.shape[-1])
            flat_labels_mask = labels_mask.view(-1, labels_mask.shape[-1])
            flat_x_mask = x_mask.view(-1, x_mask.shape[-1])
            out_shape = (shift_labels_pre.shape[0], shift_labels_pre.shape[1])  # [B, S]
        else:
            flat_shift_labels = shift_labels_pre
            flat_labels_mask = labels_mask
            flat_x_mask = x_mask
            out_shape = None  # [B]

        # Compute log-probs for the correct tokens
        log_probs = F.log_softmax(shift_logits, dim=-1)  # [B_flat, L-1, V]
        token_logps = log_probs.gather(-1, flat_shift_labels.unsqueeze(-1)).squeeze(-1)  # [B_flat, L-1]

        # Build valid mask: only target positions and excluding 'X'
        valid_mask = (flat_labels_mask > 0)
        if target_len_wo_bos > 0:
            valid_tail = valid_mask[..., -target_len_wo_bos:]
            valid_mask = valid_mask.clone()
            valid_mask[..., -target_len_wo_bos:] = valid_tail & (~flat_x_mask)

        # Sum log-probabilities over valid target tokens
        seq_logps = (token_logps * valid_mask.to(token_logps.dtype)).sum(dim=-1)

        # If we flattened set dimension, restore it
        if out_shape is not None:
            seq_logps = seq_logps.view(*out_shape)

        return seq_logps

    def sample(self, x_source, latent_source, latent_target, num_samples=1, return_texts=False):
        """
        Sample sequences from the conditioned model.
        
        Args:
            latent: Latent distribution embedding
            num_samples: Number of samples to generate per latent
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs, and optionally decoded texts
        """
        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"Progen2Generator.sample expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        device = latent_source.device
        # TODO: I think there might be additional weird behavior if we flatten inputs before calling the sample method.
        batch_size = latent_source.size(0)
        
        # Get BOS token ID for start of generation
        # TODO: do we really want to default to bos_token_id = 1 here?
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            # Default to 1 if no BOS token is defined
            bos_token_id = 1
        
        # Generate samples
        all_samples = []
        src_ids_all = x_source['progen_input_ids']
        src_mask_all = x_source['progen_attention_mask']
        for sample_idx in range(num_samples):
            # Add noise for diversity if generating multiple samples
            if num_samples > 1:
                # TODO: how do we know that this is the correct noise scale? It might be too high.
                noise_scale = 0.1
                noisy_latent_source = latent_source + noise_scale * torch.randn_like(latent_source)
                noisy_latent_target = latent_target + noise_scale * torch.randn_like(latent_target)
            else:
                noisy_latent_source = latent_source
                noisy_latent_target = latent_target
            
            # TODO: remove because unnecessary because I already throw an error when inputs were flattened?
            # Build prompt for this sample: choose different source sequences if available
            if src_ids_all.dim() == 3:
                set_size = src_ids_all.shape[1]
                set_idx = sample_idx % set_size
                src_ids = src_ids_all[:, set_idx, :]
                src_mask = src_mask_all[:, set_idx, :]
            else:
                src_ids = src_ids_all
                src_mask = src_mask_all
            
            # Simplified: all source sequences have fixed length equal to self.seq_length
            if src_ids.size(-1) != self.seq_length:
                raise ValueError(
                    f"Expected source sequence length {self.seq_length}, got {src_ids.size(-1)}"
                )
            sep_ids = torch.full((batch_size, 1), fill_value=self.sep_token_id, dtype=src_ids.dtype, device=device)
            start_ids = torch.cat([src_ids, sep_ids], dim=-1)
            start_mask = torch.ones((batch_size, start_ids.size(1)), dtype=src_mask.dtype, device=device)
                
            with torch.no_grad():
                generated_target = self._generate_text(
                    start_ids.clone(),
                    start_mask.clone(),
                    noisy_latent_source,
                    noisy_latent_target,
                    self.seq_length - 1,
                    self.temperature
                )
            # Prepend BOS to match input format [BOS + target_amino_acids]
            bos_ids_batch = torch.full((batch_size, 1), fill_value=bos_token_id, dtype=generated_target.dtype, device=device)
            full_target = torch.cat([bos_ids_batch, generated_target], dim=-1)
            all_samples.append(full_target)
        
        # Combine samples 
        result = torch.stack(all_samples, dim=1)  # [batch_size, num_samples, seq_len]
        
        if return_texts:
            # Decode texts
            all_texts = []
            for batch_idx in range(batch_size):
                batch_texts = []
                for sample_idx in range(num_samples):
                    ids = result[batch_idx, sample_idx]
                    text = self.tokenizer.decode(ids, skip_special_tokens=True)
                    # Make sure we have at least some text
                    if not text.strip():
                        text = "Generated sequence was empty."
                    batch_texts.append(text)
                all_texts.append(batch_texts)
            
            return result, all_texts
        
        return result

    # TODO: will the generated sequences always be exactly 1000 amino acids long?
    # TODO: in general, I need to go over this method in more detail to make sure everything is correct.
    def _generate_text(self, input_ids, attention_mask, latent_source, latent_target, target_length, temperature=1.0):
        """
        Helper method for text generation using the conditioned Progen2 model.
        
        Args:
            input_ids: Starting token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
            target_length: number of target tokens to generate (excluding BOS)
            temperature: Sampling temperature
            
        Returns:
            Generated target token IDs (length == target_length)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize with the starting tokens
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        
        # Build list of allowed amino acid token ids (20 standard AAs)
        standard_amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
            'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        allowed_token_ids = [self.tokenizer.convert_tokens_to_ids(t) for t in standard_amino_acids]
        # Filter out any invalid ids
        allowed_token_ids = [tid for tid in allowed_token_ids if isinstance(tid, int) and tid >= 0]
        if len(allowed_token_ids) != 20:
            raise ValueError("Tokenizer did not return valid ids for standard amino acids; cannot sample.")

        allowed_mask = None  # lazily initialized with correct vocab size

        # Generate exactly target_length tokens for the target sequence
        for _ in range(target_length):
            # Forward pass
            with torch.no_grad():
                logits = self.model(cur_input_ids, cur_attention_mask, latent_source, latent_target)
            
            # Get logits for next token prediction (last position)
            next_token_logits = logits[:, -1, :] / temperature

            # Initialize allowed mask on first step (depends on vocab size)
            if allowed_mask is None:
                vocab_size = next_token_logits.size(-1)
                # Keep only ids within vocab range
                allowed_mask = torch.zeros(vocab_size, dtype=torch.bool, device=next_token_logits.device)
                allowed_mask[torch.tensor(allowed_token_ids, dtype=torch.long, device=next_token_logits.device)] = True

            # Restrict sampling strictly to allowed amino acids
            disallowed = ~allowed_mask
            # TODO: make sure reshaping/expanding is correct here.
            next_token_logits = next_token_logits.masked_fill(disallowed.unsqueeze(0).expand_as(next_token_logits), float('-inf'))
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # Append next token to sequence
            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            
            # Update attention mask
            next_mask = torch.ones_like(next_token)
            cur_attention_mask = torch.cat([cur_attention_mask, next_mask], dim=1)
        
        # Return only the newly generated target tokens
        return cur_input_ids[:, -target_length:]