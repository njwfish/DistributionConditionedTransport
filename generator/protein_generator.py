import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

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
        use_gradient_checkpointing: bool = False,
        precision: str | None = None,
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
        super().__init__()
        
        # Initialize Progen2 model
        # Determine requested dtype early to load weights in that dtype
        requested_dtype = None
        if precision:
            p = precision.lower()
            if p in ("fp16", "half"):
                requested_dtype = torch.float16
            elif p in ("bf16", "bfloat16"):
                requested_dtype = torch.bfloat16
            elif p in ("fp32", "float32"):
                requested_dtype = torch.float32

        self.progen2 = AutoModelForCausalLM.from_pretrained(
            progen2_name,
            trust_remote_code=True,
            torch_dtype=requested_dtype,
            low_cpu_mem_usage=True,
        )

        # TODO: not sure whether this actually helps/works...
        # Memory optimizations
        if use_gradient_checkpointing and hasattr(self.progen2, 'gradient_checkpointing_enable'):
            try:
                self.progen2.gradient_checkpointing_enable()
                # disable cache to allow gradient checkpointing
                if hasattr(self.progen2.config, 'use_cache'):
                    self.progen2.config.use_cache = False
            except Exception:
                pass

        # Set model dtype if requested
        if requested_dtype is not None:
            self.progen2.to(dtype=requested_dtype)
        
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
        else:
            # Default value if none of the above attributes exist
            self.hidden_dim = 768
            print(f"Warning: Could not determine hidden dimension from model config. Using default: {self.hidden_dim}")
        
        self.condition_method = condition_method
        
        # Project latent to correct dimension (same approach as in GPT-2)
        # Note: input dimension is doubled since we concatenate source and target latents
        combined_latent_dim = latent_dim * 2
        
        if self.condition_method == "prefix":
            # For prefix conditioning, project to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(combined_latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
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

        # Align dtype of conditioning projection with model
        if requested_dtype is not None:
            self.condition_proj.to(dtype=requested_dtype)

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
        
        # Ensure latent dtype matches projection weights to avoid matmul dtype mismatch
        proj_weight_dtype = self.condition_proj[0].weight.dtype if isinstance(self.condition_proj, nn.Sequential) else combined_latent.dtype
        combined_latent = combined_latent.to(proj_weight_dtype)
        # Project combined latent to correct dimension
        condition = self.condition_proj(combined_latent)
        
        if self.condition_method == "prefix":
            # Use the condition as a prefix hidden state
            # Create a prefix token
            
            # Handle attention mask dimension properly - check shape and process individually for memory efficiency
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # MEMORY OPTIMIZATION: Process sequences individually instead of batching
                set_size, seq_len = attention_mask.shape[1:]
                all_logits = []
                
                for seq_idx in range(set_size):
                    # Process each sequence in the set individually
                    seq_input_ids = input_ids[:, seq_idx, :]  # [batch_size, seq_len]
                    seq_attention_mask = attention_mask[:, seq_idx, :]  # [batch_size, seq_len]
                    
                    # Process this individual sequence
                    seq_logits = self._forward_single_sequence(
                        seq_input_ids, seq_attention_mask, condition, method="prefix"
                    )
                    all_logits.append(seq_logits)
                
                # Concatenate results back to [batch_size * set_size, seq_len, vocab_size]
                logits = torch.cat(all_logits, dim=0)
                
            else:
                # Standard processing for single sequences
                logits = self._forward_single_sequence(input_ids, attention_mask, condition, method="prefix")
            
        elif self.condition_method == "additive":
            # Handle attention mask dimension properly - process individually for memory efficiency
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # MEMORY OPTIMIZATION: Process sequences individually instead of batching
                set_size, seq_len = attention_mask.shape[1:]
                all_logits = []
                
                for seq_idx in range(set_size):
                    # Process each sequence in the set individually
                    seq_input_ids = input_ids[:, seq_idx, :]  # [batch_size, seq_len]
                    seq_attention_mask = attention_mask[:, seq_idx, :]  # [batch_size, seq_len]
                    
                    # Process this individual sequence
                    seq_logits = self._forward_single_sequence(
                        seq_input_ids, seq_attention_mask, condition, method="additive"
                    )
                    all_logits.append(seq_logits)
                
                # Concatenate results back to [batch_size * set_size, seq_len, vocab_size]
                logits = torch.cat(all_logits, dim=0)
                
            else:
                # Standard processing for single sequences
                logits = self._forward_single_sequence(input_ids, attention_mask, condition, method="additive")
        
        return logits

    def _forward_single_sequence(self, input_ids, attention_mask, condition, method="prefix"):
        """
        Memory-optimized forward pass for a single sequence.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            condition: Projected condition [batch_size, hidden_dim]
            method: "prefix" or "additive"
            
        Returns:
            Logits for the sequence [batch_size, seq_len, vocab_size]
        """
        if method == "prefix":
            batch_size = input_ids.shape[0]
            
            # Add prefix to attention mask
            prefix_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
            extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            
            # Get embeddings for the input sequence
            if hasattr(self.progen2.transformer, 'wte'):
                token_embeds = self.progen2.transformer.wte(input_ids)
            else:
                token_embeds = self.progen2.get_input_embeddings()(input_ids)
            
            # Match dtype and create prefix embedding
            condition = condition.to(token_embeds.dtype)
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Concatenate prefix embedding with input embeddings
            combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            
            # Run through Progen2 with custom embeddings
            outputs = self.progen2(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            
            # Get logits and remove the prefix logit
            logits = outputs.logits[:, 1:, :]
            
        elif method == "additive":
            # Process with the model first
            outputs = self.progen2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
            
            # Get the final hidden states
            hidden_states = outputs.hidden_states[-1]
            
            # Add the condition to each position
            condition = condition.to(hidden_states.dtype)
            condition_broadcast = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            hidden_states = hidden_states + condition_broadcast
            
            # Project back to vocabulary
            logits = self.progen2.lm_head(hidden_states)
            
        return logits


class Progen2Generator(nn.Module):
    """Generator class using conditioned Progen2 for the distribution embeddings framework."""
    
    def __init__(
        self,
        progen2_name="hugohrban/progen2-medium",
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix",
        temperature=1.0,
        max_length=512,
        use_gradient_checkpointing: bool = False,
        precision: str | None = None,
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
            max_length: Maximum length for generation
        """
        super().__init__()

        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_progen2=freeze_progen2,
            condition_method=condition_method,
            use_gradient_checkpointing=use_gradient_checkpointing,
            precision=precision,
        )
        
        self.temperature = temperature
        self.max_length = max_length
        
        # Initialize tokenizer (for generation)
        self.tokenizer = AutoTokenizer.from_pretrained(progen2_name, trust_remote_code=True)
        
        # Add special tokens if they don't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|pad|>'
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<|bos|>'
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '<|eos|>'
    
    def loss(self, x_source, x_target, latent_source, latent_target):
        """
        Calculate the loss for the generator.
        
        Args:
            x: Dictionary containing 'progen_input_ids' and 'progen_attention_mask'
            latent: Latent distribution embedding
        
        Returns:
            Negative log likelihood loss
        """
        source_ids = x_source['progen_input_ids']
        source_attention_mask = x_source['progen_attention_mask']
        
        target_ids = x_target['progen_input_ids']
        # TODO: is it correct that we have no use for the target attention mask?
        target_attention_mask = x_target['progen_attention_mask']
        
        # TODO: Figure out whether you want to do seq to seq or mask to seq.
        # Shift for causal language modeling: predict each token using previous tokens
        #logits = self.model(source_ids, source_attention_mask, latent_source, latent_target)
        logits = self.model(target_ids, target_attention_mask, latent_source, latent_target)
        shift_logits = logits[:, :-1, :]
        
        # Reshape input_ids if needed to match logits
        if len(target_ids.shape) == 3 and len(shift_logits.shape) == 3:
            if target_ids.shape[0] * target_ids.shape[1] == shift_logits.shape[0]:
                # Reshape input_ids to match the reshaped logits
                target_ids = source_ids.view(-1, target_ids.shape[-1])
        
        shift_labels = target_ids[:, 1:]
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        # If model is in reduced precision, temporarily compute loss in fp32 for stability
        logits_dtype = shift_logits.dtype
        if logits_dtype in (torch.float16, torch.bfloat16):
            loss = loss_fct(
                shift_logits.float().reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        else:
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
        
        return loss

    # TODO: currently this is not being conditioned on the source samples, but I think that's fine since the PLM doesn't need it.
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
        device = latent_source.device
        batch_size = latent_source.size(0)
        
        # Get BOS token ID for start of generation
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            bos_token_id = self.tokenizer.bos_token_id
        else:
            # Default to 1 if no BOS token is defined
            bos_token_id = 1
        
        # Initialize with the starting tokens
        start_ids = torch.tensor([[bos_token_id]] * batch_size, device=device)
        start_mask = torch.ones_like(start_ids)
        
        # Generate samples
        all_samples = []
        for _ in range(num_samples):
            # Add noise for diversity if generating multiple samples
            if num_samples > 1:
                noise_scale = 0.1
                noisy_latent_source = latent_source + noise_scale * torch.randn_like(latent_source)
                noisy_latent_target = latent_target + noise_scale * torch.randn_like(latent_target)
            else:
                noisy_latent_source = latent_source
                noisy_latent_target = latent_target
                
            with torch.no_grad():
                out = self._generate_text(
                    start_ids.clone(),
                    start_mask.clone(),
                    noisy_latent_source,
                    noisy_latent_target,
                    self.max_length,
                    self.temperature
                )
            all_samples.append(out)
        
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

    def _generate_text(self, input_ids, attention_mask, latent_source, latent_target, max_length, temperature=1.0):
        """
        Helper method for text generation using the conditioned Progen2 model.
        
        Args:
            input_ids: Starting token IDs
            attention_mask: Attention mask
            latent: Latent distribution embedding
            max_length: Maximum sequence length
            temperature: Sampling temperature
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize with the starting tokens
        cur_input_ids = input_ids
        cur_attention_mask = attention_mask
        
        # Get EOS token ID for stopping generation
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Generate tokens up to max_length or until all sequences have EOS
        for _ in range(max_length - cur_input_ids.size(1)):
            # Forward pass
            with torch.no_grad():
                logits = self.model(cur_input_ids, cur_attention_mask, latent_source, latent_target)
            
            # Get logits for next token prediction (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample next token
            next_token = torch.multinomial(probs, 1)
            
            # If a sequence is finished, use EOS token
            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(1),
                    torch.full_like(next_token, eos_token_id),
                    next_token
                )
            
            # Append next token to sequence
            cur_input_ids = torch.cat([cur_input_ids, next_token], dim=1)
            
            # Update attention mask
            next_mask = torch.ones_like(next_token)
            cur_attention_mask = torch.cat([cur_attention_mask, next_mask], dim=1)
            
            # Mark sequences as finished if EOS token is generated
            if eos_token_id is not None:
                finished = finished | (next_token.squeeze(-1) == eos_token_id)
                
                # If all sequences are finished, stop generation
                if finished.all():
                    break
        
        return cur_input_ids