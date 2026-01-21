import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

class ConditionedProgen2(nn.Module):
    """Progen2 model conditioned on source sequence and latent vectors for source-to-target protein generation."""
    
    def __init__(
        self, 
        progen2_name='hugohrban/progen2-medium', 
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix"
    ):
        """
        Initialize a conditioned Progen2 model for source-to-target generation.
        
        Args:
            progen2_name: Name of the pretrained Progen2 model
            latent_dim: Dimension of the latent distribution embedding
            condition_dim: Dimension to project the condition to
            freeze_progen2: Whether to freeze the Progen2 parameters
            condition_method: How to condition the model ('prefix' or 'additive')
        """
        super().__init__()
        
        # Initialize Progen2 model
        self.progen2 = AutoModelForCausalLM.from_pretrained(progen2_name, trust_remote_code=True)
        
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
        
        # Project combined latents (source + target) to correct dimension
        # We concatenate latent_source and latent_target, so input dim is 2 * latent_dim
        if self.condition_method == "prefix":
            # For prefix conditioning, project combined latents to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim * 2, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        elif self.condition_method == "additive":
            # For additive conditioning, project combined latents to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim * 2, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.hidden_dim)
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")

    def forward(self, input_ids, attention_mask, latent_source, latent_target, target_start_idx=None):
        """
        Forward pass through the conditioned Progen2 model.
        
        Args:
            input_ids: Tensor of token IDs [batch_size, seq_len] (concatenated: source + target)
                       Format: <|bos|>SOURCE<|eos|><|bos|>TARGET<|eos|>
            attention_mask: Attention mask [batch_size, seq_len]
            latent_source: Latent distribution embedding for source [batch_size, latent_dim]
            latent_target: Latent distribution embedding for target [batch_size, latent_dim]
            target_start_idx: Index where target sequence starts (for prefix insertion)
            
        Returns:
            Logits for next token prediction
        """
        batch_size = input_ids.shape[0]
        
        # Combine latent_source and latent_target by concatenation
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)  # [batch_size, latent_dim * 2]
        
        # Project combined latent to correct dimension
        condition = self.condition_proj(combined_latent)
        
        if self.condition_method == "prefix":
            # Use the condition as a prefix hidden state inserted right before target
            
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # Reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
            # Get embeddings for the input sequence (directly from token embeddings or wte)
            if hasattr(self.progen2.transformer, 'wte'):
                # Usually Progen2 uses wte for word token embeddings
                token_embeds = self.progen2.transformer.wte(input_ids)
            else:
                # Fallback to manual embedding lookup
                token_embeds = self.progen2.get_input_embeddings()(input_ids)
            
            # Create prefix embedding
            prefix_embeds = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            if target_start_idx is not None:
                # Insert prefix right before target sequence
                # Structure: <|bos|>SOURCE<|eos|><|pad|>... [PREFIX] <|bos|>TARGET<|eos|>
                source_embeds = token_embeds[:, :target_start_idx, :]
                target_embeds = token_embeds[:, target_start_idx:, :]
                combined_embeds = torch.cat([source_embeds, prefix_embeds, target_embeds], dim=1)
                
                # Update attention mask similarly
                source_mask = attention_mask[:, :target_start_idx]
                target_mask = attention_mask[:, target_start_idx:]
                prefix_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                extended_attention_mask = torch.cat([source_mask, prefix_attention, target_mask], dim=1)
            else:
                # Fallback: prepend prefix to entire sequence (for sampling)
                combined_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
                prefix_attention = torch.ones(batch_size, 1, dtype=attention_mask.dtype, device=attention_mask.device)
                extended_attention_mask = torch.cat([prefix_attention, attention_mask], dim=1)
            
            # Run through Progen2 with custom embeddings
            outputs = self.progen2(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                return_dict=True
            )
            
            # Get logits and remove the prefix logit
            # The prefix was inserted at target_start_idx, so we need to remove that position
            if target_start_idx is not None:
                # Remove the logit at the prefix position
                logits_before = outputs.logits[:, :target_start_idx, :]
                logits_after = outputs.logits[:, target_start_idx + 1:, :]
                logits = torch.cat([logits_before, logits_after], dim=1)
            else:
                # Prefix was at the beginning, remove first logit
                logits = outputs.logits[:, 1:, :]
            
        elif self.condition_method == "additive":
            # Handle attention mask dimension properly - check shape and reshape if needed
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                # Reshape if needed
                set_size, seq_len = attention_mask.shape[1:]
                attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                input_ids = input_ids.view(batch_size * set_size, seq_len)
                condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
                batch_size = batch_size * set_size
            
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
            condition = condition.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            hidden_states = hidden_states + condition
            
            # Project back to vocabulary
            logits = self.progen2.lm_head(hidden_states)
        
        return logits


class Progen2Generator(nn.Module):
    """Generator class using conditioned Progen2 for source-to-target protein generation."""
    
    def __init__(
        self,
        progen2_name="hugohrban/progen2-medium",
        latent_dim=32,
        condition_dim=256,
        freeze_progen2=False,
        condition_method="prefix",
        temperature=1.0,
        max_length=128
    ):
        """
        Initialize the Progen2 generator for source-to-target generation.
        
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
        self.temperature = temperature
        self.max_length = max_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(progen2_name, trust_remote_code=True)
        
        # Set special tokens to match the dataset's tokenizer configuration
        # (unconditional assignment to ensure consistency)
        self.tokenizer.pad_token = '<|pad|>'
        self.tokenizer.bos_token = '<|bos|>'
        self.tokenizer.eos_token = '<|eos|>'
        
        # Initialize model
        # Note: No separator token needed - source and target sequences are already
        # tokenized with BOS/EOS tokens, so concatenation yields:
        # <|bos|>SOURCE<|eos|><|bos|>TARGET<|eos|>
        # The EOS-BOS boundary clearly separates source from target.
        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_progen2=freeze_progen2,
            condition_method=condition_method
        )
    
    def _concatenate_source_target(self, x_source, x_target):
        """
        Concatenate source and target sequences.
        
        Since both source and target are already tokenized with BOS/EOS tokens,
        the concatenation yields: <|bos|>SOURCE<|eos|><|bos|>TARGET<|eos|>
        The EOS-BOS boundary naturally separates source from target.
        
        Args:
            x_source: Dictionary containing 'progen_input_ids' and 'progen_attention_mask' for source
            x_target: Dictionary containing 'progen_input_ids' and 'progen_attention_mask' for target
            
        Returns:
            combined_input_ids: Concatenated input IDs [batch_size, source_len + target_len]
            combined_attention_mask: Concatenated attention mask
            target_start_idx: Index where the target sequence starts
        """
        source_ids = x_source['progen_input_ids']
        source_mask = x_source['progen_attention_mask']
        target_ids = x_target['progen_input_ids']
        target_mask = x_target['progen_attention_mask']
        
        # Handle 3D tensors (batch_size, set_size, seq_len)
        if len(source_ids.shape) == 3:
            # Concatenate along the sequence dimension (dim=2)
            combined_input_ids = torch.cat([source_ids, target_ids], dim=2)
            combined_attention_mask = torch.cat([source_mask, target_mask], dim=2)
        else:
            # Concatenate: source + target
            combined_input_ids = torch.cat([source_ids, target_ids], dim=1)
            combined_attention_mask = torch.cat([source_mask, target_mask], dim=1)
        
        # Target starts right after source
        # For 3D tensors, seq_len is shape[2]; for 2D, it's shape[1]
        source_seq_len = source_ids.shape[2] if len(source_ids.shape) == 3 else source_ids.shape[1]
        target_start_idx = source_seq_len
        
        return combined_input_ids, combined_attention_mask, target_start_idx
    
    def loss(self, x_source, x_target, latent_source, latent_target):
        """
        Calculate the loss for the generator.

        The model is conditioned on x_source, latent_source, and latent_target,
        and the loss measures how well it generates x_target.

        Args:
            x_source: Dictionary containing 'progen_input_ids' and 'progen_attention_mask' for source
            x_target: Dictionary containing 'progen_input_ids' and 'progen_attention_mask' for target
            latent_source: Latent distribution embedding for source [batch_size, latent_dim]
            latent_target: Latent distribution embedding for target [batch_size, latent_dim] (can be None)

        Returns:
            Negative log likelihood loss (computed only on target tokens)
        """
        # If target latent is None, use source latent (source-only conditioning)
        if latent_target is None:
            latent_target = latent_source

        # Concatenate source and target sequences
        combined_input_ids, combined_attention_mask, target_start_idx = self._concatenate_source_target(
            x_source, x_target
        )

        # Handle potential 3D tensor shapes (batch_size, set_size, seq_len)
        original_shape = None
        if len(combined_input_ids.shape) == 3:
            original_shape = combined_input_ids.shape
            batch_size, set_size, seq_len = original_shape
            combined_input_ids = combined_input_ids.view(batch_size * set_size, seq_len)
            combined_attention_mask = combined_attention_mask.view(batch_size * set_size, seq_len)
            latent_source = latent_source.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
            latent_target = latent_target.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)
        
        # Forward pass through the model
        # Pass target_start_idx so prefix is inserted right before target
        logits = self.model(combined_input_ids, combined_attention_mask, latent_source, latent_target, target_start_idx)
        
        # For causal LM: shift logits and labels
        # We only want to compute loss on the target portion
        # logits[:, i, :] predicts token at position i+1
        # So to predict target tokens (starting at target_start_idx), 
        # we need logits from position (target_start_idx - 1) onwards
        
        # Get the logits that predict target tokens
        # logits[:, target_start_idx-1, :] predicts the first target token
        target_logits = logits[:, target_start_idx - 1:-1, :]  # Predictions for target tokens
        
        # Get the target labels (the actual target token IDs)
        target_labels = combined_input_ids[:, target_start_idx:]  # Target token IDs
        

        # Calculate loss only on target tokens
        loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_labels.reshape(-1)
        )
        
        return loss

    def sample(self, x_source, latent_source, latent_target, num_samples=1, return_texts=False):
        """
        Sample target sequences conditioned on source sequence and latent embeddings.
        
        Args:
            x_source: Dictionary containing 'progen_input_ids' and 'progen_attention_mask' for source
            latent_source: Latent distribution embedding for source [batch_size, latent_dim]
            latent_target: Latent distribution embedding for target [batch_size, latent_dim]
            num_samples: Number of samples to generate per input
            return_texts: Whether to also return decoded texts
        
        Returns:
            Generated token IDs (target only), and optionally decoded texts
        """
        device = latent_source.device
        batch_size = latent_source.size(0)
        
        source_ids = x_source['progen_input_ids']
        source_mask = x_source['progen_attention_mask']
        
        # Track where target starts (right after source)
        # This is where the prefix will be inserted, consistent with training
        source_len = source_ids.shape[1]
        target_start_idx = source_len
        
        # Create the starting sequence: source + BOS token for target
        # Source is: <|bos|>SOURCE<|eos|><|pad|>...
        # We start target generation with: <|bos|>
        bos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.bos_token)
        bos_token = torch.full((batch_size, 1), bos_token_id, dtype=source_ids.dtype, device=device)
        bos_mask = torch.ones((batch_size, 1), dtype=source_mask.dtype, device=device)
        
        start_ids = torch.cat([source_ids, bos_token], dim=1)
        start_mask = torch.cat([source_mask, bos_mask], dim=1)
        
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
                    self.temperature,
                    target_start_idx
                )
            # Extract only the generated target portion (after source)
            # The BOS token we added is now part of the target
            target_out = out[:, target_start_idx:]
            all_samples.append(target_out)
        
        # Pad samples to same length and combine
        max_len = max(s.shape[1] for s in all_samples)
        padded_samples = []
        for s in all_samples:
            if s.shape[1] < max_len:
                pad_len = max_len - s.shape[1]
                pad_tensor = torch.full(
                    (s.shape[0], pad_len), 
                    self.tokenizer.pad_token_id, 
                    dtype=s.dtype, 
                    device=s.device
                )
                s = torch.cat([s, pad_tensor], dim=1)
            padded_samples.append(s)
        
        result = torch.stack(padded_samples, dim=1)  # [batch_size, num_samples, seq_len]
        
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

    def _generate_text(self, input_ids, attention_mask, latent_source, latent_target, max_target_length, temperature=1.0, target_start_idx=None):
        """
        Helper method for text generation using the conditioned Progen2 model.
        
        Args:
            input_ids: Starting token IDs (source + BOS token for target)
            attention_mask: Attention mask
            latent_source: Latent distribution embedding for source
            latent_target: Latent distribution embedding for target
            max_target_length: Maximum length for the generated target sequence
            temperature: Sampling temperature
            target_start_idx: Index where target starts (for prefix insertion)
            
        Returns:
            Generated token IDs (full sequence including source + generated target)
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
        
        # Generate tokens up to max_target_length or until all sequences have EOS
        # max_target_length refers to the target portion only
        for _ in range(max_target_length):
            # Forward pass with target_start_idx for proper prefix positioning
            with torch.no_grad():
                logits = self.model(cur_input_ids, cur_attention_mask, latent_source, latent_target, target_start_idx)
            
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