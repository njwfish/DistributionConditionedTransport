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
        use_gradient_checkpointing: bool = False,
        precision: Optional[str] = None,
        forward_microbatch_size: Optional[int] = None,
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
        debug_logger = logging.getLogger('debug_performance')
        debug_logger.info(f"=== ConditionedProgen2.__init__ STARTED ===")
        debug_logger.info(f"Model name: {progen2_name}")
        
        # Check environment for potential issues
        debug_logger.info(f"HF_HOME: {os.environ.get('HF_HOME', 'not set')}")
        debug_logger.info(f"TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', 'not set')}")
        debug_logger.info(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE', 'not set')}")
        
        super().__init__()
        
        # Initialize Progen2 model
        # Determine requested dtype early to load weights in that dtype
        debug_logger.info("Determining precision settings...")
        requested_dtype = None
        if precision:
            p = precision.lower()
            if p in ("fp16", "half"):
                requested_dtype = torch.float16
            elif p in ("bf16", "bfloat16"):
                requested_dtype = torch.bfloat16
            elif p in ("fp32", "float32"):
                requested_dtype = torch.float32
        debug_logger.info(f"Requested dtype: {requested_dtype}")

        # Check if model might be cached locally
        from transformers import AutoConfig
        try:
            debug_logger.info("Checking if model config is available locally...")
            config_check_start = time.time()
            config = AutoConfig.from_pretrained(resolve_local_or_repo(progen2_name), trust_remote_code=True, local_files_only=True)
            debug_logger.info(f"Model config found locally (took {time.time() - config_check_start:.2f}s)")
        except Exception:
            debug_logger.info("Model not found locally - will need to download")
        
        debug_logger.info(f"Starting AutoModelForCausalLM.from_pretrained for {progen2_name}...")
        model_load_start = time.time()
        
        # Add more explicit loading parameters that might help with speed
        try:
            self.progen2 = AutoModelForCausalLM.from_pretrained(
                resolve_local_or_repo(progen2_name),
                trust_remote_code=True,
                torch_dtype=requested_dtype,
                device_map=None,  # Don't automatically place on GPU yet
                local_files_only=False,  # Allow downloading if needed
                resume_download=True,  # Resume interrupted downloads
            )
            debug_logger.info(f"ProGen2 model loaded successfully (took {time.time() - model_load_start:.2f}s)")
        except Exception as e:
            debug_logger.error(f"Failed to load ProGen2 model: {e}")
            debug_logger.info("Trying fallback loading strategy...")
            # Fallback with minimal parameters
            self.progen2 = AutoModelForCausalLM.from_pretrained(
                resolve_local_or_repo(progen2_name),
                trust_remote_code=True,
            )
            debug_logger.info(f"ProGen2 model loaded with fallback (took {time.time() - model_load_start:.2f}s)")

        # TODO: not sure whether this actually helps/works...
        # Memory optimizations
        debug_logger.info("Applying memory optimizations...")
        # Always disable KV cache in training graphs to avoid large retained states
        try:
            if hasattr(self.progen2, 'config') and hasattr(self.progen2.config, 'use_cache'):
                self.progen2.config.use_cache = False
                debug_logger.info("Disabled use_cache on ProGen2 config")
        except Exception as e:
            debug_logger.warning(f"Failed to set use_cache=False: {e}")
        if use_gradient_checkpointing and hasattr(self.progen2, 'gradient_checkpointing_enable'):
            try:
                self.progen2.gradient_checkpointing_enable()
                # disable cache to allow gradient checkpointing
                if hasattr(self.progen2.config, 'use_cache'):
                    self.progen2.config.use_cache = False
                debug_logger.info("Gradient checkpointing enabled")
            except Exception as e:
                debug_logger.warning(f"Failed to enable gradient checkpointing: {e}")

        # Set model dtype if requested
        if requested_dtype is not None:
            debug_logger.info(f"Converting model to dtype: {requested_dtype}")
            self.progen2.to(dtype=requested_dtype)
        
        # Freeze Progen2 if specified
        if freeze_progen2:
            debug_logger.info("Freezing ProGen2 parameters...")
            for param in self.progen2.parameters():
                param.requires_grad = False
            debug_logger.info("ProGen2 parameters frozen")
        
        # Get the embedding dimension from the model config
        # Different models might use different attribute names
        debug_logger.info("Determining hidden dimension from model config...")
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
            debug_logger.warning(f"Could not determine hidden dimension from model config. Using default: {self.hidden_dim}")
        
        debug_logger.info(f"Hidden dimension: {self.hidden_dim}")
        self.condition_method = condition_method
        # Microbatch size controls how many sequences we process at a time during forward
        self.forward_microbatch_size = int(forward_microbatch_size) if (forward_microbatch_size is not None and int(forward_microbatch_size) > 0) else None
        
        # Project latent to correct dimension (same approach as in GPT-2)
        # Note: input dimension is doubled since we concatenate source and target latents
        combined_latent_dim = latent_dim * 2
        debug_logger.info(f"Creating conditioning projection network (input_dim={combined_latent_dim}, condition_dim={condition_dim}, hidden_dim={self.hidden_dim})...")
        
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
            debug_logger.info(f"Setting conditioning projection dtype to: {requested_dtype}")
            self.condition_proj.to(dtype=requested_dtype)
        
        debug_logger.info("=== ConditionedProgen2.__init__ COMPLETED ===")        

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
        # Debug logging at forward entry
        debug_logger = get_debug_logger()
        debug_logger.log_memory("GEN_FORWARD_ENTRY", "Generator forward() method entry")
        debug_logger.log_tensor_info("GEN_INPUT_IDS", "input_ids", input_ids)
        debug_logger.log_tensor_info("GEN_ATTN_MASK", "attention_mask", attention_mask)
        debug_logger.log_tensor_info("GEN_LATENT_SRC", "latent_source", latent_source)
        debug_logger.log_tensor_info("GEN_LATENT_TGT", "latent_target", latent_target)
        
        import logging
        import time
        debug_logger = logging.getLogger('debug_performance')
        forward_start = time.time()
        
        batch_size = input_ids.shape[0]
        debug_logger.debug(f"Generator forward: input shape {input_ids.shape}, batch_size {batch_size}")
        
        # Combine the two latents - concatenate them to create a richer conditioning signal
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)
        
        # Ensure latent dtype matches projection weights to avoid matmul dtype mismatch
        proj_weight_dtype = self.condition_proj[0].weight.dtype if isinstance(self.condition_proj, nn.Sequential) else combined_latent.dtype
        combined_latent = combined_latent.to(proj_weight_dtype)
        # Project combined latent to correct dimension
        condition = self.condition_proj(combined_latent)
        
        if self.condition_method == "prefix":

            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                set_size, seq_len = attention_mask.shape[1:]
                # Flatten set dimension
                reshaped_input_ids = input_ids.view(batch_size * set_size, seq_len)
                reshaped_attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                expanded_condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)

                # Microbatch across the flattened batch if requested to control memory
                if self.forward_microbatch_size is not None and self.forward_microbatch_size < reshaped_input_ids.shape[0]:
                    outputs_list = []
                    for start in range(0, reshaped_input_ids.shape[0], self.forward_microbatch_size):
                        end = min(start + self.forward_microbatch_size, reshaped_input_ids.shape[0])
                        # Lazily move only the micro-batch to the model device to reduce peak memory
                        device = next(self.progen2.parameters()).device
                        ids_mb = reshaped_input_ids[start:end].to(device, non_blocking=True)
                        mask_mb = reshaped_attention_mask[start:end].to(device, non_blocking=True)
                        cond_mb = expanded_condition[start:end].to(device, non_blocking=True)
                        out_chunk = self._forward_single_sequence(
                            ids_mb,
                            mask_mb,
                            cond_mb,
                            method="prefix",
                        )
                        # Move output to CPU immediately to free GPU memory
                        outputs_list.append(out_chunk.cpu())
                        # free refs promptly
                        del ids_mb, mask_mb, cond_mb, out_chunk
                        # Force memory cleanup
                        torch.cuda.empty_cache()
                    # Concatenate on CPU then move back to GPU  
                    device = next(self.progen2.parameters()).device
                    logits = torch.cat(outputs_list, dim=0).to(device)
                else:
                    logits = self._forward_single_sequence(
                        reshaped_input_ids, reshaped_attention_mask, expanded_condition, method="prefix"
                    )
            else:
                # Microbatch across batch dimension if needed
                if self.forward_microbatch_size is not None and self.forward_microbatch_size < input_ids.shape[0]:
                    outputs_list = []
                    for start in range(0, input_ids.shape[0], self.forward_microbatch_size):
                        end = min(start + self.forward_microbatch_size, input_ids.shape[0])
                        device = next(self.progen2.parameters()).device
                        ids_mb = input_ids[start:end].to(device, non_blocking=True)
                        mask_mb = attention_mask[start:end].to(device, non_blocking=True)
                        cond_mb = condition[start:end].to(device, non_blocking=True)
                        out_chunk = self._forward_single_sequence(
                            ids_mb, mask_mb, cond_mb, method="prefix"
                        )
                        # Move output to CPU immediately to free GPU memory
                        outputs_list.append(out_chunk.cpu())
                        del ids_mb, mask_mb, cond_mb, out_chunk
                        # Force memory cleanup
                        torch.cuda.empty_cache()
                    # Concatenate on CPU then move back to GPU
                    device = next(self.progen2.parameters()).device
                    logits = torch.cat(outputs_list, dim=0).to(device)
                else:
                    logits = self._forward_single_sequence(input_ids, attention_mask, condition, method="prefix")
            
        elif self.condition_method == "additive":
            if len(attention_mask.shape) == 3:  # [batch_size, set_size, seq_len]
                set_size, seq_len = attention_mask.shape[1:]
                reshaped_input_ids = input_ids.view(batch_size * set_size, seq_len)
                reshaped_attention_mask = attention_mask.view(batch_size * set_size, seq_len)
                expanded_condition = condition.unsqueeze(1).repeat(1, set_size, 1).view(batch_size * set_size, -1)

                if self.forward_microbatch_size is not None and self.forward_microbatch_size < reshaped_input_ids.shape[0]:
                    outputs_list = []
                    for start in range(0, reshaped_input_ids.shape[0], self.forward_microbatch_size):
                        end = min(start + self.forward_microbatch_size, reshaped_input_ids.shape[0])
                        device = next(self.progen2.parameters()).device
                        ids_mb = reshaped_input_ids[start:end].to(device, non_blocking=True)
                        mask_mb = reshaped_attention_mask[start:end].to(device, non_blocking=True)
                        cond_mb = expanded_condition[start:end].to(device, non_blocking=True)
                        out_chunk = self._forward_single_sequence(
                            ids_mb,
                            mask_mb,
                            cond_mb,
                            method="additive",
                        )
                        # Move output to CPU immediately to free GPU memory
                        outputs_list.append(out_chunk.cpu())
                        del ids_mb, mask_mb, cond_mb, out_chunk
                        # Force memory cleanup
                        torch.cuda.empty_cache()
                    # Concatenate on CPU then move back to GPU
                    device = next(self.progen2.parameters()).device
                    logits = torch.cat(outputs_list, dim=0).to(device)
                else:
                    logits = self._forward_single_sequence(
                        reshaped_input_ids, reshaped_attention_mask, expanded_condition, method="additive"
                    )
            else:
                if self.forward_microbatch_size is not None and self.forward_microbatch_size < input_ids.shape[0]:
                    outputs_list = []
                    for start in range(0, input_ids.shape[0], self.forward_microbatch_size):
                        end = min(start + self.forward_microbatch_size, input_ids.shape[0])
                        device = next(self.progen2.parameters()).device
                        ids_mb = input_ids[start:end].to(device, non_blocking=True)
                        mask_mb = attention_mask[start:end].to(device, non_blocking=True)
                        cond_mb = condition[start:end].to(device, non_blocking=True)
                        out_chunk = self._forward_single_sequence(
                            ids_mb, mask_mb, cond_mb, method="additive"
                        )
                        # Move output to CPU immediately to free GPU memory
                        outputs_list.append(out_chunk.cpu())
                        del ids_mb, mask_mb, cond_mb, out_chunk
                        # Force memory cleanup
                        torch.cuda.empty_cache()
                    # Concatenate on CPU then move back to GPU
                    device = next(self.progen2.parameters()).device
                    logits = torch.cat(outputs_list, dim=0).to(device)
                else:
                    logits = self._forward_single_sequence(input_ids, attention_mask, condition, method="additive")
        
        debug_logger.debug(f"Generator forward completed in {time.time() - forward_start:.3f}s")
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
            
            # Debug logging before ProGen2 call
            debug_logger = get_debug_logger()
            debug_logger.log_memory("PROGEN_PRE", "Before ProGen2 model call")
            debug_logger.log_tensor_info("PROGEN_EMBEDS", "combined_embeds", combined_embeds)
            debug_logger.log_tensor_info("PROGEN_MASK", "extended_attention_mask", extended_attention_mask)
            
            # Run through Progen2 with custom embeddings - CRITICAL MEMORY POINT
            # Enforce no cache and avoid returning hidden states to reduce memory
            debug_logger.log_memory("PROGEN_START", "Calling ProGen2 model")
            outputs = self.progen2(
                inputs_embeds=combined_embeds,
                attention_mask=extended_attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True
            )
            debug_logger.log_memory("PROGEN_DONE", "ProGen2 model call complete")
            
            # Get logits and remove the prefix logit
            logits = outputs.logits[:, 1:, :]
            
        elif method == "additive":
            # Process with the model first
            outputs = self.progen2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
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
        precision: Optional[str] = None,
        forward_microbatch_size: Optional[int] = None,
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
        import logging
        import time
        debug_logger = logging.getLogger('debug_performance')
        debug_logger.info("=== Progen2Generator.__init__ STARTED ===")
        
        super().__init__()

        debug_logger.info("Creating ConditionedProgen2 model...")
        conditioned_model_start = time.time()
        self.model = ConditionedProgen2(
            progen2_name=progen2_name,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            freeze_progen2=freeze_progen2,
            condition_method=condition_method,
            use_gradient_checkpointing=use_gradient_checkpointing,
            precision=precision,
            forward_microbatch_size=forward_microbatch_size,
        )
        debug_logger.info(f"ConditionedProgen2 model created (took {time.time() - conditioned_model_start:.2f}s)")
        
        self.temperature = temperature
        self.max_length = max_length
        
        # Initialize tokenizer (for generation)
        debug_logger.info(f"Loading tokenizer for {progen2_name}...")
        tokenizer_start = time.time()
        self.tokenizer = AutoTokenizer.from_pretrained(resolve_local_or_repo(progen2_name), trust_remote_code=True)
        debug_logger.info(f"Tokenizer loaded (took {time.time() - tokenizer_start:.2f}s)")
        
        # Add special tokens if they don't exist
        debug_logger.info("Setting up special tokens...")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = '<|pad|>'
        if self.tokenizer.bos_token is None:
            self.tokenizer.bos_token = '<|bos|>'
        if self.tokenizer.eos_token is None:
            self.tokenizer.eos_token = '<|eos|>'
        debug_logger.info("Special tokens configured")
        # Resolve separator token id ('<|endoftext|>') and pad token id
        self.sep_token = '<|endoftext|>'
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.sep_token)
        # TODO: is this correct?
        self.pad_token_id = self.tokenizer.pad_token_id if (hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None) else 0
        
        debug_logger.info(f"Separator token id: {self.sep_token_id}; Pad token id: {self.pad_token_id}")
        debug_logger.info("=== Progen2Generator.__init__ COMPLETED ===")        
    
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
        target_attention_mask = x_target['progen_attention_mask']
        
        # Concatenate source + <|endoftext|> + target for conditioning on source content
        sep_shape = list(source_ids.shape[:-1]) + [1]
        sep_ids = torch.full(sep_shape, fill_value=self.sep_token_id, dtype=source_ids.dtype, device=source_ids.device)
        sep_mask = torch.ones_like(sep_ids, dtype=source_attention_mask.dtype)
        
        concat_ids = torch.cat([source_ids, sep_ids, target_ids], dim=-1)
        concat_attention_mask = torch.cat([source_attention_mask, sep_mask, target_attention_mask], dim=-1)
        
        # Debug logging before forward pass
        debug_logger = get_debug_logger()
        debug_logger.log_memory("GEN_CONCAT_DONE", f"Sequences concatenated")
        debug_logger.log_tensor_info("GEN_CONCAT_IDS", "concat_ids", concat_ids)
        debug_logger.log_tensor_info("GEN_LATENT_SRC", "latent_source", latent_source)
        debug_logger.log_tensor_info("GEN_LATENT_TGT", "latent_target", latent_target)
        
        # Forward pass on concatenated input - THIS IS WHERE OOM HAPPENS
        debug_logger.log_memory("GEN_FORWARD_START", "Starting generator forward pass")
        logits = self.model(concat_ids, concat_attention_mask, latent_source, latent_target)
        debug_logger.log_memory("GEN_FORWARD_DONE", "Generator forward pass complete")
        shift_logits = logits[:, :-1, :]
        
        # Prepare labels: only compute loss on the target portion
        shift_labels_pre = concat_ids[..., 1:]
        source_len = source_ids.shape[-1]
        target_len = target_ids.shape[-1]
        
        # Build a mask that selects only the target tokens in the shifted labels
        labels_mask = torch.zeros_like(shift_labels_pre, dtype=target_attention_mask.dtype)
        labels_mask[..., source_len:] = target_attention_mask
        
        # Apply mask using ignore_index=-100
        shift_labels = shift_labels_pre.masked_fill(labels_mask == 0, -100)
        
        # If model flattened set dimension, align labels accordingly
        if shift_labels.dim() == 3 and shift_logits.dim() == 3:
            if shift_labels.shape[0] * shift_labels.shape[1] == shift_logits.shape[0]:
                shift_labels = shift_labels.view(-1, shift_labels.shape[-1])
        
        # Calculate loss
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
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
        
        # Generate samples
        all_samples = []
        src_ids_all = x_source['progen_input_ids']
        src_mask_all = x_source['progen_attention_mask']
        for sample_idx in range(num_samples):
            # Add noise for diversity if generating multiple samples
            if num_samples > 1:
                noise_scale = 0.1
                noisy_latent_source = latent_source + noise_scale * torch.randn_like(latent_source)
                noisy_latent_target = latent_target + noise_scale * torch.randn_like(latent_target)
            else:
                noisy_latent_source = latent_source
                noisy_latent_target = latent_target
            
            # Build prompt for this sample: choose different source sequences if available
            if src_ids_all.dim() == 3:
                set_size = src_ids_all.shape[1]
                set_idx = sample_idx % set_size
                src_ids = src_ids_all[:, set_idx, :]
                src_mask = src_mask_all[:, set_idx, :]
            else:
                src_ids = src_ids_all
                src_mask = src_mask_all
            
            src_lengths = src_mask.sum(dim=-1).to(torch.long)
            max_src_len = int(src_lengths.max().item() if src_lengths.numel() > 0 else 0)
            max_prompt_len = max_src_len + 1  # +1 for separator
            
            # TODO: make sure we really want to fill with pad token here.
            start_ids = torch.full((batch_size, max_prompt_len), fill_value=self.pad_token_id, dtype=src_ids.dtype, device=device)
            start_mask = torch.zeros((batch_size, max_prompt_len), dtype=src_mask.dtype, device=device)
            
            for i in range(batch_size):
                L = int(src_lengths[i].item())
                if L > 0:
                    start_ids[i, :L] = src_ids[i, :L]
                start_ids[i, L] = self.sep_token_id
                start_mask[i, : L + 1] = 1
                
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