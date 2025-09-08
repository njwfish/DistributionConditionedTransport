import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from utils.hf_local import resolve_local_or_repo
from utils.nan_debug import get_nan_logger

import math

class ESM2_Baseline_Generator(nn.Module):
    """Pretrained baseline generator based on ESM2"""

    def __init__(
        self,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        freeze_esm2: bool = False,
        temperature: float = 1.0,
        seq_length: Optional[int] = 1000,
    ):
        super().__init__()

        # Tokenizer and config/model
        resolved_name = resolve_local_or_repo(model_name)
        self.tokenizer = EsmTokenizer.from_pretrained(resolved_name)
        self.model = EsmForMaskedLM.from_pretrained(resolved_name)
        self.config = self.model.config

        # Freeze all parameters: using pretrained ESM2 as fixed baseline
        for param in self.model.parameters():
            param.requires_grad = False

        self.bos_id = self.tokenizer.cls_token_id
        self.eos_id = self.tokenizer.eos_token_id

        aa_tokens = [
            "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
            "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
        ]
        self.aa_ids = [self.tokenizer.convert_tokens_to_ids(a) for a in aa_tokens]

        self.temperature = temperature
        self.seq_length = seq_length


    # TODO: force the last token to always be the EOS token.    
    @torch.no_grad()
    def sample(self, latent_target: torch.Tensor, num_samples: int = 1, return_texts: bool = False):
        """Baseline sampling that feeds external latent_target directly into lm_head.

        - Accepts latent_target of shape (B, D) or (B, L, D) with D == hidden_size.
        - Returns [batch_size, num_samples, seq_len] token ids for ESM vocabulary.
        """
        self.model.eval()
        device = latent_target.device

        # Compute logits directly from provided hidden states
        logits = self.model.lm_head(latent_target.to(device)) 

        # Temperature scaling
        logits = logits / max(self.temperature, 1e-8)

        # Constrain vocabulary: AA tokens for inner positions, BOS/EOS at ends
        mask_logits = torch.full_like(logits, -float('inf'))
        
        mask_logits[:, 1:-1, self.aa_ids] = 0.0
        if self.bos_id is not None:
            mask_logits[:, 0, :] = -float('inf')
            mask_logits[:, 0, self.bos_id] = 0.0
        if self.eos_id is not None:
            mask_logits[:, -1, :] = -float('inf')
            mask_logits[:, -1, self.eos_id] = 0.0

        probs = F.softmax(logits + mask_logits, dim=-1)

        results = []
        texts_all = []
        for _ in range(num_samples):
            xt = torch.distributions.Categorical(probs=probs).sample()  # (B, L)
            # Enforce BOS/EOS tokens at boundaries after sampling (redundant but safe)
            if self.bos_id is not None:
                xt[:, 0] = self.bos_id
            if self.eos_id is not None:
                xt[:, -1] = self.eos_id

            results.append(xt)

            if return_texts:
                batch_texts = []
                for b in range(B):
                    ids_b = xt[b]
                    text = self.tokenizer.decode(ids_b.tolist(), skip_special_tokens=True).replace(" ", "")
                    if not text:
                        text = ""
                    batch_texts.append(text)
                texts_all.append(batch_texts)

        # Stack to [B, num_samples, L]
        samples = torch.stack(results, dim=1)
        if return_texts:
            # current texts_all is [num_samples][B], transpose to [B][num_samples]
            texts_per_batch = list(map(list, zip(*texts_all))) if len(texts_all) > 0 else [[] for _ in range(B)]
            return samples, texts_per_batch
        return samples

    def compute_pseudo_log_likelihood(self, sequences, aggregate: str = "mean"):
        """Compute pseudo-log-likelihood of protein sequences.
        
        For each position in the sequence (excluding BOS/EOS), mask it and compute
        the log probability of the actual amino acid at that position using the model.
        
        Args:
            sequences: Either:
                - List of protein sequence strings (e.g., ["MKTV", "ACDE"])
                - Tensor of token IDs with shape [batch_size, seq_len]
            aggregate: How to aggregate across positions ("mean", "sum", or "none")
                - "mean": Return mean log-likelihood per position
                - "sum": Return sum of log-likelihoods
                - "none": Return log-likelihood for each position [batch_size, seq_len-2]
        
        Returns:
            torch.Tensor: Pseudo-log-likelihood values
                - If aggregate="mean" or "sum": [batch_size]
                - If aggregate="none": [batch_size, seq_len-2]
        """
        self.model.eval()
        
        # Convert sequences to token IDs if needed
        if isinstance(sequences, list) and isinstance(sequences[0], str):
            # Add BOS and EOS tokens to sequences
            sequences_with_special = [f"<cls>{seq}<eos>" for seq in sequences]
            tokenized = self.tokenizer(sequences_with_special, 
                                     return_tensors="pt", 
                                     padding=True,
                                     add_special_tokens=False)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized.get("attention_mask", None)
        else:
            input_ids = sequences
            attention_mask = None
            
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        batch_size, seq_len = input_ids.shape
        
        # Store log probabilities for each position (excluding BOS and EOS)
        log_probs = []
        
        with torch.no_grad():
            # For each position between BOS and EOS
            for pos in range(1, seq_len - 1):
                # Create masked input
                masked_input = input_ids.clone()
                original_tokens = masked_input[:, pos].clone()
                masked_input[:, pos] = self.tokenizer.mask_token_id
                
                # Get model predictions
                if attention_mask is not None:
                    outputs = self.model(input_ids=masked_input, attention_mask=attention_mask)
                else:
                    outputs = self.model(input_ids=masked_input)
                    
                logits = outputs.logits
                
                # Get log probabilities at the masked position
                position_logits = logits[:, pos, :]  # [batch_size, vocab_size]
                position_log_probs = F.log_softmax(position_logits, dim=-1)
                
                # Extract log probability of the original token
                token_log_probs = position_log_probs.gather(1, original_tokens.unsqueeze(1)).squeeze(1)
                log_probs.append(token_log_probs)
        
        # Stack log probabilities: [batch_size, num_positions]
        log_probs = torch.stack(log_probs, dim=1)
        
        # Apply aggregation
        if aggregate == "mean":
            return log_probs.mean(dim=1)
        elif aggregate == "sum":
            return log_probs.sum(dim=1)
        elif aggregate == "none":
            return log_probs
        else:
            raise ValueError(f"Invalid aggregate option: {aggregate}. Choose from 'mean', 'sum', 'none'.")


