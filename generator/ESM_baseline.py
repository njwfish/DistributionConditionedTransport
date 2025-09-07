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


