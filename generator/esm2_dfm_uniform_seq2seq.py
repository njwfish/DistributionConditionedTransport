import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from utils.hf_local import resolve_local_or_repo

import math

# TODO: cite source (the original DFM paper).
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

    def __init__(self, config: EsmConfig, latent_dim: int = 32, condition_dim: int = 256, condition_method: str = "additive", scale_time: bool = False):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.condition_method = condition_method
        self.scale_time = scale_time

        if self.condition_method not in ["additive", "prefix", "no_use"]:
            raise ValueError(f"Unknown conditioning method: {self.condition_method}")

        # Combine source and target latents as in Progen2 (concatenate)
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

        # Time embedding
        if self.scale_time:
            time_embed = transformer_timestep_embedding(t, self.hidden_size)[:, None, :]
        else:
            time_embed = transformer_timestep_embedding(t * 1000, self.hidden_size)[:, None, :]
        hidden_states = hidden_states + time_embed

        # TODO: currently the conditioning here is only done in the additive mode. I have code for all cases, just need to add it back in.
        # Latent conditioning
        combined_latent = torch.cat([latent_source, latent_target], dim=-1)
        cond = self.condition_proj(combined_latent).to(hidden_states.dtype)  # (B, D)
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
        freeze_esm2: bool = True,
        condition_method: str = "additive",
        scale_time: bool = False,
        temperature: float = 1.0,
        max_length: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
        precision: Optional[str] = None,
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
            condition_method=condition_method,
            scale_time=scale_time,
        )

        # Precision setup (optional)
        if precision is not None:
            p = precision.lower()
            if p in ("fp16", "half"):
                self.model = self.model.to(dtype=torch.float16)
            elif p in ("bf16", "bfloat16"):
                self.model = self.model.to(dtype=torch.bfloat16)
            elif p in ("fp32", "float32"):
                self.model = self.model.to(dtype=torch.float32)

        # TODO: I think this is freezing all parameters including the language model head, right? Or wrong? If right, change.
        # TODO: in any case it would be good to have the option to only un-freeze the language model head.
        # Freezing
        if freeze_esm2:
            for param in self.model.parameters():
                param.requires_grad = False
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
        self.max_length = max_length

    def loss(self, x_source, x_target, latent_source: torch.Tensor, latent_target: torch.Tensor) -> torch.Tensor:
        """Uniform corruption flow-matching loss that mutates source toward target.

        We start from the target labels x1 = target tokens. For each sample, pick t ~ U(0,1),
        copy the source tokens as xt, then corrupt a fraction (1 - t) of positions with
        uniformly random amino acids (not mask tokens). Predict x1 from (xt, t, latents).

        Inputs mirror Progen2Generator.loss: x_source and x_target provide tokenizer-specific
        fields for ESM (esm_input_ids, esm_attention_mask).
        """
        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"ESM2_DFM_Generator.loss expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        # Validate input shapes (allow [B, L] or [B, S, L])
        if x_source["esm_input_ids"].dim() not in (2, 3) or x_source["esm_attention_mask"].dim() not in (2, 3):
            raise ValueError(
                f"ESM2_DFM_Generator.loss expects source tensors with 2 or 3 dims."
                f" Got input_ids.dim={x_source['esm_input_ids'].dim()}, attention_mask.dim={x_source['esm_attention_mask'].dim()}"
            )
        if x_source["esm_input_ids"].dim() == 2 and x_source["esm_input_ids"].size(0) != latent_source.size(0):
            raise ValueError(
                f"ESM2_DFM_Generator.loss received 2D source input_ids with batch={x_source['esm_input_ids'].size(0)} but latents batch={latent_source.size(0)}."
                f" This suggests inputs were flattened across set dimension. Keep inputs un-flattened with shape [B, S, L]."
            )
        if x_target["esm_input_ids"].dim() not in (2, 3) or x_target["esm_attention_mask"].dim() not in (2, 3):
            raise ValueError(
                f"ESM2_DFM_Generator.loss expects target tensors with 2 or 3 dims."
                f" Got input_ids.dim={x_target['esm_input_ids'].dim()}, attention_mask.dim={x_target['esm_attention_mask'].dim()}"
            )
        if x_target["esm_input_ids"].dim() == 2 and x_target["esm_input_ids"].size(0) != latent_target.size(0):
            raise ValueError(
                f"ESM2_DFM_Generator.loss received 2D target input_ids with batch={x_target['esm_input_ids'].size(0)} but latents batch={latent_target.size(0)}."
                f" This suggests inputs were flattened across set dimension. Keep inputs un-flattened with shape [B, S, L]."
            )
        input_ids_source = x_source["esm_input_ids"]  # (B, L)
        attention_mask_source = x_source["esm_attention_mask"]  # (B, L)
        input_ids_target = x_target["esm_input_ids"]  # (B, L)
        attention_mask_target = x_target["esm_attention_mask"]  # (B, L)

        # Keep un-flattened shapes; model handles batching, and set microbatching is upstream
        if input_ids_source.ndim == 3:
            B, S, L = input_ids_source.shape
            input_ids_source = input_ids_source.view(B * S, L)
            attention_mask_source = attention_mask_source.view(B * S, L)
            input_ids_target = input_ids_target.view(B * S, L)
            attention_mask_target = attention_mask_target.view(B * S, L)
            # TODO: it was suggested to use one of the two, I suppose one is wrong and the other is right. Figure out which is which (referring to commented lines).
            latent_source = latent_source.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)
            latent_target = latent_target.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)
            #latent_source = latent_source.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
            #latent_target = latent_target.unsqueeze(1).expand(B, S, -1).reshape(B * S, -1)
        device = input_ids_target.device
        B, L = input_ids_target.shape

        # TODO: it's kind of strange to have the time-conditioning here if we really have no noise at all. I mean, at that point the conditioning on t is just meaningless.
        # Sample times and build corrupted xt by uniform amino acid noise on source sequence
        t = torch.rand((B,), device=device)
        xt = input_ids_source.clone()

        # Build uniform noise across AA tokens only
        aa_ids_tensor = torch.tensor(self.aa_ids, device=device)
        uniform_idx = torch.randint(0, aa_ids_tensor.numel(), (B, L), device=device)
        uniform_noise = aa_ids_tensor[uniform_idx]

        corrupt_mask = (torch.rand((B, L), device=device) < (1 - t[:, None]))
        # Keep BOS/EOS fixed
        if self.bos_id is not None:
            corrupt_mask[:, 0] = False
        if self.eos_id is not None:
            corrupt_mask[:, -1] = False
        # Respect attention mask of source sequence
        if attention_mask_source is not None:
            corrupt_mask = corrupt_mask & (attention_mask_source > 0)
        xt[corrupt_mask] = uniform_noise[corrupt_mask]

        # Forward through conditioned ESM
        logits = self.model(
            input_ids=xt,
            attention_mask=attention_mask_source,
            t=t,
            latent_source=latent_source,
            latent_target=latent_target,
        )  # (B, L, V)

        # Cross-entropy to target tokens x1; ignore padded target positions
        labels = input_ids_target.clone()
        if attention_mask_target is not None:
            labels[attention_mask_target == 0] = -100
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='mean')
        return loss

    @torch.no_grad()
    def sample(self, x_source, latent_source: torch.Tensor, latent_target: torch.Tensor, num_samples: int = 1, return_texts: bool = False):
        """Mutational sampling starting from source sequences with discrete flow steps.

        Mirrors Progen2Generator.sample signature and return shape:
        - Returns [batch_size, num_samples, seq_len] token ids for ESM vocabulary.
        - Optionally returns decoded strings.
        """
        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"ESM2_DFM_Generator.sample expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        # Validate source shapes (allow [B, L] or [B, S, L])
        if x_source["esm_input_ids"].dim() not in (2, 3) or x_source["esm_attention_mask"].dim() not in (2, 3):
            raise ValueError(
                f"ESM2_DFM_Generator.sample expects source tensors with 2 or 3 dims."
                f" Got input_ids.dim={x_source['esm_input_ids'].dim()}, attention_mask.dim={x_source['esm_attention_mask'].dim()}"
            )
        if x_source["esm_input_ids"].dim() == 2 and x_source["esm_input_ids"].size(0) != latent_source.size(0):
            raise ValueError(
                f"ESM2_DFM_Generator.sample received 2D source input_ids with batch={x_source['esm_input_ids'].size(0)} but latents batch={latent_source.size(0)}."
                f" This suggests inputs were flattened across set dimension. Keep inputs un-flattened with shape [B, S, L]."
            )

        self.model.eval()
        device = latent_source.device

        src_ids_all = x_source["esm_input_ids"]
        src_mask_all = x_source["esm_attention_mask"]

        # Flatten set dim if present for starting point selection per sample
        if src_ids_all.ndim == 3:
            B, S, L = src_ids_all.shape
        else:
            B, L = src_ids_all.shape
            S = 1

        def select_source_for_sample(sample_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            if src_ids_all.ndim == 3:
                set_idx = sample_idx % S
                return src_ids_all[:, set_idx, :].to(device), src_mask_all[:, set_idx, :].to(device)
            return src_ids_all.to(device), src_mask_all.to(device)

        results = []
        texts_all = []

        for s in range(num_samples):
            xt, attention_mask = select_source_for_sample(s)
            # Enforce BOS/EOS presence if tokenizer uses them
            if self.bos_id is not None:
                xt[:, 0] = self.bos_id
            if self.eos_id is not None:
                attention_mask[:, -1] = 1

            # Discrete flow stepping from t=0 to 1
            t = 0.0
            dt = 0.01
            # TODO: you might want to implement a check at the end of the loop to ensure that there are no t = 1 effects (although I think that is just specific for mask-based DFM).
            while t < 1.0:
                logits = self.model(
                    input_ids=xt,
                    attention_mask=attention_mask,
                    t=torch.full((B,), t, device=device),
                    latent_source=latent_source,
                    latent_target=latent_target,
                )
                # Temperature scaling
                logits = logits / max(self.temperature, 1e-8)
                x1_probs = F.softmax(logits, dim=-1)  # (B, L, V)

                # Constrain vocabulary to AA tokens at inner positions, BOS/EOS at ends
                mask = torch.full_like(x1_probs, -float('inf'))
                mask[:, 1:-1, self.aa_ids] = 0.0
                if self.bos_id is not None:
                    mask[:, 0, self.bos_id] = 0.0
                if self.eos_id is not None:
                    mask[:, -1, self.eos_id] = 0.0
                logits_masked = torch.log(x1_probs + 1e-9) + mask
                x1_probs = F.softmax(logits_masked, dim=-1)

                # Step probabilities from uniform corruption scheme
                step_probs = ((dt / max(1.0 - t, 1e-6)) * x1_probs).clamp(max=1.0)

                # Zero out diagonal, then set diagonal to keep-prob so rows sum to 1
                step_probs = step_probs.clone()
                step_probs.scatter_(-1, xt[:, :, None], 0.0)
                stay_prob = (1.0 - step_probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
                step_probs.scatter_(-1, xt[:, :, None], stay_prob)

                # Sample next xt
                xt = torch.distributions.Categorical(step_probs).sample()

                t += dt

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
            # Convert to list-of-list per batch
            # current texts_all is [num_samples][B], transpose to [B][num_samples]
            texts_per_batch = list(map(list, zip(*texts_all))) if len(texts_all) > 0 else [[] for _ in range(B)]
            return samples, texts_per_batch
        return samples


