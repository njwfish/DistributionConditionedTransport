import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

from utils.hf_local import resolve_local_or_repo
from utils.nan_debug import get_nan_logger

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

    def __init__(self, config: EsmConfig, latent_dim: int = 32, condition_dim: int = 256, condition_method: str = "additive", num_condition_layers: int = 1, use_delta_token: bool = True, gate_method: str = "time_linear"):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.condition_method = condition_method
        self.num_condition_layers = num_condition_layers
        self.use_delta_token = use_delta_token
        self.gate_method = gate_method

        if self.condition_method not in ["additive", "cross_attn"]:
            raise ValueError(f"Unknown conditioning method: {self.condition_method}")

        # Combine source and target latents as in Progen2 (concatenate)
        combined_latent_dim = latent_dim * 2
        self.condition_proj = nn.Sequential(
            nn.Linear(combined_latent_dim, condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, self.hidden_size)
        )

        # Cross-attention conditioning stack (external conditioner). Constructed even if unused for checkpoint compatibility.
        num_heads = getattr(config, "num_attention_heads", 4)
        attn_dropout = getattr(config, "attention_probs_dropout_prob", 0.0)
        hidden_dropout = getattr(config, "hidden_dropout_prob", 0.0)

        self.latent_src_proj = nn.Linear(latent_dim, self.hidden_size)
        self.latent_tgt_proj = nn.Linear(latent_dim, self.hidden_size)
        self.latent_delta_proj = nn.Linear(latent_dim, self.hidden_size)

        # Use a compact FFN sized by condition_dim to keep conditioner lightweight
        ffn_inner_dim = max(condition_dim, 1)
        self.condition_blocks = nn.ModuleList()
        for _ in range(self.num_condition_layers):
            block = nn.ModuleDict({
                "attn": nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=num_heads, dropout=attn_dropout, batch_first=True),
                "ln1": nn.LayerNorm(self.hidden_size),
                "ffn": nn.Sequential(
                    nn.Linear(self.hidden_size, ffn_inner_dim),
                    nn.GELU(),
                    nn.Dropout(hidden_dropout),
                    nn.Linear(ffn_inner_dim, self.hidden_size),
                    nn.Dropout(hidden_dropout),
                ),
                "ln2": nn.LayerNorm(self.hidden_size),
            })
            self.condition_blocks.append(block)

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
        # Check early for non-finites from base ESM
        try:
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                nan_logger = get_nan_logger()
                nan_logger.log("Non-finite hidden_states from ESM base")
                nan_logger.log_tensor("taesmf.hidden_states_pre", hidden_states)
        except Exception:
            pass

        # Time embedding
        time_embed = transformer_timestep_embedding(t, self.hidden_size).to(hidden_states.dtype)[:, None, :]

        hidden_states = hidden_states + time_embed
        try:
            if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                nan_logger = get_nan_logger()
                nan_logger.log("Non-finite after adding time embedding")
                nan_logger.log_tensor("taesmf.time_embed", time_embed)
                nan_logger.log_tensor("taesmf.hidden_plus_time", hidden_states)
        except Exception:
            pass

        # Latent conditioning
        if self.condition_method == "additive":
            combined_latent = torch.cat([latent_source, latent_target], dim=-1)
            cond = self.condition_proj(combined_latent)  # (B, D)
            hidden_states = hidden_states + cond.unsqueeze(1)
            try:
                if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                    nan_logger = get_nan_logger()
                    nan_logger.log("Non-finite after additive conditioning")
                    nan_logger.log_tensor("taesmf.cond_add", cond)
            except Exception:
                pass
        elif self.condition_method == "cross_attn":
            # Build latent memory tokens [src, tgt, (tgt - src)] → (B, M, D)
            src_token = self.latent_src_proj(latent_source)
            tgt_token = self.latent_tgt_proj(latent_target)
            memory_tokens = [src_token, tgt_token]
            if self.use_delta_token:
                delta = latent_target - latent_source
                delta_token = self.latent_delta_proj(delta)
                memory_tokens.append(delta_token)
            memory = torch.stack(memory_tokens, dim=1)

            # Compute gating scalar per-batch
            if self.gate_method == "time_linear":
                alpha = t.view(-1, 1, 1).to(dtype=hidden_states.dtype)
            else:
                alpha = None

            # Run conditioner in FP32 with autocast disabled to avoid bf16 NaNs
            orig_dtype = hidden_states.dtype
            with torch.autocast(device_type='cuda', enabled=False):
                hidden_states = hidden_states.float()
                memory = memory.float()
                if alpha is not None:
                    alpha = alpha.float()

                # Run through external conditioner blocks
                for block in self.condition_blocks:
                    attn_out, _ = block["attn"](query=hidden_states, key=memory, value=memory, need_weights=False)
                    if alpha is not None:
                        attn_out = attn_out * alpha
                    hidden_states = block["ln1"](hidden_states + attn_out)
                    ffn_out = block["ffn"](hidden_states)
                    hidden_states = block["ln2"](hidden_states + ffn_out)
                    try:
                        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
                            nan_logger = get_nan_logger()
                            nan_logger.log("Non-finite inside cross-attn conditioning block")
                            nan_logger.log_tensor("taesmf.attn_out", attn_out)
                            nan_logger.log_tensor("taesmf.ffn_out", ffn_out)
                            nan_logger.log_tensor("taesmf.hidden_post_block", hidden_states)
                    except Exception:
                        pass
            hidden_states = hidden_states.to(orig_dtype)
        else:
            raise ValueError(f"Unknown conditioning method: {self.condition_method}")
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
        condition_method: str = "additive",
        temperature: float = 1.0,
        seq_length: Optional[int] = 1000,
        sample_dt: float = 0.1,
        num_condition_layers: int = 1,
        use_delta_token: bool = True,
        gate_method: str = "time_linear",
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
            num_condition_layers=num_condition_layers,
            use_delta_token=use_delta_token,
            gate_method=gate_method,
        )

        # Precision setup removed; model runs in default dtype

        # TODO: I think this is freezing all parameters including the language model head, right? Or wrong? If right, change.
        # TODO: in any case it would be good to have the option to only un-freeze the language model head.
        # TODO: there might be some conflicts when only un-freezing the language model head, figure out if this is the case.
        # Freezing
        if freeze_esm2:
            # Freeze only the ESM backbone; keep LM head and conditioning modules trainable
            for param in self.model.esm.parameters():
                param.requires_grad = False
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            # Conditioning modules remain trainable
            modules_to_keep = [
                getattr(self.model, "condition_proj", None),
                getattr(self.model, "latent_src_proj", None),
                getattr(self.model, "latent_tgt_proj", None),
                getattr(self.model, "latent_delta_proj", None),
                getattr(self.model, "condition_blocks", None),
            ]
            for mod in modules_to_keep:
                if mod is None:
                    continue
                for param in mod.parameters():
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
    # TODO: I scanned the dataset and since there is no padding, we should generally be fine. But I think this code is a little shaky when it comes to handling padding.
    # TODO: make sure the attention mask of the dataset is actually correct (I think it should basically be all 1s).
    def loss(self, x_source, x_target, latent_source: torch.Tensor, latent_target: torch.Tensor) -> torch.Tensor:

        # Validate latent dimensions and normalize per-sample
        if latent_source.dim() != 2 or latent_target.dim() != 2:
            raise ValueError(
                f"ESM2_DFM_Generator.loss expects 2D latents shaped (batch_size, latent_dim)."
                f" Got latent_source.shape={tuple(latent_source.shape)},"
                f" latent_target.shape={tuple(latent_target.shape)}"
            )
        latent_source = latent_source / torch.norm(latent_source, dim=-1, keepdim=True).clamp_min(1e-12)
        latent_target = latent_target / torch.norm(latent_target, dim=-1, keepdim=True).clamp_min(1e-12)

        input_ids_source = x_source["esm_input_ids"]  
        attention_mask_source = x_source["esm_attention_mask"] 
        input_ids_target = x_target["esm_input_ids"]  
        attention_mask_target = x_target["esm_attention_mask"] 

        # TODO: why does the ESM-DFM model do all this reshaping work, while the Progen2 code doesn't?
        if input_ids_source.ndim == 3:
            B, S, L = input_ids_source.shape
            input_ids_source = input_ids_source.view(B * S, L)
            attention_mask_source = attention_mask_source.view(B * S, L)
            input_ids_target = input_ids_target.view(B * S, L)
            attention_mask_target = attention_mask_target.view(B * S, L)
            # TODO: make sure reshaping is correct here. You should just print out the shapes of the inputs this function receives at some point.
            latent_source = latent_source.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)
            latent_target = latent_target.unsqueeze(1).repeat(1, S, 1).view(B * S, -1)

        device = input_ids_target.device
        B, L = input_ids_target.shape

        # Sample times and build x_t by mixing source/target across all valid positions
        # Following the discrete flow-matching interpolation: x_t = where(Bernoulli(t), x_1, x_0)
        t = torch.rand((B,), device=device)

        # Valid (non-padded) positions
        if attention_mask_source is not None and attention_mask_target is not None:
            valid_mask = (attention_mask_source > 0) & (attention_mask_target > 0)
        elif attention_mask_target is not None:
            valid_mask = (attention_mask_target > 0)
        elif attention_mask_source is not None:
            valid_mask = (attention_mask_source > 0)
        else:
            valid_mask = torch.ones((B, L), dtype=torch.bool, device=device)

        bern = torch.rand((B, L), device=device) < t[:, None]
        xt = torch.where(valid_mask, torch.where(bern, input_ids_target, input_ids_source), input_ids_target)

        # Enforce BOS/EOS tokens at boundaries
        if self.bos_id is not None:
            xt[:, 0] = self.bos_id
        if self.eos_id is not None:
            xt[:, -1] = self.eos_id

        # Removed latent dtype alignment to model parameters

        # Forward through conditioned ESM
        attn_mask_xt = valid_mask.to(dtype=torch.long)
        logits = self.model(input_ids=xt,
            attention_mask=attn_mask_xt,
            t=t,
            latent_source=latent_source,
            latent_target=latent_target)
        # Early NaN check on logits to pinpoint source
        try:
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                nan_logger = get_nan_logger()
                nan_logger.log("Non-finite logits from ESM forward")
                nan_logger.log_tensor("gen.logits", logits)
        except Exception:
            pass

        # Cross-entropy to target tokens x1; ignore padded target positions
        labels = input_ids_target.clone()
        if attention_mask_target is not None:
            labels[attention_mask_target == 0] = -100
            
        # Ignore ambiguous 'X' residues in target
        labels[labels == self.x_id] = -100
        
        # TODO: why are we transposing here?
        loss = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction='mean')
        # If non-finite, dump detailed context
        try:
            if not torch.isfinite(loss):
                nan_logger = get_nan_logger()
                nan_logger.log("Non-finite generator loss detected")
                nan_logger.log_tensor("gen.logits_sample", logits[..., : min(logits.size(-1), 64)])
                nan_logger.log_tensor("gen.labels", labels)
                nan_logger.log_tensor("gen.xt", xt)
                nan_logger.log_tensor("gen.valid_mask", valid_mask)
                nan_logger.log_tensor("gen.latent_source", latent_source)
                nan_logger.log_tensor("gen.latent_target", latent_target)
        except Exception:
            pass
        return loss

    # TODO: force the last token to always be the EOS token.    
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
            # TODO: make sure both cases are correct.
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
                xt[:, -1] = self.eos_id
                # TODO: why are we setting this here explicitly?
                attention_mask[:, -1] = 1

            # Enforce expected sequence length
            expected_len = self.seq_length
            if xt.size(1) != expected_len:
                raise ValueError(f"Source sequence length {xt.size(1)} does not match expected seq_length {expected_len}")

            # Discrete flow stepping from t=0 to 1 using mixture update
            t = 0.0
            # TODO: you might want to implement a check at the end of the loop to ensure that there are no t = 1 effects (although I think that is just specific for mask-based DFM).
            while t < 1.0 - 1e-6:
                logits = self.model(
                    input_ids=xt,
                    attention_mask=attention_mask,
                    t=torch.full((B,), t, device=device),
                    latent_source=latent_source,
                    latent_target=latent_target,
                )
                
                # TODO: should we just remove temperature scaling here?
                # Temperature scaling
                logits = logits / max(self.temperature, 1e-8)

                # Constrain vocabulary to AA tokens at inner positions, BOS/EOS at ends by masking logits
                mask_logits = torch.full_like(logits, -float('inf'))
                mask_logits[:, 1:-1, self.aa_ids] = 0.0
                if self.bos_id is not None:
                    mask_logits[:, 0, :] = -float('inf')
                    mask_logits[:, 0, self.bos_id] = 0.0
                if self.eos_id is not None:
                    mask_logits[:, -1, :] = -float('inf')
                    mask_logits[:, -1, self.eos_id] = 0.0

                p1 = F.softmax(logits + mask_logits, dim=-1)  # (B, L, V)

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

                # Enforce BOS/EOS tokens at boundaries after sampling
                if self.bos_id is not None:
                    xt[:, 0] = self.bos_id
                if self.eos_id is not None:
                    xt[:, -1] = self.eos_id

                t += h

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


