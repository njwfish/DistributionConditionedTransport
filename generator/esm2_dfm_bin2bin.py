import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch.nn as nn

import logging
import wandb
import os
from transformers import EsmForMaskedLM, EsmTokenizer, Trainer, TrainingArguments
from transformers import EsmConfig

import numpy as np
import random

import math

from utils.process_seqs import hamming_distance

# NOTE: taken without any modification from from https://github.com/andrew-cr/discrete_flow_models/blob/800395d172be6b950d2ab87bcf154d752bd2cf76/flow_model.py#L136 
# From https://github.com/yang-song/score_sde_pytorch/ which is from
#  https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
def transformer_timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    # assumes timesteps is in the range 0 to 1000
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # magic number 10000 is from transformers
    emb = math.log(max_positions) / (half_dim - 1)
    
    # emb = math.log(2.) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    
    # emb = tf.range(num_embeddings, dtype=jnp.float32)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=jnp.float32)[:, None] * emb[None, :]
    emb = timesteps.float()[:, None] * emb[None, :]
    
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    # TODO: was this really already there in the original version? Make sure to check.
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb
    

# TODO: need to add a couple of arguments to class call (and make sure to include them when constructing the class, I don't want some mismatch with defaults)
class TimeAwareEsm(EsmForMaskedLM):
    def __init__(self, config, latent_dim=32, condition_dim=256, condition_method="additive",scale_time=False):
        super().__init__(config)
        self.D = config.hidden_size

        self.condition_method = condition_method
        self.scale_time = scale_time
        # TODO: need to implement self.hidden_dim... or do I?
        # TODO: remove this "no_use" clause later, it's just for debugging and seeing whether the distribution embeddings actually learn something.
        if self.condition_method in ["additive", "prefix", "no_use"]:
            # For additive conditioning, project to hidden states
            self.condition_proj = nn.Sequential(
                nn.Linear(latent_dim, condition_dim),
                nn.GELU(),
                nn.Linear(condition_dim, self.D)
            )
        else:
            raise ValueError(f"Unknown conditioning method: {condition_method}")


    def forward(self,
                input_ids=None,
                attention_mask=None,
                t=None,                  # <-- new arg
                labels=None,
                condition=None):
        """
        t: Tensor, shape (B,) in [0,1], your noise level per sample.
        """
        if self.condition_method == "additive":
            # 1) run the usual ESM forward to get embeddings
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state  # (B, L, D)

            # 2) compute time embeddings and add them in
            if t is not None:
                # time_emb: (B, D) -> expand to (B, 1, D) and broadcast
                # convert to [0,1000] + sinusoidal embed
                # TODO: figuer out whether there was a good reason for not doing the stretching of time by a factor of 1000 or not and which one is better here for you?
                if self.scale_time: 
                    te = transformer_timestep_embedding(
                            (t).to(hidden_states.device),
                            embedding_dim=hidden_states.size(-1)
                        )[:, None, :]            
                else: 
                    te = transformer_timestep_embedding(
                            (t * 1000).to(hidden_states.device),
                            embedding_dim=hidden_states.size(-1)
                        )[:, None, :]
                hidden_states = hidden_states + te
            
            #hidden_states = hidden_states + condition
            condition_expanded = condition.unsqueeze(1)
            hidden_states = hidden_states + condition_expanded
            
            # 3) pass through LM head
            logits = self.lm_head(hidden_states)  # (B, L, vocab)
            

        elif self.condition_method == "no_use":
            # 1) run the usual ESM forward to get embeddings
            outputs = self.esm(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            hidden_states = outputs.last_hidden_state  # (B, L, D)

            # 2) compute time embeddings and add them in
            if t is not None:
                # time_emb: (B, D) -> expand to (B, 1, D) and broadcast
                # convert to [0,1000] + sinusoidal embed
                # TODO: figuer out whether there was a good reason for not doing the stretching of time by a factor of 1000 or not and which one is better here for you?
                if self.scale_time: 
                    te = transformer_timestep_embedding(
                            (t).to(hidden_states.device),
                            embedding_dim=hidden_states.size(-1)
                        )[:, None, :]            
                else: 
                    te = transformer_timestep_embedding(
                            (t * 1000).to(hidden_states.device),
                            embedding_dim=hidden_states.size(-1)
                        )[:, None, :]
                hidden_states = hidden_states + te

            # 3) pass through LM head
            logits = self.lm_head(hidden_states)  # (B, L, vocab)
        elif self.condition_method == "prefix":
            # 1) get the token embeddings for the real sequence (includes CLS at pos 0)
            # TODO: make sure this is settled: so are you doing the correct thing here or are the arguments different? Maybe better to give arguments as keyword arguments?
            token_embeds = self.esm.embeddings(input_ids, attention_mask)

            # TODO: you are treating the CLS token in a special manner but I think you are doing this asymmetrically to the EOS token.
            # 2) split off the CLS embedding and the remainder
            cls_embed       = token_embeds[:, :1, :]               # [B, 1, D]
            rest_token_embs = token_embeds[:, 1:, :]               # [B, L-1, D]

            # 3) build your prefix embed
            prefix_embeds   = condition.unsqueeze(1)               # [B, 1, D]

            # 4) concatenate in the order [CLS] [PREFIX] [TOKENS…]
            combined_embeds = torch.cat([cls_embed, prefix_embeds, rest_token_embs], dim=1)  # [B, (1+1+(L-1)), D] == [B, L+1, D]

            # 5) extend the attention mask the same way
            cls_mask     = attention_mask[:, :1]                   # [B, 1]
            rest_mask    = attention_mask[:, 1:]                   # [B, L-1]
            prefix_mask  = torch.ones_like(cls_mask)               # [B, 1]
            extended_mask = torch.cat([cls_mask, prefix_mask, rest_mask], dim=1)  
                                                                # [B, L+1]
            # The encoder expects a 3D attention mask. EsmModel provides a helper for this.
            extended_attention_mask = self.esm.get_extended_attention_mask(
                extended_mask, combined_embeds.shape[:2], combined_embeds.device
            )
            # TODO: make sure this is settled: so are you doing the correct thing here or are the arguments different? Maybe better to give arguments as keyword arguments?
            encoder_outputs = self.esm.encoder(
                combined_embeds,
                attention_mask=extended_attention_mask,
            )
            
            # TODO: I wouldn't be so sure that the index 0 is correct here
            hidden_states = encoder_outputs[0] # This is the equivalent of outputs.last_hidden_state
            # 7) (optional) add your time embedding everywhere
            if t is not None:
                te = transformer_timestep_embedding(
                        (t * 1000).to(hidden_states.device),
                        embedding_dim=hidden_states.size(-1)
                    )[:, None, :]                                 # [B, 1, D]
                hidden_states = hidden_states + te               # broadcast to [B, L+1, D]

            # 8) drop BOTH CLS (pos 0) and prefix (pos 1) before LM head
            # hidden_states: [B, L+1, D]  (with CLS at 0, prefix at 1, then tokens 2…L)
            first_token = hidden_states[:, :1, :]    # [B, 1, D], keeps token 0
            rest_tokens = hidden_states[:, 2:, :]     # [B, L-1, D], drops tokens 0–1, so actually keeps 2…
            hs_tokens = torch.cat([first_token, rest_tokens], dim=1)  # [B, L, D]

            # TODO: check whether you should adjust the positional embeddings since you are effectively shifting every sequence element by one position when including the prefix token I think. But no idea right now whether that even is a problem.

            # TODO: make sure the lm head has no interactions between any of the tokens, otherwise the cutting out of the prefix token might lead to weird behavior.
            # 9) predict exactly the original L-1 "real" tokens
            logits = self.lm_head(hs_tokens)                     # [B, L-1, vocab]


        else:
            raise ValueError(f"Unknown conditioning method: {self.condition_method}")

        return {"logits": logits}



class ESM2_DFM_Generator:
    """Generator class using conditioned Progen2 for the distribution embeddings framework."""
    
    # TODO: there are some additional keyword arguments given by esm2_dfm.yaml, make sure to either delete them from the .yaml file or include them in the function signature.
    def __init__(
        self,
        model_name="facebook/esm2_t6_8M_UR50D",
        latent_dim=32,
        condition_dim=256,
        freeze_esm2=True,
        condition_method="additive",
        scale_time=False,
        max_length=None,
        reject_sample=False,
        seq2seq_mode=False
    ):  
        self.seed = 0
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.reject_sample=reject_sample
        self.max_length = max_length
        self.seq2seq_mode = seq2seq_mode
        self.config = EsmConfig.from_pretrained(model_name)
        self.model = TimeAwareEsm.from_pretrained(model_name, config=self.config,latent_dim=latent_dim,condition_dim=condition_dim,condition_method=condition_method,scale_time=scale_time) 
        self.condition_method = condition_method
        if freeze_esm2:
            print("***FREEZING***")
            ## TODO: make sure that you are exactly unfreezing the parts that you need and exactly freezing the parts that you want to freeze.
            #for name, param in self.model.esm.encoder.named_parameters():
            #    param.requires_grad = False

            ## TODO: make sure that you are exactly unfreezing the parts that you need and exactly freezing the parts that you want to freeze.
            ## TODO: make sure to understand exactly why you would freeze the LayerNorms and whether this is typically done. Like, what is the reasonign here?
            ## freeze LayerNorms in the encoder too (optional apparently)
            #for name, param in self.model.esm.named_parameters():
            #    if "layer_norm" in name.lower():
            #        param.requires_grad = False
            # totally freeze the entire ESM2 model (backbone + LM‐head + any other submodule)
            for param in self.model.parameters():
                param.requires_grad = False

            # but now un‐freeze just your conditioning MLP
            for param in self.model.condition_proj.parameters():
                param.requires_grad = True

        #self.model_parameters = list(self.model.lm_head.parameters())
        
        # Initialize tokenizer (for generation)
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)

        self.mask_token = self.tokenizer.mask_token
        self.mask_token_id = self.tokenizer.mask_token_id

        # begin / end‐of‐sequence
        self.bos   = self.tokenizer.cls_token
        self.bos_id = self.tokenizer.cls_token_id
        self.eos  = self.tokenizer.eos_token
        self.eos_id = self.tokenizer.eos_token_id

        vocab_dict = self.tokenizer.get_vocab()

        # sort tokens by their id so you get them in "alphabet" order
        full_alphabet = [tok for tok, idx in sorted(vocab_dict.items(), key=lambda x: x[1])]

        print(full_alphabet)
        # the 20 canonical amino acids
        aa_tokens = ["A","C","D","E","F","G","H","I","K","L",
                    "M","N","P","Q","R","S","T","V","W","Y"]
        
        self.aa_ids = [self.tokenizer.convert_tokens_to_ids(aa) for aa in aa_tokens]
        
        print(f"AA_range: {self.aa_ids} and {self.bos_id} and {self.eos_id} and {self.bos} and {self.eos}")
    

    def loss(self, x, latent):
        if self.seq2seq_mode:
            return self._loss_seq2seq(x, latent)
        else:
            return self._loss_original(x, latent)

    def _loss_seq2seq(self, x, latent):
        """New seq2seq approach: mask positions where sequences differ"""
        # Prepare inputs and labels
        # x now contains both samples_high and samples_low
        input_ids_high = x["esm_input_ids"]  # samples_high sequences
        input_ids_low = x["esm_input_ids_low"]  # samples_low sequences
        attention_mask = x["esm_attention_mask"]
        
        # Determine conditioning
        if self.condition_method == "no_use":
            condition = None
        elif self.condition_method in ("additive", "prefix"):
            condition = self.model.condition_proj(latent)
        else:
            raise ValueError(f"Condition Method {self.condition_method} not implemented")

        # Handle batch dimensions - expand input_ids_low to match input_ids_high
        if input_ids_high.ndim == 3:
            B1, B2, L = input_ids_high.shape
            
            # input_ids_low should have shape (B1, 1, L) - expand it to (B1, B2, L)
            if input_ids_low.shape == (B1, 1, L):
                # TODO: make sure the expanding is behaving correctly.
                # Expand the single sequence to match all B2 sequences
                input_ids_low = input_ids_low.expand(B1, B2, L)
            elif input_ids_low.shape != (B1, B2, L):
                raise ValueError(f"input_ids_low shape {input_ids_low.shape} cannot be expanded to match input_ids_high shape {(B1, B2, L)}")
            
            # Reshape both to 2D
            input_ids_high = input_ids_high.view(B1 * B2, L)
            input_ids_low = input_ids_low.view(B1 * B2, L)
            attention_mask = attention_mask.view(B1 * B2, L)
            
            if condition is not None:
                condition = (
                    condition.unsqueeze(1)
                            .repeat(1, B2, 1)
                            .view(B1 * B2, -1)
                )
        elif input_ids_high.ndim == 2:
            # For 2D case, both should have matching shapes
            if input_ids_low.shape != input_ids_high.shape:
                raise ValueError(f"input_ids_low shape {input_ids_low.shape} must match input_ids_high shape {input_ids_high.shape}")

        # Create masked input by masking positions where sequences differ
        B, D = input_ids_high.shape
        xt = input_ids_low.clone()  # Start with samples_low sequence
        
        # Mask positions where sequences differ (excluding special tokens)
        diff_mask = (input_ids_high != input_ids_low)
        diff_mask[:, 0] = False  # Don't mask BOS token
        diff_mask[:, -1] = False  # Don't mask EOS token
        
        xt[diff_mask] = self.mask_token_id
        
        # Check if any tokens are masked
        total_masked = diff_mask.sum().item()
        if total_masked == 0:
            # No differences - return zero loss
            return torch.tensor(0.0, device=xt.device, requires_grad=True)

        # Forward pass with t=1 (no time dependency)
        t = torch.ones((B,), device=xt.device)
        outputs = self.model(
            input_ids=xt,
            attention_mask=attention_mask,
            t=t,
            condition=condition
        )
        logits = outputs["logits"]

        # Verify that model output has expected batch dimension
        assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
        device = logits.device

        # Compute loss only on masked positions
        labels = input_ids_high.clone()
        labels[~diff_mask] = -1  # ignore positions that weren't masked
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-1
        )
        
        return loss

    def _loss_original(self, x, latent):
        """Original approach: random masking with time-based flow matching"""
        # Prepare inputs and labels
        input_ids = x["esm_input_ids"]
        attention_mask = x["esm_attention_mask"]
        labels = input_ids.clone()

        # Determine conditioning
        if self.condition_method == "no_use":
            condition = None
        elif self.condition_method in ("additive", "prefix"):
            condition = self.model.condition_proj(latent)
        else:
            raise ValueError(f"Condition Method {self.condition_method} not implemented")

        # Flatten batch if necessary
        if input_ids.ndim == 3:
            B1, B2, L = input_ids.shape
            input_ids = input_ids.view(B1 * B2, L)
            attention_mask = attention_mask.view(B1 * B2, L)
            labels = labels.view(B1 * B2, L)
            if condition is not None:
                condition = (
                    condition.unsqueeze(1)
                            .repeat(1, B2, 1)
                            .view(B1 * B2, -1)
                )

        # Sample timesteps and apply mask
        B, D = labels.shape
        t = torch.rand((B,))
        xt = labels.clone()
        mask = torch.rand((B,D)) < (1 - t[:, None])
        mask[:, 0] = False
        mask[:, -1] = False  # keep special tokens
        xt[mask] = self.mask_token_id
        
        # TODO: I am not 100% happy with this since the returned error is indistinguishable from perfect performance, but I think it is ok for now.
        # Check if any tokens are masked
        total_masked = mask.sum().item()
        if total_masked == 0:
            # No tokens masked - return zero loss (common at high timesteps)
            return torch.tensor(0.0, device=xt.device, requires_grad=True)

        # Forward pass
        outputs = self.model(
            input_ids=xt,
            attention_mask=attention_mask,
            t=t,
            condition=condition
        )
        logits = outputs["logits"]

        # Verify that model output has expected batch dimension
        assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
        device = logits.device

        # Compute loss
        labels_before = labels.clone()  # Keep original for debugging
        labels[xt != self.mask_token_id] = -1  # ignore unmasked positions
        
        # Try to compute loss and catch any issues
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=-1
        )
        
        return loss

    # MAJOR TODO: the number of positions to mask is not the same as the glob_edit_limit. It should be delta_x.

    def _sample_batch(self, latent_i, samples_low_seq=None, seq2seq_edit_limit=None, wt_seq=None, glob_edit_limit=None):
        """Conditional sampling based on seq2seq_mode"""
        if self.seq2seq_mode:
            return self._sample_batch_seq2seq(latent_i, samples_low_seq, seq2seq_edit_limit, wt_seq, glob_edit_limit)
        else:
            return self._sample_batch_original(latent_i)

    def _sample_batch_seq2seq(self, latent_i, samples_low_seq, seq2seq_edit_limit, wt_seq=None, glob_edit_limit=None):
        """
        New sampling approach: start with samples_low sequence, mask positions based on 
        differences from wt_seq, and unmask all at once.
        
        Args:
            latent_i: Latent conditioning vector
            samples_low_seq: Input sequence (samples_low) as token IDs with BOS/EOS
            seq2seq_edit_limit: Maximum number of positions to mask
            wt_seq: Wild-type sequence (string format) to compare against
            glob_edit_limit: Maximum number of allowed maskable positions
        """
        # TODO: This method appears to handle general batch sizes but should be tested
        # thoroughly for B>1 case to ensure proper tensor shapes and sampling behavior
        device = latent_i.device
        B = latent_i.size(0)

        # TODO: This method currently only works correctly for B=1. For B>1, the return shape
        # and sampling logic need to be updated to handle multiple batch elements properly.
        if B != 1:
            raise NotImplementedError("_sample_batch_seq2seq currently only supports batch_size=1")
        
        
        # Start with the samples_low sequence (including BOS and EOS)
        xt = samples_low_seq.clone()  # Shape: (B, D) where D includes BOS and EOS
        D = xt.size(1)
        
        attention_mask = torch.ones_like(xt)
        condition = self.model.condition_proj(latent_i)
        
        # Sample number of tokens to mask uniformly between 1 and seq2seq_edit_limit
        num_to_mask = torch.randint(1, seq2seq_edit_limit + 1, (B,))
        
        # For each sequence in the batch, determine maskable positions
        for b in range(B):
            # Decode the input sequence without BOS/EOS to compare with wt_seq
            input_seq_tokens = xt[b, 1:-1]  # Remove BOS and EOS
            input_seq_str = self.tokenizer.decode(input_seq_tokens.tolist(), skip_special_tokens=True).replace(" ", "")
            
            # Find positions where input sequence differs from wt_seq
            diff_positions = []
            if wt_seq is not None:
                assert len(input_seq_str) == len(wt_seq), "Input sequence and wild-type sequence must have same length"
                for pos in range(len(wt_seq)):
                    if input_seq_str[pos] != wt_seq[pos]:
                        diff_positions.append(pos + 1)  # +1 to account for BOS token
            
            # Convert to tensor
            diff_positions = torch.tensor(diff_positions, device=device, dtype=torch.long)
            
            # Determine all possible maskable positions (excluding BOS and EOS)
            all_maskable = torch.arange(1, D-1, device=device, dtype=torch.long)
            
            # If we have fewer differences than glob_edit_limit, add more positions
            if len(diff_positions) < glob_edit_limit and glob_edit_limit is not None:
                # Find positions that are not already in diff_positions
                remaining_positions = all_maskable[~torch.isin(all_maskable, diff_positions)]
                
                # Calculate how many more positions we need
                num_additional = glob_edit_limit - len(diff_positions)
                
                if len(remaining_positions) >= num_additional:
                    # Randomly select additional positions
                    perm = torch.randperm(len(remaining_positions))
                    additional_positions = remaining_positions[perm[:num_additional]]
                    # Combine with difference positions
                    allowed_maskable = torch.cat([diff_positions, additional_positions])
                else:
                    # Use all remaining positions
                    allowed_maskable = torch.cat([diff_positions, remaining_positions])
            else:
                # Use only the difference positions (or all if wt_seq is None)
                if len(diff_positions) > 0:
                    allowed_maskable = diff_positions
                else:
                    # Fallback: use all maskable positions if no differences found
                    allowed_maskable = all_maskable
            
            # Ensure we don't try to mask more positions than available
            actual_num_to_mask = min(num_to_mask[b].item(), len(allowed_maskable))
            
            if actual_num_to_mask > 0:
                # Randomly select positions to mask from allowed positions
                perm = torch.randperm(len(allowed_maskable))
                positions_to_mask = allowed_maskable[perm[:actual_num_to_mask]]
                
                # Apply masking
                xt[b, positions_to_mask] = self.mask_token_id
        
        # Single forward pass to unmask all positions at once
        t = torch.ones((B,), device=device)  # Always use t=1
        outputs = self.model(
            input_ids=xt,
            attention_mask=attention_mask,
            t=t,
            condition=condition
        )
        
        logits = outputs["logits"]  # shape (B, D, V)
        B, D, V = logits.size()
        
        # Verify that model output has expected batch dimension
        assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
        device = logits.device
        
        # Apply vocabulary constraints
        mask = torch.full((B, D, V), -float('inf'), device=device)
        
        # Allow only AAs at middle positions
        mask[:, 1:-1, self.aa_ids] = 0.0
        
        # Allow BOS at pos 0, EOS at pos -1
        mask[:, 0, self.bos_id] = 0.0
        mask[:, -1, self.eos_id] = 0.0
        
        # Apply constraints
        logits = logits + mask
        
        # Sample from the constrained distribution
        probs = F.softmax(logits, dim=-1)
        x1 = Categorical(probs).sample()
        
        # Replace only the masked positions
        masked_positions = (xt == self.mask_token_id)
        xt[masked_positions] = x1[masked_positions]
        
        # Return without BOS and EOS tokens
        return xt[:, 1:-1]  # Shape: (B, D-2)

    def _sample_batch_original(self, latent_i):
        """Original sampling approach: iterative demasking from fully masked sequence"""
        device = latent_i.device
        B = latent_i.size(0)
        
        # TODO: This method currently only works correctly for B=1. For B>1, the return shape
        # and sampling logic need to be updated to handle multiple batch elements properly.
        if B != 1:
            raise NotImplementedError("_sample_batch_original currently only supports batch_size=1")
        
        D = self.max_length + 2
        xt = torch.full((B, D), self.mask_token_id, device=device, dtype=torch.long)
        xt[:, 0] = self.bos_id
        xt[:, -1] = self.eos_id
        attention_mask = torch.ones_like(xt)
        condition = self.model.condition_proj(latent_i)

        t = 0.0
        dt = 0.001
        while t < 1.0:
            outputs = self.model(
                input_ids=xt,
                attention_mask=attention_mask,
                t=torch.full((B,), t, device=device),
                condition=condition
            )

            logits = outputs["logits"]  # shape (B, D, V)

            # Verify that model output has expected batch dimension
            assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
            _, D_logits, V = logits.size()
            #print(f"DOUBLE CHECKING:  {B}, {D_logits} and {V}")
            device = logits.device

            # 1) start from all–∞
            mask = torch.full((B, D_logits, V), -float('inf'), device=device)

            # 2) allow only AAs at *every* position
            mask[:, 1:-1, self.aa_ids] = 0.0

            # 3) allow BOS at pos 0, EOS at pos D–1
            mask[:, 0, self.bos_id]   = 0.0
            mask[:, -1, self.eos_id]  = 0.0

            # 4) apply
            logits = logits + mask

            probs = F.softmax(logits, dim=-1)
            x1 = Categorical(probs).sample()

            # decide which positions to unmask …
            will_unmask = (torch.rand((B, D_logits), device=device) < (dt / (1 - t)))
            # mask out CLS/EOS etc…
            currently_masked = (xt == self.mask_token_id)
            will_unmask &= currently_masked
            will_unmask[:, 0] = will_unmask[:, -1] = False

            xt[will_unmask] = x1[will_unmask]
            
            # Check if all positions are unmasked (excluding BOS and EOS)
            remaining_masked = (xt[:, 1:-1] == self.mask_token_id).any()
            if not remaining_masked:
                break
            
            t += dt
        
        remaining = (xt == self.mask_token_id)
        if remaining.any():
            # get model logits at exactly t=1.0
            t1 = torch.tensor([1.0] * B, device=device)
            outputs = self.model(
                input_ids=xt,
                attention_mask=attention_mask,
                t=t1,
                condition=condition
            )
            logits = outputs["logits"]

            # Verify that model output has expected batch dimension
            assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
            _, D_logits, V = logits.size()
            #print(f"DOUBLE CHECKING:  {B}, {D_logits} and {V}")
            device = logits.device

            # 1) start from all–∞
            mask = torch.full((B, D_logits, V), -float('inf'), device=device)

            # 2) allow only AAs at *every* position
            mask[:, 1:-1, self.aa_ids] = 0.0

            # 3) allow BOS at pos 0, EOS at pos D–1
            mask[:, 0, self.bos_id]   = 0.0
            mask[:, -1, self.eos_id]  = 0.0

            # 4) apply
            logits = logits + mask

            probs = F.softmax(logits, dim=-1)
            x1 = Categorical(probs).sample()
            
            # fill in only the still-masked positions
            xt[remaining] = x1[remaining]

        # TODO: This is a bit clunky. Clunky code 001. Will probably be weird/break if you go to larger batch sizes.
        # Return the inner tokens without BOS/EOS, maintaining batch dimension
        # Shape: (B, D-2)
        return xt[:, 1:-1]

    def sample(self, latent, wt_seq=None, glob_edit_limit=None, num_samples=1, return_texts=False, exclude_sequences=None, max_rejection_attempts=5, samples_low_seq=None, seq2seq_edit_limit=None):
        """
        Sampling method that supports both approaches based on seq2seq_mode.
        
        Args:
            latent: Latent vectors for conditioning
            wt_seq: Wild-type sequence for validation
            glob_edit_limit: Maximum allowed edit distance from wt_seq
            num_samples: Number of samples to generate
            return_texts: Whether to return text sequences in addition to token IDs
            exclude_sequences: Optional list of sequences (strings) to avoid generating
            max_rejection_attempts: Maximum number of rejection attempts before giving up
            samples_low_seq: Input sequence (samples_low) as token IDs with BOS/EOS (required for seq2seq_mode)
            seq2seq_edit_limit: Maximum number of positions to mask during sampling (used in seq2seq_mode)
        """
        if self.seq2seq_mode:
            if samples_low_seq is None:
                raise ValueError("samples_low_seq is required for seq2seq_mode")

        # Set model to evaluation mode for consistent sampling behavior
        was_training = self.model.training
        self.model.eval()

        B = latent.size(0)
        
        # TODO: Currently this method is designed for B=1. For B>1, the logic needs to be updated
        # to handle multiple batch elements properly, including rejection sampling per batch element
        # and proper tensor shape management.
        if B != 1:
            raise NotImplementedError("sample() currently only supports batch_size=1. For B>1, need to implement proper batch handling.")
        
        all_ids = []
        all_texts = []
        # TODO: note how below you implement rejection sampling for sequences already contained in exclude_sequences, but you don't check whether among the newly generated sequences there are any duplicates.
        # Convert exclude_sequences to a set for faster lookup
        excluded_set = set(exclude_sequences) if exclude_sequences is not None else set()

        print("!!!! LATENT SHAPE", latent.shape)
        
        for i in range(num_samples):
            latent_i = latent[:]

            # Rejection sampling loop
            attempts = 0
            while attempts < max_rejection_attempts:
                # Generate sequence using appropriate approach
                if self.seq2seq_mode:
                    seq_ids = self._sample_batch(latent_i, samples_low_seq, seq2seq_edit_limit, wt_seq, glob_edit_limit)
                else:
                    seq_ids = self._sample_batch(latent_i)
                
                # TODO: This is a bit clunky. Clunky code 001. Be consistent in how you handle the batch dimension and don't forget that at some point in the future you might want to do batch sizes larger than 1.
                # seq_ids has shape (B, D-2), we need to handle the batch dimension
                if seq_ids.ndim == 2:
                    # TODO: Clunky code 001: this will definitely break at some point.
                    # Take the first sequence from the batch
                    seq_ids_single = seq_ids[0]
                else:
                    seq_ids_single = seq_ids
                
                # Decode sequence
                text = self.tokenizer.decode(seq_ids_single.tolist(), skip_special_tokens=True).replace(" ", "")
                
                # Check if this sequence should be excluded
                if text not in excluded_set and hamming_distance(text, wt_seq) <= glob_edit_limit:
                    # Sequence is acceptable, break out of rejection loop
                    break
                
                attempts += 1
                print(f"Sequence rejected (attempt {attempts}/{max_rejection_attempts}): {text}")
                if attempts == max_rejection_attempts:
                    print(f"Warning: Reached maximum rejection attempts ({max_rejection_attempts}) for sample {i}. Using last generated sequence.")
            
            # Store this sample
            all_ids.append(seq_ids_single)   # shape (D-2,)
            if return_texts:
                all_texts.append([text])  # wrap in list for consistency with expected API
        
        # Stack all samples: (num_samples, D-2)
        all_ids = torch.stack(all_ids, dim=0)
        
        if return_texts:
            # For B=1: return list with single element containing all sample texts
            # Format: [['text1', 'text2', ...]] where outer list has B=1 elements
            all_texts_formatted = [list(text_list[0] for text_list in all_texts)]
            

            return all_ids, all_texts_formatted
        
        # Restore original training mode
        if was_training:
            self.model.train()

        return all_ids
