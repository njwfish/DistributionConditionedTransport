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
    

    def loss(self, x_source, x_target, latent_source, latent_target):
        """Original approach: random masking with time-based flow matching"""
        # Prepare inputs and labels
        input_ids_source = x_source["esm_input_ids"]
        attention_mask_source = x_source["esm_attention_mask"]
        labels_source = input_ids_source.clone()

        input_ids_target = x_target["esm_input_ids"]
        attention_mask_target = x_target["esm_attention_mask"]
        labels_target = input_ids_target.clone()

        # Determine conditioning
        if self.condition_method == "no_use":
            condition_source = None
            condition_target = None
        elif self.condition_method in ("additive", "prefix"):
            condition_source = self.model.condition_proj(latent_source)
            condition_target = self.model.condition_proj(latent_target)
        else:
            raise ValueError(f"Condition Method {self.condition_method} not implemented")

        # Flatten batch if necessary
        if input_ids_source.ndim == 3:
            B1, B2, L = input_ids_source.shape
            input_ids_source = input_ids_source.view(B1 * B2, L)
            attention_mask_source = attention_mask_source.view(B1 * B2, L)
            labels_source = labels_source.view(B1 * B2, L)
            if condition_source is not None:
                condition_source = (
                    condition_source.unsqueeze(1)
                            .repeat(1, B2, 1)
                            .view(B1 * B2, -1)
                )
        # Flatten batch if necessary
        if input_ids_target.ndim == 3:
            B1, B2, L = input_ids_target.shape
            input_ids_target = input_ids_target.view(B1 * B2, L)
            attention_mask_target = attention_mask_target.view(B1 * B2, L)
            labels_target = labels_target.view(B1 * B2, L)
            if condition_target is not None:
                condition_target = (
                    condition_target.unsqueeze(1)
                            .repeat(1, B2, 1)
                            .view(B1 * B2, -1)
                )
        

        # Sample timesteps and apply mask
        B, D = labels_source.shape
        t = torch.rand((B,))
        xt = labels_target.clone()
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
            attention_mask=attention_mask_target,
            t=t,
            condition_source=condition_source,
            condition_target=condition_target
        )
        logits = outputs["logits"]

        # Verify that model output has expected batch dimension
        assert logits.size(0) == B, f"Model output batch size {logits.size(0)} doesn't match input batch size {B}"
        device = logits.device

        # Compute loss
        labels_before = labels_target.clone()  # Keep original for debugging
        labels_target[xt != self.mask_token_id] = -1  # ignore unmasked positions
        
        # Try to compute loss and catch any issues
        
        loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels_target,
            ignore_index=-1
        )
        
        return loss


    def _sample_batch(self, latent_source, latent_target):
        """Original sampling approach: iterative demasking from fully masked sequence"""
        device = latent_target.device
        B = latent_target.size(0)
        
        # TODO: This method currently only works correctly for B=1. For B>1, the return shape
        # and sampling logic need to be updated to handle multiple batch elements properly.
        if B != 1:
            raise NotImplementedError("_sample_batch_original currently only supports batch_size=1")
        
        D = self.max_length + 2
        xt = torch.full((B, D), self.mask_token_id, device=device, dtype=torch.long)
        xt[:, 0] = self.bos_id
        xt[:, -1] = self.eos_id
        attention_mask = torch.ones_like(xt)
        condition_source = self.model.condition_proj(latent_source)
        condition_target = self.model.condition_proj(latent_target)

        t = 0.0
        dt = 0.001
        while t < 1.0:
            outputs = self.model(
                input_ids=xt,
                attention_mask=attention_mask,
                t=torch.full((B,), t, device=device),
                condition_source=condition_source,
                condition_target=condition_target
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
                condition_source=condition_source,
                condition_target=condition_target
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

    def sample(self,x_source, latent_source, latent_target, num_samples=1, return_texts=False):
        """
        Sampling method that supports both approaches based on seq2seq_mode.
        
        Args:
            x_source: Source sequence
            latent_source: Source latent vector
            latent_target: Target latent vector
            num_samples: Number of samples to generate
            return_texts: Whether to return text sequences in addition to token IDs
        """

        self.model.eval()

        B = latent_target.size(0)
        
        # TODO: Currently this method is designed for B=1. For B>1, the logic needs to be updated
        # to handle multiple batch elements properly, including rejection sampling per batch element
        # and proper tensor shape management.
        if B != 1:
            raise NotImplementedError("sample() currently only supports batch_size=1. For B>1, need to implement proper batch handling.")
        
        all_ids = []
        all_texts = []
        # TODO: note how below you implement rejection sampling for sequences already contained in exclude_sequences, but you don't check whether among the newly generated sequences there are any duplicates.
        # Convert exclude_sequences to a set for faster lookup

        print("!!!! LATENT SHAPE", latent_target.shape)
        
        for i in range(num_samples):

            
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


        return all_ids
