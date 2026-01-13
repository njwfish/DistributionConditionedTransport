"""
Conditioned Predictor Loss Manager.

This is a modified version of predictor.py that handles conditioning:
1. Concatenates cell_cond with input features for the encoder (43 -> 45 dimensions)
2. Passes treat_cond to the conditioned predictor

The encoder will receive features concatenated with cell type one-hot encoding:
    source_input = [source_samples, cell_cond_source]  # shape: (batch, set_size, 45)
    target_input = [target_samples, cell_cond_target]  # shape: (batch, set_size, 45)

The predictor receives source_latent concatenated with treat_cond.
"""

import torch


class PredictorConditionedLossManager:
    """
    Loss manager that incorporates conditioning information:
    - cell_cond is concatenated with input features before encoding
    - treat_cond is passed to the conditioned predictor
    """
    
    def __init__(self, predictor_loss_weight=1.0, generator_source_only=False):
        self.predictor_loss_weight = predictor_loss_weight
        self.generator_source_only = generator_source_only
        
    def loss(self, encoder, generator, predictor, batch, device):
        """
        Compute loss with conditioning.
        
        Expected batch keys:
            - source_samples: (batch_size, set_size, 43)
            - target_samples: (batch_size, set_size, 43)  
            - cell_cond_source: (batch_size, set_size, 2)
            - cell_cond_target: (batch_size, set_size, 2)
            - treat_cond: (batch_size, set_size, 11)
            - train_predictor_bool: (batch_size,)
        """
        losses = {}
        loss = 0

        # Handle tensor samples (expected for trellis dataset)
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            # Get conditioning information
            cell_cond_source = batch['cell_cond_source'].to(device)
            cell_cond_target = batch['cell_cond_target'].to(device)
            treat_cond = batch['treat_cond'].to(device)
            
            # Concatenate cell_cond with samples for encoder input
            # source_samples: (batch, set_size, 43) -> (batch, set_size, 45)
            # cell_cond_source: (batch, set_size, 2)
            source_input = torch.cat([source_samples, cell_cond_source], dim=-1)
            target_input = torch.cat([target_samples, cell_cond_target], dim=-1)
            
            # Encode the conditioned inputs
            source_latent = encoder(source_input)
            target_latent = encoder(target_input)

            # Compute predictor loss with treatment conditioning
            # NOTE: defaulting to false such that existing dataset classes simply won't cotrain accidentally.
            train_predictor_bools = batch.get("train_predictor_bool", False)
            
            # For predictor loss, we need to aggregate treat_cond from (batch, set_size, 11) to (batch, 11)
            # We use mean across set_size dimension (all cells in a set have the same treatment)
            treat_cond_aggregated = treat_cond.mean(dim=1)  # (batch, 11)
            
            predictor_loss = predictor.loss(
                source_latent,
                target_latent,
                treat_cond_aggregated,
                train_predictor_bools,
            )

            if self.generator_source_only:
                target_latent = None

            # Generator loss uses original (non-conditioned) samples
            # The cell_cond is only used for computing better latents via the encoder
            recon_loss = generator.loss(
                source_samples.view(-1, *source_samples.shape[2:]), 
                target_samples.view(-1, *target_samples.shape[2:]),
                source_latent, 
                target_latent
            )

        else:
            # For dictionary samples (like PubMed dataset), move tensors to device
            source_samples = {}
            target_samples = {}
            
            for key, value in batch['source_samples'].items():
                if isinstance(value, torch.Tensor):
                    source_samples[key] = value.to(device)
                else:
                    source_samples[key] = value
                    
            for key, value in batch['target_samples'].items():
                if isinstance(value, torch.Tensor):
                    target_samples[key] = value.to(device)
                else:
                    target_samples[key] = value

            # Get conditioning information for dictionary format
            cell_cond_source = batch['cell_cond_source'].to(device)
            cell_cond_target = batch['cell_cond_target'].to(device)
            treat_cond = batch['treat_cond'].to(device)

            # Note: For dictionary samples, we need a different approach to concatenate cell_cond
            # This depends on how the samples are structured - for now, assume tensor-like handling
            # Encode samples to latent space
            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            # Get train_predictor_bool from batch (defaulting to False for backwards compatibility)
            train_predictor_bools = batch.get("train_predictor_bool", False)
            
            # Aggregate treat_cond
            treat_cond_aggregated = treat_cond.mean(dim=1)
            
            predictor_loss = predictor.loss(
                source_latent,
                target_latent,
                treat_cond_aggregated,
                train_predictor_bools,
            )

            if self.generator_source_only:
                target_latent = None

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent)


        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # Add predictor loss
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses
