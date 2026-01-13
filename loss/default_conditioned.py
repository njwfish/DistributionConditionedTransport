"""
Conditioned Default Loss Manager.

This is a modified version of default.py that handles cell_cond conditioning:
- Concatenates cell_cond with input features for the encoder (43 -> 45 dimensions)

This loss manager does NOT cotrain a predictor. Use this with train_predictor_posthoc: true
when the predictor should be trained separately after the main training.
"""

import torch


class LossManager:
    """
    Default loss manager with cell_cond conditioning for encoder inputs.
    
    Concatenates cell_cond_source/cell_cond_target with source/target samples
    before passing to the encoder.
    """
    
    def loss(self, encoder, generator, predictor, batch, device):
        """
        Compute loss with cell_cond conditioning.
        
        Expected batch keys:
            - source_samples: (batch_size, set_size, 43)
            - target_samples: (batch_size, set_size, 43)  
            - cell_cond_source: (batch_size, set_size, 2)
            - cell_cond_target: (batch_size, set_size, 2)
        """
        losses = {}
        loss = 0

        # Handle both tensor and dictionary samples for source and target
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            # Get cell conditioning information
            cell_cond_source = batch['cell_cond_source'].to(device)
            cell_cond_target = batch['cell_cond_target'].to(device)
            
            # Concatenate cell_cond with samples for encoder input
            # source_samples: (batch, set_size, 43) -> (batch, set_size, 45)
            source_input = torch.cat([source_samples, cell_cond_source], dim=-1)
            target_input = torch.cat([target_samples, cell_cond_target], dim=-1)
            
            # Encode the conditioned inputs
            source_latent = encoder(source_input)
            target_latent = encoder(target_input)
                
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

            # Encode samples to latent space
            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        return loss, losses
