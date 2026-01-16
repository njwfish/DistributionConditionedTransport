"""
Loss manager for index-based one-hot encoding.

This loss manager is designed to work with IndexOneHotEncoder, passing dataset indices
to the encoder instead of actual sample data. The encoder then produces latent 
representations based solely on which data item (e.g., pfam family) was used.
"""

import torch


class LossManager:
    """
    Loss manager that passes indices (source_idx, target_idx) to the encoder 
    instead of the actual sample data.
    
    This is designed for use with IndexOneHotEncoder where the latent representation
    is simply a one-hot encoding of the dataset index.
    """
    
    def loss(self, encoder, generator, predictor, batch, device):
        """
        Compute loss using index-based encoding.
        
        Args:
            encoder: IndexOneHotEncoder that takes indices and returns latents
            generator: Generator model (e.g., DFM generator)
            predictor: Optional predictor model (unused in this loss manager)
            batch: Batch dict containing:
                - source_samples: dict or tensor of source samples
                - target_samples: dict or tensor of target samples  
                - source_idx: tensor of source dataset indices
                - target_idx: tensor of target dataset indices
            device: torch device
            
        Returns:
            Tuple of (total_loss, losses_dict)
        """
        losses = {}
        loss = 0
        
        # Get indices from batch
        source_idx = batch['source_idx'].to(device)
        target_idx = batch['target_idx'].to(device)
        
        # Encode using indices (not samples!)
        # The encoder returns one-hot vectors based on the index
        source_latent = encoder(source_idx)
        target_latent = encoder(target_idx)
        
        # Handle both tensor and dictionary samples for generator loss
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            recon_loss = generator.loss(
                source_samples.view(-1, *source_samples.shape[2:]), 
                target_samples.view(-1, *target_samples.shape[2:]),
                source_latent, 
                target_latent
            )
        else:
            # For dictionary samples (like Pfam dataset), move tensors to device
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

            # Generator loss uses samples for reconstruction, but latents from indices
            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        return loss, losses
