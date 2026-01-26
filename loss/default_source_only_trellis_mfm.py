import torch

class LossManager:
    """
    Source-only loss manager.

    Only encodes the source distribution and passes zeros for the target latent.
    This allows the model to learn transport conditioned only on the source.
    """

    def loss(self, encoder, generator, predictor, batch, device):
        losses = {}
        loss = 0

        # Handle both tensor and dictionary samples for source and target
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)

            source_latent = encoder(source_samples)
            # Use zeros instead of encoding target - same shape as source_latent
            target_latent = torch.zeros_like(source_latent)

            # Extract treatment conditioning - take first element from each set
            # since treat_cond is the same for all cells in a set
            # treat_cond shape: [batch_size, set_size, num_treatments] -> [batch_size, num_treatments]
            treat_cond = batch['treat_cond'][:, 0, :].to(device)

            recon_loss = generator.loss(
                source_samples.view(-1, *source_samples.shape[2:]),
                target_samples.view(-1, *target_samples.shape[2:]),
                source_latent,
                target_latent,
                treat_cond
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
            # Use zeros instead of encoding target - same shape as source_latent
            target_latent = torch.zeros_like(source_latent)

            # For dictionary samples, treat_cond may not be available
            # Create zeros as placeholder if not present
            if 'treat_cond' in batch:
                treat_cond = batch['treat_cond'][:, 0, :].to(device)
            else:
                # Default to zeros if treat_cond not available (for non-trellis datasets)
                treat_cond = torch.zeros(source_latent.shape[0], 11, device=device)

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent, treat_cond)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        return loss, losses