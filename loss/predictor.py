import torch


class PredictorLossManager:
    def __init__(self, predictor_loss_weight=1.0, generator_source_only=False):
        self.predictor_loss_weight = predictor_loss_weight
        self.generator_source_only = generator_source_only
        
    def loss(self, encoder, generator, predictor, batch, device):
        losses = {}
        loss = 0

        # Handle both tensor and dictionary samples for source and target
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            # Compute predictor loss
            predictor_loss = predictor.loss(source_latent, target_latent)

            # Compute generator loss (always use true target_latent)
            target_latent_for_generator = None if self.generator_source_only else target_latent

            recon_loss = generator.loss(
                source_samples.view(-1, *source_samples.shape[2:]), 
                target_samples.view(-1, *target_samples.shape[2:]),
                source_latent, 
                target_latent_for_generator
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

            # Compute predictor loss
            predictor_loss = predictor.loss(source_latent, target_latent)

            # Compute generator loss (always use true target_latent)
            target_latent_for_generator = None if self.generator_source_only else target_latent

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)


        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # Add predictor loss
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses