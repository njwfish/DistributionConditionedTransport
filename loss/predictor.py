import torch
    

class PredictorLossManager:
    def __init__(self, use_predicted_latent=False):
        self.use_predicted_latent = use_predicted_latent

    def loss(self, encoder, generator, batch, device):
        losses = {}
        loss = 0

        # Handle both tensor and dictionary samples for source and target
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            # compute predictor loss and get predicted target latent
            predictor_loss, pred_target_latent = encoder.predictor.loss(source_latent, target_latent)

            # compute generator loss
            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent

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

            pred_target_latent = encoder.predictor(source_latent)

            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        losses['predictor_loss'] = predictor_loss

        return loss, losses