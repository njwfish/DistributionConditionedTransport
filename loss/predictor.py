import torch
    

class PredictorLossManager:
    def __init__(self, use_predicted_latent=False, predictor_loss_weight=1.0):
        self.use_predicted_latent = use_predicted_latent
        self.predictor_loss_weight = predictor_loss_weight

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
            # Check if predictor requires dt conditioning
            if hasattr(encoder.predictor, 'requires_dt') and encoder.predictor.requires_dt:
                # dt-conditioned predictor
                if 'dt' not in batch:
                    raise ValueError("dt not found in batch but required by dt-conditioned predictor")
                dt = batch['dt'].to(device)
                predictor_loss, pred_target_latent = encoder.predictor.loss(source_latent, target_latent, dt)
            else:
                # non-dt-conditioned predictor
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

            # Handle predictors (both dt-conditioned and non-dt-conditioned)
            # Check if predictor requires dt conditioning
            if hasattr(encoder.predictor, 'requires_dt') and encoder.predictor.requires_dt:
                # dt-conditioned predictor
                if 'dt' not in batch:
                    raise ValueError("dt not found in batch but required by dt-conditioned predictor")
                dt = batch['dt'].to(device)
                predictor_loss, pred_target_latent = encoder.predictor.loss(source_latent, target_latent, dt)
            else:
                # non-dt-conditioned predictor
                predictor_loss, pred_target_latent = encoder.predictor.loss(source_latent, target_latent)

            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)

        # TODO: make sure you understand why exactly the loss only gets the recon loss term and not the predictor loss.

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # TODO: actually, I think there still is an issue, because I think this will still not add any predictor loss for the older models (so the ones not conditioned on dt). But check this.
        # Add predictor loss
        # TODO: is this the right way to go about this? I mean, how could one possibly estimate the correct predictor loss weight?
        # TODO: if you directly want to condition on the predicted latent during training you might be better off not adding an extra loss term here, but we can just leave it for now and see how things perform.
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses