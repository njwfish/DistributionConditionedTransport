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
            if getattr(encoder.predictor, 'requires_dt', False):
                # dt-conditioned predictor
                dt = batch['dt'].to(device) if 'dt' in batch else torch.zeros(source_latent.shape[0], device=device)
                predictor_loss, pred_target_latent = encoder.predictor.loss(source_latent, target_latent, dt)
            else:
                # standard predictor
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

            # Handle dt-conditioned predictors
            if getattr(encoder.predictor, 'requires_dt', False):
                # dt-conditioned predictor
                dt = batch['dt'].to(device) if 'dt' in batch else torch.zeros(source_latent.shape[0], device=device)
                pred_target_latent = encoder.predictor(source_latent, dt)
                # Compute predictor loss for this case
                # TODO: make sure that this should really always be computed and added to the loss, no matter which setting we are using (joint or separate training of predictor).
                predictor_loss, _ = encoder.predictor.loss(source_latent, target_latent, dt)
            else:
                # standard predictor
                pred_target_latent = encoder.predictor(source_latent)
                # Compute predictor loss for standard predictors too
                predictor_loss, _ = encoder.predictor.loss(source_latent, target_latent)

            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)

        # TODO: make sure you understand why exactly the loss only gets the recon loss term and not the predictor loss.

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # TODO: actually, I think there still is an issue, because I think this will still not add any predictor loss for the older models (so the ones not conditioned on dt). But check this.
        # Add predictor loss (it will be 0 for non-dt-conditioned predictors or if not computed above)
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses