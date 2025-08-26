import torch
    

class PredictorLossManager:
    def __init__(self, use_predicted_latent=False, predictor_loss_weight=1.0, generator_source_only=False):
        self.use_predicted_latent = use_predicted_latent
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

            # compute predictor loss and get predicted target latent
            # Check if predictor requires dt conditioning
            if predictor.requires_condition:
                condition_scalars = (batch['source_idx'].to(device), batch['target_idx'].to(device))
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent, condition_scalars)
            else:
                # non-dt-conditioned predictor
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent)

            # compute generator loss
            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent
            if self.generator_source_only:
                target_latent_for_generator = None

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

            # Check if predictor requires conditioning
            if predictor.requires_condition:
                condition_scalars = (batch['source_idx'].to(device), batch['target_idx'].to(device))
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent, condition_scalars)
            else:
                # no conditioning
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent)

            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent
            if self.generator_source_only:
                target_latent_for_generator = None

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)


        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # Add predictor loss
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses