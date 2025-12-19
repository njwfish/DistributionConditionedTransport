import torch
from utils.conditioning import get_train_predictor_bool
    

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
            # Build conditioning tuple per predictor.condition_type
            train_predictor_bools = batch.get("train_predictor_bool", False) #get_train_predictor_bool(batch, device)
            predictor_loss = predictor.loss(
                source_latent,
                target_latent,
                train_predictor_bools,
            )

            if self.generator_source_only:
                target_latent = None

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

            # Build conditioning tuple per predictor.condition_type
            train_predictor_bools = get_train_predictor_bool(batch, device)
            predictor_loss = predictor.loss(
                source_latent,
                target_latent,
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