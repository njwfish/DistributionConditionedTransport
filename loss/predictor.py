import torch


class PredictorLossManager:
    def __init__(self, predictor_loss_weight=1.0, generator_source_only=False, predictor_cotrain_direct=False):
        """
        Args:
            predictor_loss_weight: Weight for the predictor loss term (used when predictor_cotrain_direct=False)
            generator_source_only: If True, target_latent is not passed to generator
            predictor_cotrain_direct: If True, use predictor output as target_latent for generator
                                      (when train_predictor_bool=True) instead of adding a predictor loss term.
                                      This trains the predictor through the generator's gradients.
        """
        self.predictor_loss_weight = predictor_loss_weight
        self.generator_source_only = generator_source_only
        self.predictor_cotrain_direct = predictor_cotrain_direct
        
    def loss(self, encoder, generator, predictor, batch, device):
        losses = {}
        loss = 0

        # Handle both tensor and dictionary samples for source and target
        if isinstance(batch['source_samples'], torch.Tensor):
            source_samples = batch['source_samples'].to(device)
            target_samples = batch['target_samples'].to(device)
            
            source_latent = encoder(source_samples)
            target_latent = encoder(target_samples)

            # compute predictor loss
            # NOTE: defaulting to false such that existing dataset classes simply won't cotrain accidentally.
            train_predictor_bools = batch.get("train_predictor_bool", False)
            
            if self.predictor_cotrain_direct:
                # Use predictor output directly as target_latent for samples where train_predictor_bool is True
                # This trains the predictor through the generator's gradients instead of a separate loss term
                if not isinstance(train_predictor_bools, torch.Tensor):
                    train_predictor_bools_tensor = torch.tensor(train_predictor_bools, device=device)
                else:
                    train_predictor_bools_tensor = train_predictor_bools.to(device)
                
                # Get predicted target latent from predictor
                pred_target_latent = predictor(source_latent)
                
                # Create combined target latent: use predicted for train_predictor_bool=True, true for False
                if train_predictor_bools_tensor.any():
                    # Expand mask to match latent dimensions for proper broadcasting
                    mask = train_predictor_bools_tensor.bool().unsqueeze(-1)
                    target_latent_for_generator = torch.where(mask, pred_target_latent, target_latent)
                else:
                    # No samples need predictor output, use true target_latent
                    target_latent_for_generator = target_latent
                
                # No separate predictor loss when using direct cotraining
                predictor_loss = torch.tensor(0.0, device=device)
            else:
                # Original behavior: add predictor loss term
                predictor_loss = predictor.loss(
                    source_latent,
                    target_latent,
                    train_predictor_bools,
                )
                target_latent_for_generator = target_latent

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

            # Get train_predictor_bool from batch (defaulting to False for backwards compatibility)
            train_predictor_bools = batch.get("train_predictor_bool", False)
            
            if self.predictor_cotrain_direct:
                # Use predictor output directly as target_latent for samples where train_predictor_bool is True
                # This trains the predictor through the generator's gradients instead of a separate loss term
                if not isinstance(train_predictor_bools, torch.Tensor):
                    train_predictor_bools_tensor = torch.tensor(train_predictor_bools, device=device)
                else:
                    train_predictor_bools_tensor = train_predictor_bools.to(device)
                
                # Get predicted target latent from predictor
                pred_target_latent = predictor(source_latent)
                
                # Create combined target latent: use predicted for train_predictor_bool=True, true for False
                if train_predictor_bools_tensor.any():
                    # Expand mask to match latent dimensions for proper broadcasting
                    mask = train_predictor_bools_tensor.bool().unsqueeze(-1)
                    target_latent_for_generator = torch.where(mask, pred_target_latent, target_latent)
                else:
                    # No samples need predictor output, use true target_latent
                    target_latent_for_generator = target_latent
                
                # No separate predictor loss when using direct cotraining
                predictor_loss = torch.tensor(0.0, device=device)
            else:
                # Original behavior: add predictor loss term
                predictor_loss = predictor.loss(
                    source_latent,
                    target_latent,
                    train_predictor_bools,
                )
                target_latent_for_generator = target_latent

            if self.generator_source_only:
                target_latent_for_generator = None

            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)


        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss
        
        # Add predictor loss
        loss += predictor_loss * self.predictor_loss_weight
        losses['predictor_loss'] = predictor_loss * self.predictor_loss_weight

        return loss, losses