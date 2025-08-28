import torch


# TODO: make sure all your modifications here below are sensible.
# TODO: remove conditioning on predictor during training as an option.

class PredictorLossManager:
    def __init__(self, use_predicted_latent=False, predictor_loss_weight=1.0, set_size=None, microbatch_set_size=None, empty_cache_between_microbatches=True):
        self.use_predicted_latent = use_predicted_latent
        self.predictor_loss_weight = predictor_loss_weight
        self.microbatch_set_size = microbatch_set_size
        self.empty_cache_between_microbatches = empty_cache_between_microbatches
        self.use_microbatching = bool(self.microbatch_set_size) and self.microbatch_set_size > 0
        self.set_size = set_size

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
            if predictor.requires_condition:
                condition_scalars = (batch['source_idx'].to(device), batch['target_idx'].to(device))
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent, condition_scalars)
            else:
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent)

            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent

            recon_loss = generator.loss(
                source_samples.view(-1, *source_samples.shape[2:]),
                target_samples.view(-1, *target_samples.shape[2:]),
                source_latent,
                target_latent_for_generator
            )
            if torch.is_grad_enabled():
                (recon_loss + self.predictor_loss_weight * predictor_loss).backward()
            loss = recon_loss + self.predictor_loss_weight * predictor_loss
            losses['reconstruction_loss'] = recon_loss.detach()
            losses['predictor_loss'] = (self.predictor_loss_weight * predictor_loss).detach()
            return loss.detach(), losses

        else:
            # For dictionary samples (ESM datasets), move tensors to device
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

            # Compute predictor loss/predicted latent
            if predictor.requires_condition:
                condition_scalars = (batch['source_idx'].to(device), batch['target_idx'].to(device))
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent, condition_scalars)
            else:
                predictor_loss, pred_target_latent = predictor.loss(source_latent, target_latent)

            # Microbatch across set dimension when requested
            if self.use_microbatching:


                source_latent_detached = source_latent.detach().requires_grad_(True)
                if self.use_predicted_latent:
                    target_for_gen = pred_target_latent
                else:
                    target_for_gen = target_latent
                target_latent_detached = target_for_gen.detach().requires_grad_(True)

                total_recon_value = 0.0
                for start_s in range(0, set_size, self.microbatch_set_size):
                    end_s = min(start_s + self.microbatch_set_size, set_size)

                    x_source_mb = {}
                    x_target_mb = {}
                    for key, v in source_samples.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == set_size:
                            x_source_mb[key] = v[:, start_s:end_s, ...]
                        else:
                            x_source_mb[key] = v
                    for key, v in target_samples.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == set_size:
                            x_target_mb[key] = v[:, start_s:end_s, ...]
                        else:
                            x_target_mb[key] = v

                    loss_mb = generator.loss(
                        x_source_mb,
                        x_target_mb,
                        source_latent_detached,
                        target_latent_detached,
                    )
                    # Normalize by total set_size to keep loss scale consistent
                    frac = (end_s - start_s) / max(set_size, 1)
                    loss_mb_scaled = loss_mb * frac
                    if torch.is_grad_enabled():
                        loss_mb_scaled.backward()
                    total_recon_value += float(loss_mb_scaled.detach().item())

                    del x_source_mb, x_target_mb, loss_mb, loss_mb_scaled
                    if self.empty_cache_between_microbatches and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Backprop latent grads to encoder/predictor once (retain graph for predictor loss)
                if torch.is_grad_enabled():
                    torch.autograd.backward(
                        [source_latent, target_for_gen],
                        [source_latent_detached.grad, target_latent_detached.grad],
                        retain_graph=True
                    )
                    # Now backprop predictor loss
                    (self.predictor_loss_weight * predictor_loss).backward()

                total_tensor = torch.tensor(total_recon_value, device=device) + (self.predictor_loss_weight * predictor_loss.detach())
                losses['reconstruction_loss'] = torch.tensor(total_recon_value, device=device)
                losses['predictor_loss'] = (self.predictor_loss_weight * predictor_loss).detach()
                return total_tensor.detach(), losses

            # No microbatching requested: single pass
            target_latent_for_generator = target_latent if not self.use_predicted_latent else pred_target_latent
            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent_for_generator)
            total = recon_loss + self.predictor_loss_weight * predictor_loss
            if torch.is_grad_enabled():
                total.backward()
            losses['reconstruction_loss'] = recon_loss.detach()
            losses['predictor_loss'] = (self.predictor_loss_weight * predictor_loss).detach()
            return total.detach(), losses