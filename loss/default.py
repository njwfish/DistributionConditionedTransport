import torch


class LossManager:
    def __init__(self, set_size = None, microbatch_set_size=None, empty_cache_between_microbatches=True):
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

            # Microbatch across set dimension when requested
            if self.use_microbatching:

                source_latent_detached = source_latent.detach().requires_grad_(True)
                target_latent_detached = target_latent.detach().requires_grad_(True)

                total_loss_value = 0.0
                for start_s in range(0, self.set_size, self.microbatch_set_size):
                    end_s = min(start_s + self.microbatch_set_size, self.set_size)

                    x_source_mb = {}
                    x_target_mb = {}
                    for key, v in source_samples.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == self.set_size:
                            x_source_mb[key] = v[:, start_s:end_s, ...]
                        else:
                            x_source_mb[key] = v
                    for key, v in target_samples.items():
                        if isinstance(v, torch.Tensor) and v.dim() >= 3 and v.shape[1] == self.set_size:
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
                    frac = (end_s - start_s) / max(self.set_size, 1)
                    # TODO: make sure we really need the scaling here, maybe the loss method of the generator already takes care of it
                    loss_mb_scaled = loss_mb * frac
                    if torch.is_grad_enabled():
                        loss_mb_scaled.backward()
                    total_loss_value += float(loss_mb_scaled.detach().item())

                    # Cleanup microbatch tensors to reduce memory
                    del x_source_mb, x_target_mb, loss_mb, loss_mb_scaled
                    #if self.empty_cache_between_microbatches and torch.cuda.is_available():
                    #    torch.cuda.empty_cache()

                # Backprop latent grads to encoder once
                if torch.is_grad_enabled():
                    torch.autograd.backward(
                        [source_latent, target_latent],
                        [source_latent_detached.grad, target_latent_detached.grad]
                    )

                recon_tensor = torch.tensor(total_loss_value, device=device)
                losses['reconstruction_loss'] = recon_tensor.detach()
                return recon_tensor.detach(), losses

            # No microbatching requested: single pass
            recon_loss = generator.loss(source_samples, target_samples, source_latent, target_latent)
            if torch.is_grad_enabled():
                recon_loss.backward()

        loss += recon_loss
        losses['reconstruction_loss'] = recon_loss.detach()
        return loss.detach(), losses