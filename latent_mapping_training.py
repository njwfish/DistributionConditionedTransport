import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import logging
from predictor.predictor import MLPPredictor, RidgePredictor
import hydra

# TODO: make sure this really doesn't see the held-out data for the forecasting task.
class LatentMappingTrainer:
    def __init__(
        self,
        model,
        num_epochs=100,
        learning_rate=1e-3,
        batch_size=32,
        log_interval=10,
        save_interval=20,
        use_tqdm=True
    ):
        """
        Initialize the latent mapping trainer.
        
        Args:
            model: Hydra config for the predictor model to instantiate
            num_epochs: Number of epochs to train for
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training (number of set pairs)
            log_interval: How often to log training metrics (in batches)
            save_interval: How often to save model checkpoints (in epochs)
            use_tqdm: Whether to use tqdm progress bars
        """
        self.model_config = model
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.use_tqdm = use_tqdm
        
        self.logger = logging.getLogger(__name__)
        self.model = None  # Will be instantiated during training
        self.optimizer = None
    
    def train(
        self,
        encoder,
        dataloader,
        device=None,
        output_dir='./latent_mapping_outputs',
    ):
        """
        Train the latent mapping model.
        
        Args:
            encoder: Trained encoder model
            dataloader: DataLoader for source/target pairs
            device: Device to run training on
            output_dir: Directory to save outputs
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move encoder to device and set to eval mode
        encoder.to(device)
        encoder.eval()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine latent dimension by running a sample through encoder
        with torch.no_grad():
            sample_batch = next(iter(dataloader))
            if isinstance(sample_batch['source_samples'], torch.Tensor):
                sample_source = sample_batch['source_samples'][:1].to(device)  # Take first sample
                sample_encoding = encoder(sample_source)
                latent_dim = sample_encoding.shape[-1]
            else:
                # Handle dictionary samples
                sample_source = {}
                for key, value in sample_batch['source_samples'].items():
                    if isinstance(value, torch.Tensor):
                        sample_source[key] = value[:1].to(device)  # Take first sample
                    else:
                        sample_source[key] = [value[0]]  # Take first item if list
                sample_encoding = encoder(sample_source)
                latent_dim = sample_encoding.shape[-1]
        
        self.logger.info(f"Detected latent dimension: {latent_dim}")
        
        # Instantiate the predictor model with the detected latent dimension
        self.model = hydra.utils.instantiate(self.model_config, latent_dim=latent_dim)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.logger.info(f"Created {self.model.__class__.__name__} mapping model with {sum(p.numel() for p in self.model.parameters())} parameters")
        self.logger.info(f"Starting latent mapping training on {device}...")
        
        start_time = time.time()
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            
            # Create progress bar if requested
            if self.use_tqdm:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            else:
                pbar = dataloader
            
            for batch_idx, batch in enumerate(pbar):
                # Extract source and target latents
                source_latents, target_latents = self._extract_latents(batch, encoder, device)
                
                # Forward pass through mapping model
                if getattr(self.model, 'requires_dt', False):
                    # dt-conditioned predictor
                    dt = batch['dt'].to(device) if 'dt' in batch else torch.zeros(source_latents.shape[0], device=device)
                    predicted_target_latents = self.model(source_latents, dt)
                    # Compute loss
                    loss, _ = self.model.loss(source_latents, target_latents, dt)
                else:
                    # standard predictor
                    predicted_target_latents = self.model(source_latents)
                    # Compute loss (MSE)
                    loss, _ = self.model.loss(predicted_target_latents, target_latents)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Record loss
                epoch_losses.append(loss.item())
                
                # Log every log_interval batches
                if batch_idx % self.log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.6f}")
                    if self.use_tqdm:
                        pbar.set_postfix(loss=f"{loss.item():.6f}")
            
            # Calculate average loss for this epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            self.logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}")
            
            # Save model checkpoint at regular intervals
            if (epoch + 1) % self.save_interval == 0 or (epoch + 1) == self.num_epochs:
                checkpoint_path = os.path.join(output_dir, f"latent_mapping_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_epoch_loss,
                    'mapping_method': self.model.__class__.__name__,
                }, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_model_path = os.path.join(output_dir, "final_latent_mapping_model.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'mapping_method': self.model.__class__.__name__,
        }, final_model_path)
        
        total_time = time.time() - start_time
        self.logger.info(f"Latent mapping training completed in {total_time:.2f} seconds")
        self.logger.info(f"Final model saved to {final_model_path}")
        
        return final_model_path
    
    def _extract_latents(self, batch, encoder, device):
        """Extract source and target latents from a batch."""
        with torch.no_grad():
            if isinstance(batch['source_samples'], torch.Tensor):
                # Handle tensor samples
                source_samples = batch['source_samples'].to(device)
                target_samples = batch['target_samples'].to(device)
                
                source_latents = encoder(source_samples)  # [batch_size, latent_dim]
                target_latents = encoder(target_samples)  # [batch_size, latent_dim]
                
            else:
                # Handle dictionary samples (like text data)
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
                
                source_latents = encoder(source_samples)  # [batch_size, latent_dim]
                target_latents = encoder(target_samples)  # [batch_size, latent_dim]
        
        return source_latents, target_latents
    
    def evaluate(self, encoder, dataloader, device=None, num_eval_batches=10):
        """Evaluate the trained mapping model."""
        if self.model is None:
            raise RuntimeError("Model must be trained before evaluation")
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder.to(device)
        encoder.eval()
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                if num_eval_batches is not None and num_batches >= num_eval_batches:
                    break
                
                # Extract latents
                source_latents, target_latents = self._extract_latents(batch, encoder, device)
                
                # Forward pass
                if getattr(self.model, 'requires_dt', False):
                    # dt-conditioned predictor
                    dt = batch['dt'].to(device) if 'dt' in batch else torch.zeros(source_latents.shape[0], device=device)
                    predicted_target_latents = self.model(source_latents, dt)
                    # Compute loss
                    loss, _ = self.model.loss(source_latents, target_latents, dt)
                else:
                    # standard predictor
                    predicted_target_latents = self.model(source_latents)
                    # Compute loss
                    loss, _ = self.model.loss(predicted_target_latents, target_latents)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.logger.info(f"Evaluation Loss: {avg_loss:.6f}")
        return avg_loss

# TODO: make sure this really leaves everything unchanged.
def load_latent_mapping_model(cfg, checkpoint_path, device=None):
    """Load a trained latent mapping model from checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = hydra.utils.instantiate(cfg.predictor)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model 