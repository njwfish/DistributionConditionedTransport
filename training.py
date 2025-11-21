import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import shutil
from tqdm import tqdm
import logging
import wandb
from utils.hash_utils import get_output_dir, find_matching_output_dir
from utils.visualization import visualize_text_data, visualize_coupled_data
from omegaconf import OmegaConf, DictConfig
import contextlib

class Trainer:
    def __init__(
        self,
        num_epochs=100,
        log_interval=10,
        save_interval=20,
        eval_interval=5,
        sub_epoch_interval=1_000,
        early_stopping=True,
        patience=10,
        use_tqdm=True,
        mask_context_prob=0.0,
        sub_epoch=None,
        gradient_accumulation_steps=1,
        use_amp=False,
    ):
        """
        Initialize the trainer.
        
        Args:
            num_epochs: Number of epochs to train for
            log_interval: How often to log training metrics (in batches)
            save_interval: How often to save model checkpoints (in epochs)
            eval_interval: How often to run evaluation (in epochs)
            early_stopping: Whether to use early stopping
            patience: Number of evaluations with no improvement before early stopping
            use_tqdm: Whether to use tqdm progress bars
        """
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.eval_interval = eval_interval
        self.sub_epoch_interval = sub_epoch_interval
        if sub_epoch is None:
            self.sub_epoch = save_interval == 1
        else:
            self.sub_epoch = sub_epoch
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_tqdm = use_tqdm
        self.mask_context_prob = mask_context_prob
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.use_amp = use_amp
        
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.no_improve_count = 0
        # log sub_epoch_interval
        self.logger.info(f"Sub epoch interval: {self.sub_epoch_interval}, save interval: {self.save_interval}, eval interval: {self.eval_interval}")
    
    def _find_similar_experiment_by_name(self, experiment_name, current_config, base_dir):
        """
        Find experiments with the same name but different epoch counts.
        
        Args:
            experiment_name: Name of the experiment (without hash)
            current_config: Current configuration
            base_dir: Base output directory
            
        Returns:
            List of tuples (directory, num_epochs) of matching experiments
        """
        matching_experiments = []
        
        # Extract current num_epochs from config
        current_num_epochs = None
        if isinstance(current_config, DictConfig):
            config_dict = OmegaConf.to_container(current_config, resolve=True)
        else:
            config_dict = current_config.copy()
        
        current_num_epochs = config_dict['training']['num_epochs']

        config_dict.pop('training')

        # Look for directories with the same experiment name prefix
        for item in os.listdir(base_dir):
            if not os.path.isdir(os.path.join(base_dir, item)):
                continue
                
            # Check if this directory has the same experiment name prefix
            if item.startswith(f"{experiment_name}_"):
                exp_dir = os.path.join(base_dir, item)
                config_path = os.path.join(exp_dir, 'config.yaml')
                
                if not os.path.exists(config_path):
                    continue
                    
                try:
                    # Load the experiment config
                    exp_config = OmegaConf.load(config_path)
                    exp_config_dict = OmegaConf.to_container(exp_config, resolve=True)
                    
                    # Check if this experiment has a different num_epochs
                    exp_num_epochs = exp_config_dict['training']['num_epochs']
                    
                    if exp_num_epochs != current_num_epochs:
                        exp_config_dict.pop('training')
                        
                        # Compare the configs without epochs
                        if self._configs_match(config_dict, exp_config_dict):
                            self.logger.info(f"Found similar experiment: {item} with {exp_num_epochs} epochs")
                            matching_experiments.append((exp_dir, exp_num_epochs))
                except Exception as e:
                    self.logger.warning(f"Error comparing config in {exp_dir}: {e}")
        
        return matching_experiments
    
    def _configs_match(self, config1, config2):
        """Compare two configs for equality."""
        if set(config1.keys()) != set(config2.keys()):
            return False
            
        for key in config1.keys():
            if config1[key] != config2[key]:
                return False
                    
        return True
    
    def _find_latest_checkpoint(self, directory):
        """Find the latest checkpoint in a directory."""
        best_model_path = os.path.join(directory, "best_model.pt")
        if os.path.exists(best_model_path):
            return best_model_path
            
        latest_checkpoint = None
        latest_epoch = -1
        
        for filename in os.listdir(directory):
            if filename.startswith("checkpoint_epoch_"):
                try:
                    epoch_num = int(filename.split("_")[-1].split(".")[0])
                    if epoch_num > latest_epoch:
                        latest_epoch = epoch_num
                        latest_checkpoint = os.path.join(directory, filename)
                except (ValueError, IndexError):
                    continue
                    
        return latest_checkpoint
    
    def train(
        self,
        encoder,
        generator,
        dataloader,
        optimizer,
        loss_manager,
        scheduler=None,
        predictor=None,
        device=None,
        output_dir='./outputs',
        config=None,
    ):
        """Train the model with W&B logging."""
        training_start = time.time()
        
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        encoder.to(device)
        generator.to(device)
        if predictor is not None:
            predictor.to(device)
        
        
        stats = {
            'train_losses': [],
            'eval_losses': [],
            'best_epoch': 0,
            'total_time': 0,
        }
        
        # If config is provided, use hash-based output directory
        if config is not None:
            output_dir = get_output_dir(config, base_dir=output_dir)
            # Log the used output directory
            self.logger.info(f"Using hash-based output directory: {output_dir}")
        else:
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Using specified output directory: {output_dir}")
        
        # Check for existing checkpoints in the output directory
        best_model_path = os.path.join(output_dir, "best_model.pt")
        last_checkpoint = None
        start_epoch = 0
        
        # Try to find the latest checkpoint if it exists
        for filename in os.listdir(output_dir):
            if filename.startswith("checkpoint_epoch_"):
                try:
                    epoch_num = int(filename.split("_")[-1].split(".")[0])
                    if last_checkpoint is None or epoch_num > start_epoch:
                        last_checkpoint = os.path.join(output_dir, filename)
                        start_epoch = epoch_num
                except (ValueError, IndexError):
                    continue
        if start_epoch == 0 and best_model_path is not None and os.path.exists(best_model_path):
            last_checkpoint = best_model_path
        
        # If no checkpoint found and we have a config, try to find a similar experiment
        if last_checkpoint is None and config is not None:
            self.logger.info("No checkpoints found. Looking for similar experiments with different epoch counts...")
            
            # Get experiment name from directory
            dir_name = os.path.basename(output_dir)
            experiment_name = dir_name.split('_')[0]  # Assume format is "name_hash"
            
            # Find base directory (parent of output_dir)
            base_dir = os.path.dirname(output_dir)
            
            # Find similar experiments with same name but different epoch counts
            similar_experiments = self._find_similar_experiment_by_name(experiment_name, config, base_dir)
            
            print("similar_experiments", similar_experiments)
            if similar_experiments:
                # Sort by epoch count (descending) to prioritize those with more epochs
                similar_experiments.sort(key=lambda x: x[1], reverse=True)
                
                for exp_dir, num_epochs in similar_experiments:
                    self.logger.info(f"Checking for checkpoints in similar experiment with {num_epochs} epochs: {exp_dir}")
                    
                    # Find the latest checkpoint in this experiment
                    checkpoint_path = self._find_latest_checkpoint(exp_dir)
                    
                    if checkpoint_path:
                        # Copy the checkpoint to our current directory
                        new_checkpoint_path = os.path.join(output_dir, os.path.basename(checkpoint_path))
                        try:
                            shutil.copy2(checkpoint_path, new_checkpoint_path)
                            self.logger.info(f"Copied checkpoint from similar experiment: {checkpoint_path} -> {new_checkpoint_path}")
                            last_checkpoint = new_checkpoint_path
                            break
                        except Exception as e:
                            self.logger.error(f"Error copying checkpoint: {e}")
        
        # Resume from checkpoint if found
        step = 0
        if last_checkpoint is not None and os.path.exists(last_checkpoint):
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            checkpoint = torch.load(last_checkpoint, weights_only=False)
            encoder.load_state_dict(checkpoint['encoder_state_dict'])

            generator.load_state_dict(checkpoint['generator_state_dict'])
            
            if predictor is not None and 'predictor_state_dict' in checkpoint:
                predictor.load_state_dict(checkpoint['predictor_state_dict'])

            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            if 'step' in checkpoint:
                step = checkpoint['step'] + 1
            else:
                step = (start_epoch - 1) * len(dataloader) + 1
            # Log resuming to W&B
            if wandb.run is not None:
                wandb.run.summary["resumed_from_epoch"] = start_epoch
        
        start_time = time.time()
        self.logger.info(f"Starting training on {device}...")
        
        # Configure AMP (prefer BF16 on CUDA if enabled)
        use_amp = bool(self.use_amp) and (device.type == "cuda")
        autocast_context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()

        # Main training loop
        for epoch in range(start_epoch, self.num_epochs):
            epoch_start = time.time()
            
            encoder.train()
            generator.train()
            if predictor is not None:
                predictor.train()
            
            epoch_losses = []
            
            # Create progress bar if requested
            if self.use_tqdm:
                pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
            else:
                pbar = dataloader
            
            
            for batch_idx, batch in enumerate(pbar):
                # Handle samples which can be either a tensor or a dictionary
                batch_loss_start = time.time()
                optimizer.zero_grad(set_to_none=True)
                with autocast_context:
                    loss, losses = loss_manager.loss(encoder, generator, predictor, batch, device)
                
                # Standard backward and optimizer step per batch
                loss.backward()
                optimizer.step()
                
                # Record batch loss
                current_loss = loss.item()
                epoch_losses.append(current_loss)
                
                # Log every log_interval batches
                if batch_idx % self.log_interval == 0:
                    self.logger.info(f"Epoch {epoch+1}, Batch {batch_idx}/{len(dataloader)}, Loss: {current_loss:.6f}")
                    if self.use_tqdm:
                        pbar.set_postfix(loss=f"{current_loss:.6f}")
                    
                    # Log batch metrics to W&B
                    if wandb.run is not None:
                        wandb.log({
                            "batch/loss": current_loss,
                            "batch/step": step,
                            "batch/epoch": epoch + 1,
                        } | {f'batch/{k}': v for k, v in losses.items()}, step=step)

                if self.sub_epoch and (step % self.sub_epoch_interval == 0) and (step != 0):
                    sub_epoch = step // self.sub_epoch_interval
                    if scheduler is not None:
                        scheduler.step()
                        current_lr = scheduler.get_last_lr()[0]
                        self.logger.info(f"Learning rate: {current_lr:.6f}")

                step += 1
            
            # Calculate average loss for this epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            stats['train_losses'].append(avg_epoch_loss)
            
            # Step the scheduler if provided
            if scheduler is not None and not self.sub_epoch:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                self.logger.info(f"Learning rate: {current_lr:.6f}")
            
            # Log epoch results
            self.logger.info(f"Epoch {epoch+1} complete. Avg Loss: {avg_epoch_loss:.6f}")
            
            # Log epoch metrics to W&B
            if wandb.run is not None:
                wandb_log = {
                    "epoch/train_loss": avg_epoch_loss,
                    "epoch/epoch": epoch + 1,
                }
                
                if scheduler is not None:
                    wandb_log["epoch/learning_rate"] = scheduler.get_last_lr()[0]
                
                wandb.log(wandb_log, step=step)
            
            # Save model checkpoint at regular intervals
            if (epoch + 1) % self.save_interval == 0:
                checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt")
                
                generator_state = generator.state_dict()
                
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'encoder_state_dict': encoder.state_dict(),
                    'generator_state_dict': generator_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_epoch_loss,
                    'step': step,
                }
                
                if predictor is not None:
                    checkpoint_data['predictor_state_dict'] = predictor.state_dict()
                
                torch.save(checkpoint_data, checkpoint_path)
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                # Log model checkpoint to W&B at the end of training
                if wandb.run is not None and (epoch + 1) == self.num_epochs:
                    wandb.save(checkpoint_path)
            
            # Evaluation and early stopping logic
            if ((epoch + 1) % self.eval_interval == 0 or (epoch + 1) == self.num_epochs):
                eval_loss = self._evaluate(encoder, generator, dataloader, device, loss_manager, predictor=predictor)
                stats['eval_losses'].append(eval_loss)
                
                self.logger.info(f"Evaluation Loss: {eval_loss:.6f}")
                
                # Log evaluation metrics to W&B
                if wandb.run is not None:
                    wandb.log({
                        "epoch/eval_loss": eval_loss,
                        "epoch/epoch": epoch + 1,
                    }, step=step)

                
                # Check if this is the best model so far
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    stats['best_epoch'] = epoch + 1
                    best_model_path = os.path.join(output_dir, "best_model.pt")
                    
                    generator_state = generator.state_dict()

                    checkpoint_data = {
                        'epoch': epoch + 1,
                        'encoder_state_dict': encoder.state_dict(),
                        'generator_state_dict': generator_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': eval_loss,
                        'step': step,
                    }
                    
                    if predictor is not None:
                        checkpoint_data['predictor_state_dict'] = predictor.state_dict()
                    
                    torch.save(checkpoint_data, best_model_path)
                    self.logger.info(f"New best model saved to {best_model_path}")
                    
                    # Log best model to W&B
                    if wandb.run is not None:
                        wandb.run.summary["best_epoch"] = epoch + 1
                        wandb.run.summary["best_loss"] = eval_loss
                        wandb.save(best_model_path)
                    
                    self.no_improve_count = 0
                else:
                    self.no_improve_count += 1
                    self.logger.info(f"No improvement for {self.no_improve_count} evaluations")
                
                # Early stopping check
                if self.early_stopping and self.no_improve_count >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    
                    # Log early stopping to W&B
                    if wandb.run is not None:
                        wandb.run.summary["stopped_early"] = True
                        wandb.run.summary["stopped_epoch"] = epoch + 1
                    
                    break
        
        # Record total training time
        stats['total_time'] = time.time() - start_time
        self.logger.info(f"Training completed in {stats['total_time']:.2f} seconds")
        
        # Log final stats to W&B
        if wandb.run is not None:
            wandb.run.summary["total_time"] = stats['total_time']
            if stats['train_losses']:
                wandb.run.summary["final_train_loss"] = stats['train_losses'][-1]
            if stats['eval_losses']:
                wandb.run.summary["final_eval_loss"] = stats['eval_losses'][-1]
        
        return output_dir, stats
    
    def _evaluate(self, encoder, generator, dataloader, device, loss_manager, predictor=None):
        """Run evaluation and return average loss."""
        encoder.eval()
        generator.eval()
        if predictor is not None:
            predictor.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # TODO: legacy code was not using loss manager here, is there any specific reason for this?
                # Use loss manager for consistent loss computation
                with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (bool(self.use_amp) and device.type == 'cuda') else contextlib.nullcontext()):
                    loss, losses = loss_manager.loss(encoder, generator, predictor, batch, device)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def generate_samples(self, encoder, generator, dataloader, num_samples=None, device=None):
        """Generate samples using the trained model."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        encoder.to(device)
        generator.to(device)
        
        encoder.eval()
        generator.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                # Handle samples which can be either a tensor or a dictionary
                if isinstance(batch['source_samples'], torch.Tensor):
                    source_samples = batch['source_samples'].to(device)
                    target_samples = batch['target_samples'].to(device)
                    batch_size, set_size, *data_shape = source_samples.shape
                    
                    # Encode samples to latent space
                    with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (bool(self.use_amp) and device.type == 'cuda') else contextlib.nullcontext()):
                        source_latent = encoder(source_samples)
                        target_latent = encoder(target_samples)

                        # TODO: if we actually want to call the generate_samples mehtod here, we need to make sure that this line is still up to date.
                        generated = generator.sample(source_samples.reshape(-1, *data_shape), source_latent, target_latent)
                    
                    return {
                        'source': source_samples.cpu(),
                        'target': target_samples.cpu(),
                        'generated': generated.cpu()
                    }
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
                    
                    # Keep raw texts for reference
                    source_raw_texts = batch.get('source_raw_texts', None)
                    target_raw_texts = batch.get('target_raw_texts', None)
                    
                    if source_raw_texts is not None:
                        set_size = len(source_raw_texts)
                        num_sets = len(source_raw_texts[0])
                        # reshape raw_texts list from [set_size, num_samples] to [num_samples, set_size]
                        source_raw_texts = [[source_raw_texts[j][i] for j in range(set_size)] for i in range(num_sets)]
                        target_raw_texts = [[target_raw_texts[j][i] for j in range(set_size)] for i in range(num_sets)]

                    # Encode samples to latent space
                    with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (bool(self.use_amp) and device.type == 'cuda') else contextlib.nullcontext()):
                        source_latent = encoder(source_samples)
                        target_latent = encoder(target_samples)
                    
                    # For dictionary data, we need to extract a representative source sample
                    # This is a simplified approach - in practice you might want something more sophisticated
                    source_sample = {}
                    for key, value in source_samples.items():
                        if isinstance(value, torch.Tensor) and value.ndim >= 2:
                            source_sample[key] = value[:, 0]  # Take first sample from each set
                        else:
                            source_sample[key] = value
                    
                    # Generate new samples
                    with (torch.autocast(device_type='cuda', dtype=torch.bfloat16) if (bool(self.use_amp) and device.type == 'cuda') else contextlib.nullcontext()):
                        generated = generator.sample(source_sample, source_latent, target_latent, num_samples=set_size, return_texts=True)
                    
                    if isinstance(generated, tuple):
                        # If generator returns both token ids and decoded texts
                        generated_ids, generated_texts = generated
                        return {
                            'source': source_samples,
                            'target': target_samples,
                            'generated': generated_ids.cpu(),
                            'source_texts': source_raw_texts,
                            'target_texts': target_raw_texts,
                            'generated_texts': generated_texts
                        }
                    else:
                        return {
                            'source': source_samples,
                            'target': target_samples,
                            'generated': generated.cpu(),
                            'source_texts': source_raw_texts,
                            'target_texts': target_raw_texts
                        }