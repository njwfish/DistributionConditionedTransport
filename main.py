import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset
import logging
import wandb
import os
import numpy as np

# Import our resolver for sum operations
import utils.hash_utils as hash_utils
from latent_mapping_training import LatentMappingTrainer

def create_train_val_split(dataset, validation_split, shuffle_before_split=True, seed=None):
    """
    Create train and validation splits from a dataset.
    
    Args:
        dataset: The original dataset
        validation_split: Fraction of data to use for validation (0.0 to 1.0)
        shuffle_before_split: Whether to shuffle indices before splitting
        seed: Random seed for reproducibility
        
    Returns:
        train_dataset, val_dataset: Subset datasets for training and validation
    """
    if validation_split <= 0.0 or validation_split >= 1.0:
        raise ValueError("validation_split must be between 0.0 and 1.0")
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    
    if shuffle_before_split:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    
    # Calculate split point
    val_size = int(validation_split * dataset_size)
    train_size = dataset_size - val_size
    
    # Simple random split
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    return train_dataset, val_dataset

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Compute config hash for reproducibility
    config_hash = hash_utils.hash_config(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    
    # Check if we have already run this experiment
    original_cwd = hydra.utils.get_original_cwd()
    base_output_dir = os.path.join(original_cwd, "outputs")
    existing_dir = hash_utils.find_matching_output_dir(cfg, base_dir=base_output_dir)
    
    if existing_dir is not None:
        logger.info(f"Found existing results for this configuration: {existing_dir}")
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        
    # Initialize W&B
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode
    )
    
    # Log config hash to W&B
    if wandb.run is not None:
        wandb.run.summary["config_hash"] = config_hash
    
    try:
        # Create the dataset
        full_dataset = hydra.utils.instantiate(cfg.dataset)

        # Create train-validation split
        validation_split = getattr(cfg.training, 'validation_split', 0.2)
        shuffle_before_split = getattr(cfg.training, 'shuffle_before_split', True)
        
        if validation_split > 0.0:
            train_dataset, val_dataset = create_train_val_split(
                full_dataset, 
                validation_split=validation_split,
                shuffle_before_split=shuffle_before_split,
                seed=cfg.seed
            )
            logger.info(f"Created train-validation split: {len(train_dataset)} train, {len(val_dataset)} validation")
        else:
            # If no validation split requested, use full dataset for training
            train_dataset = full_dataset
            val_dataset = None
            logger.info(f"No validation split requested. Using full dataset ({len(full_dataset)} samples) for training.")

        # Create collate function
        collate_fn = hydra.utils.instantiate(cfg.collate)
        
        # For collate objects (SetMixer, PairedCollate), use their collate_fn method
        if hasattr(collate_fn, 'collate_fn'):
            collate_fn = collate_fn.collate_fn

        # Improved DataLoader with parallel workers and pinned memory
        num_workers = min(4, os.cpu_count())  # Use at most 8 workers or available CPU cores
        
        # Base dataloader kwargs
        base_dataloader_kwargs = {
            'batch_size': cfg.experiment.batch_size,
            'prefetch_factor': 2,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False,
            'collate_fn': None
        }
        
        # Check if custom sampling is configured
        train_dataloader_kwargs = base_dataloader_kwargs.copy()
        train_dataloader_kwargs['dataset'] = train_dataset
        
        # Check for sampling config in both global and experiment namespaces
        sampling_config = None
        if hasattr(cfg, 'sampling') and hasattr(cfg.sampling, 'mode') and cfg.sampling.mode is not None:
            sampling_config = cfg.sampling

            
        if sampling_config is not None:
            logger.info(f"Using custom sampling with mode: {sampling_config.mode}")
            try:
                from utils.custom_sampler import CustomWeightedSampler
                # TODO: do we really want to have replacement=True?
                train_sampler = CustomWeightedSampler(
                    dataset=train_dataset,
                    sampling_mode=sampling_config.mode,
                    num_samples=getattr(sampling_config, 'num_samples', None),
                    replacement=getattr(sampling_config, 'replacement', True),
                    const_weight=getattr(sampling_config, 'const_weight', 1.0)
                )
                # Log weight statistics
                stats = train_sampler.get_weight_statistics()
                logger.info(f"Train sampler weight statistics: {stats}")
                
                # Use custom sampler (shuffle must not be specified when using a custom sampler)
                train_dataloader_kwargs['sampler'] = train_sampler
                
            except Exception as e:
                logger.error(f"Failed to create custom sampler: {e}")
                logger.info("Falling back to default sampling (shuffle=True)")
                # Use default random shuffling
                train_dataloader_kwargs['shuffle'] = True
        else:
            logger.info("Using default sampling (shuffle=True)")
            # Use default random shuffling
            train_dataloader_kwargs['shuffle'] = True
        
        # Create train dataloader
        train_dataloader = DataLoader(**train_dataloader_kwargs)
        
        # Create validation dataloader (if validation dataset exists)
        val_dataloader = None
        if val_dataset is not None:
            val_dataloader_kwargs = base_dataloader_kwargs.copy()
            val_dataloader_kwargs['dataset'] = val_dataset
            val_dataloader_kwargs['shuffle'] = False  # Don't shuffle validation data
            val_dataloader = DataLoader(**val_dataloader_kwargs)
            logger.info(f"Created validation dataloader with {len(val_dataset)} samples")
        
        # DEBUG: Inspect batch structure
        logger.info("=== BATCH STRUCTURE DEBUG ===")
        try:
            sample_batch = next(iter(train_dataloader))
            logger.info(f"Batch type: {type(sample_batch)}")
            
            if isinstance(sample_batch, dict):
                logger.info("Batch is a dictionary with keys:")
                logger.info(f"Actual contents: {sample_batch}")
                for key, value in sample_batch.items():
                    if torch.is_tensor(value):
                        logger.info(f"  {key}: tensor of shape {value.shape}, dtype {value.dtype}")
                    elif isinstance(value, (list, tuple)):
                        logger.info(f"  {key}: {type(value).__name__} of length {len(value)}")
                        if len(value) > 0:
                            logger.info(f"    First element type: {type(value[0])}")
                            if torch.is_tensor(value[0]):
                                logger.info(f"    First element shape: {value[0].shape}")
                    else:
                        logger.info(f"  {key}: {type(value)} - {value}")
                        
            elif isinstance(sample_batch, (list, tuple)):
                logger.info(f"Batch is a {type(sample_batch).__name__} of length {len(sample_batch)}")
                for i, item in enumerate(sample_batch):
                    if torch.is_tensor(item):
                        logger.info(f"  Item {i}: tensor of shape {item.shape}, dtype {item.dtype}")
                    else:
                        logger.info(f"  Item {i}: {type(item)} - {str(item)[:100]}...")
                        
            elif torch.is_tensor(sample_batch):
                logger.info(f"Batch is a tensor of shape {sample_batch.shape}, dtype {sample_batch.dtype}")
                
            else:
                logger.info(f"Batch is of type {type(sample_batch)}: {str(sample_batch)[:200]}...")
                
        except Exception as e:
            logger.error(f"Error inspecting batch structure: {e}")
        logger.info("=== END BATCH STRUCTURE DEBUG ===")
        
 
        
        
        # Create encoder
        encoder = hydra.utils.instantiate(cfg.encoder)

        if hasattr(cfg, "predictor"):
            predictor = hydra.utils.instantiate(cfg.predictor)
            # SELU by default but adding this to make sure it's the same as the encoder
            predictor.latent_act = encoder.latent_act
            encoder.predictor = predictor

        # Create generator (with model already instantiated)
        generator = hydra.utils.instantiate(cfg.generator)
        
        # Get model parameters
        model_parameters = list(encoder.parameters()) + list(generator.parameters())
        
        # Create optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)

        loss_manager = hydra.utils.instantiate(cfg.loss)

        # Create trainer
        trainer = hydra.utils.instantiate(cfg.training)
        
        # Run training with the hash-based output directory
        output_dir, stats = trainer.train(
            encoder=encoder,
            generator=generator,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_manager=loss_manager,
            output_dir=base_output_dir,
            config=cfg,
        )
        
        logger.info(f"Training completed. Best epoch: {stats['best_epoch']}")
        
        # Train latent mapping model after main training is complete (if enabled)
        if hasattr(cfg, "predictor_training"):
            logger.info("Starting latent mapping training...")

            mapping_trainer = hydra.utils.instantiate(cfg.predictor_training, model=encoder.predictor)
            
            # Create output directory for latent mapping
            latent_mapping_output_dir = os.path.join(output_dir, "latent_mapping")
            
            # Train the latent mapping model using the trained encoder
            final_mapping_model_path = mapping_trainer.train(
                encoder=encoder,
                dataloader=train_dataloader,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                output_dir=latent_mapping_output_dir
            )
            
            # Evaluate the mapping model (use validation dataloader if available)
            eval_dataloader = val_dataloader if val_dataloader is not None else train_dataloader
            eval_loss = mapping_trainer.evaluate(encoder, eval_dataloader)
            logger.info(f"Latent mapping model evaluation loss: {eval_loss:.6f}")
            
            # Log latent mapping results to W&B
            if wandb.run is not None:
                wandb.run.summary["latent_mapping_final_loss"] = eval_loss
                wandb.run.summary["latent_mapping_model_path"] = final_mapping_model_path
            
            logger.info(f"Latent mapping training completed. Model saved to: {final_mapping_model_path}")
        else:
            logger.info("Latent mapping training is disabled. Skipping...")
                    
    
    finally:
        # Make sure to finish the W&B run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 