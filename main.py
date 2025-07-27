import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import logging
import wandb
import os

# Import our resolver for sum operations
import utils.hash_utils as hash_utils
from latent_mapping_training import LatentMappingTrainer

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
        dataset = hydra.utils.instantiate(cfg.dataset)

        # Create collate function
        collate_fn = hydra.utils.instantiate(cfg.collate)
        
        # For collate objects (SetMixer, PairedCollate), use their collate_fn method
        if hasattr(collate_fn, 'collate_fn'):
            collate_fn = collate_fn.collate_fn

        # Improved DataLoader with parallel workers and pinned memory
        num_workers = min(4, os.cpu_count())  # Use at most 8 workers or available CPU cores
        # TODO: make sure the collate function was properly removed here.
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.experiment.batch_size, 
            shuffle=False,
            prefetch_factor=2,
            num_workers=num_workers,  # Parallel data loading
            pin_memory=True,  # Pin memory for faster data transfer to GPU
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between iterations
            collate_fn=None
        )
        
        # DEBUG: Inspect batch structure
        logger.info("=== BATCH STRUCTURE DEBUG ===")
        try:
            sample_batch = next(iter(dataloader))
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
            dataloader=dataloader,
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
                dataloader=dataloader,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                output_dir=latent_mapping_output_dir
            )
            
            # Evaluate the mapping model
            eval_loss = mapping_trainer.evaluate(encoder, dataloader)
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