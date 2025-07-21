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
        # NOTE: settin shuffle=False for now such that our model just pairs nearest neighbors (note that even if we do larger steps, we still want the pairing to go in the correct direction)
        dataloader = DataLoader(
            dataset, 
            batch_size=cfg.experiment.batch_size, 
            shuffle=False,
            prefetch_factor=2,
            num_workers=num_workers,  # Parallel data loading
            pin_memory=True,  # Pin memory for faster data transfer to GPU
            persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between iterations
            collate_fn=collate_fn
        )
        
        
        # Create encoder
        encoder = hydra.utils.instantiate(cfg.encoder)
        
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
        latent_mapping_config = cfg.get('latent_mapping', {})
        if latent_mapping_config.get('enabled', False):
            logger.info("Starting latent mapping training...")
            
            # Create latent mapping trainer with configurable parameters
            mapping_trainer = LatentMappingTrainer(
                mapping_method=latent_mapping_config.get('mapping_method', 'neural_network'),
                hidden_dim=latent_mapping_config.get('hidden_dim', 128),
                ridge_alpha=latent_mapping_config.get('ridge_alpha', 1e-3),
                num_epochs=latent_mapping_config.get('num_epochs', 100),
                learning_rate=latent_mapping_config.get('learning_rate', 1e-3),
                batch_size=latent_mapping_config.get('batch_size', 32),
                log_interval=latent_mapping_config.get('log_interval', 10),
                save_interval=latent_mapping_config.get('save_interval', 20),
                use_tqdm=latent_mapping_config.get('use_tqdm', True)
            )
            
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