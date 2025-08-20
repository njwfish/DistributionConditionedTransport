import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import logging
import wandb
import os
import numpy as np
import time

import torch.nn as nn

from utils.custom_sampler import CustomWeightedSampler
# Import our resolver for sum operations
import utils.hash_utils as hash_utils
from latent_mapping_training import LatentMappingTrainer




@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Create file handler for debug logging in base directory
    original_cwd = hydra.utils.get_original_cwd()
    
    
    start_time = time.time()

    # Compute config hash for reproducibility
    config_hash = hash_utils.hash_config(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    
    # Check if we have already run this experiment
    check_start = time.time()
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


        # Improved DataLoader with parallel workers and pinned memory
        dataloader_start = time.time()
        num_workers = min(2, os.cpu_count())  # Reduced from 4 to 2 to avoid DataLoader warnings and reduce memory contention
        
        # Base dataloader kwargs
        base_dataloader_kwargs = {
            'batch_size': cfg.experiment.batch_size,
            'prefetch_factor': 2,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False,
            'collate_fn': None
        }
        
        sampling_config = cfg.sampling

        logger.info(f"Using custom sampling with mode: {sampling_config.mode}")

        sampler = CustomWeightedSampler(
            dataset=dataset,
            sampling_mode=sampling_config.mode,
            num_samples=getattr(sampling_config, 'num_samples', None),
            replacement=getattr(sampling_config, 'replacement', False),
            const_weight=getattr(sampling_config, 'const_weight', 1.0),
            cfg=cfg,
        )

        dataloader = DataLoader(dataset, **base_dataloader_kwargs, sampler=sampler)
        

        # Create encoder
        encoder = hydra.utils.instantiate(cfg.encoder)

        # TODO: it would probably be good to re-implement the option to train the predictor after having trained everything else.
        predictor_start = time.time()
        if hasattr(cfg, "predictor"):
            predictor = hydra.utils.instantiate(cfg.predictor)
            if hasattr(encoder, "latent_act"):
                # SELU by default but adding this to make sure it's the same as the encoder
                predictor.latent_act = encoder.latent_act
            else:
                predictor.latent_act = nn.SELU()
            encoder.predictor = predictor

        generator = hydra.utils.instantiate(cfg.generator)
        
        # Get model parameters
        model_parameters = list(encoder.parameters()) + list(generator.parameters())
        
        # Create optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)

        loss_manager = hydra.utils.instantiate(cfg.loss)

        # Create trainer
        trainer = hydra.utils.instantiate(cfg.training)
        
        # GPU Transfer Check
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_start = time.time()
        encoder = encoder.to(device)
        generator = generator.to(device)
        
        # Run training with the hash-based output directory
        train_start = time.time()
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
        
                    
    
    finally:
        # Make sure to finish the W&B run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 