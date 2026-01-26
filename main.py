import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import logging
import wandb
from loss.default import LossManager as DefaultLossManager
import os
import numpy as np
import time

import torch.nn as nn
# Import our resolver for sum operations
import utils.hash_utils as hash_utils
from utils.seed import seed_everything, dataloader_seed_worker, make_torch_generator



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
    
    # Set random seed and deterministic behavior
    if cfg.seed is not None:
        seed_everything(int(cfg.seed), deterministic=True)
        
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
        num_workers = min(8, os.cpu_count())  # Reduced from 4 to 2 to avoid DataLoader warnings and reduce memory contention

        if hasattr(dataset, 'pairwise_distance'):
            coupling_kwargs = {'pairwise_dist_fn': dataset.pairwise_distance}
        else:
            coupling_kwargs = {}

        # Handle coupling configuration with robust error handling
        coupling = None
        if hasattr(cfg, 'coupling') and cfg.coupling is not None:
            coupling_target = cfg.coupling.get('_target_', None)
            if coupling_target is not None and coupling_target != 'types.NoneType':
                try:
                    coupling = hydra.utils.instantiate(cfg.coupling, **coupling_kwargs)
                except Exception as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Failed to instantiate coupling {coupling_target}: {e}")
                    coupling = None

        # Base dataloader kwargs
        base_dataloader_kwargs = {
            'batch_size': cfg.experiment.batch_size,
            'prefetch_factor': 2,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False,
            'collate_fn': coupling,
            'worker_init_fn': dataloader_seed_worker,
            'generator': make_torch_generator(int(cfg.seed) if hasattr(cfg, 'seed') else None),
        }
        
        sampling_config = cfg.sampling

        # Load optional lists for the sampler if specified in cfg.experiment
        sampler_kwargs = {}
        print(sampling_config._target_)

        sampler = hydra.utils.instantiate(sampling_config, dataset=dataset, **sampler_kwargs) if sampling_config._target_ != "types.NoneType" else None
        print(sampler)

        dataloader = DataLoader(
            dataset,
            **base_dataloader_kwargs,
            sampler=sampler,
            shuffle=False if sampler is not None else True,
        )
        
        # Create encoder
        encoder = hydra.utils.instantiate(cfg.encoder)
        
        train_predictor_posthoc = False
        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "train_predictor_posthoc"):
            train_predictor_posthoc = bool(cfg.experiment.train_predictor_posthoc)

        # Instantiate predictor independently (no longer part of encoder)
        predictor = None
        if hasattr(cfg, "predictor") and cfg.predictor is not None:
            predictor = hydra.utils.instantiate(cfg.predictor)
            if hasattr(encoder, "latent_act"):
                predictor.latent_act = encoder.latent_act
            else:
                predictor.latent_act = nn.SELU()

        generator = hydra.utils.instantiate(cfg.generator)
        
        # Get model parameters
        if not train_predictor_posthoc and predictor is not None:
            # joint training includes predictor
            model_parameters = list(encoder.parameters()) + list(generator.parameters()) + list(predictor.parameters())
        else:
            # predictor separate: train only encoder+generator here
            model_parameters = list(encoder.parameters()) + list(generator.parameters())
        
        # Create optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)

        if train_predictor_posthoc and cfg.loss._target_ not in ["loss.default.LossManager", "loss.default_source_only.LossManager", "loss.default_source_only_trellis_mfm.LossManager"]:
            raise ValueError("Cannot train predictor posthoc with non-default loss")
        
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
            predictor=predictor if not train_predictor_posthoc else None,
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