import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import logging
import wandb
from loss.default import LossManager as DefaultLossManager
from predictor_training import PredictorTrainer
import os
import numpy as np
import time

import torch.nn as nn
# Import our resolver for sum operations
import utils.hash_utils as hash_utils
# Import debug memory logger
from utils.debug_memory_logger import get_debug_logger



@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    # Initialize GPU memory debug logger (gated by config)
    debug_logger = None
    if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "debug_memory_logging") and bool(cfg.experiment.debug_memory_logging):
        debug_logger = get_debug_logger()
        debug_logger.log_cuda_info()
        debug_logger.log_memory("START", "Main function started")
    
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
        if debug_logger is not None:
            debug_logger.log_memory("DATASET_START", "Creating dataset")
        dataset = hydra.utils.instantiate(cfg.dataset)
        if debug_logger is not None:
            debug_logger.log_memory("DATASET_LOADED", f"Dataset created with {len(dataset)} samples")


        # Improved DataLoader with parallel workers and pinned memory
        dataloader_start = time.time()
        num_workers = min(2, os.cpu_count())  # Reduced from 4 to 2 to avoid DataLoader warnings and reduce memory contention

        if hasattr(dataset, 'pairwise_distance'):
            coupling_kwargs = {'pairwise_dist_fn': dataset.pairwise_distance}
        else:
            coupling_kwargs = {}
        coupling = hydra.utils.instantiate(cfg.coupling, **coupling_kwargs)
        
        # Base dataloader kwargs
        base_dataloader_kwargs = {
            'batch_size': cfg.experiment.batch_size,
            'prefetch_factor': 2,
            'num_workers': num_workers,
            'pin_memory': True,
            'persistent_workers': True if num_workers > 0 else False,
            'collate_fn': coupling
        }
        
        sampling_config = cfg.sampling

        # Load optional lists for the sampler if specified in cfg.experiment
        sampler_kwargs = {}
        
        # Check for specific_pairing_pth
        if hasattr(cfg.experiment, 'specific_pairing_pth'):
            specific_pairing_path = cfg.experiment.specific_pairing_pth

            data = np.load(specific_pairing_path)
            specific_pairing = data['specific_pairing'].tolist()
            sampler_kwargs['specific_pairing'] = specific_pairing
            logger.info(f"Loaded specific_pairing with {len(specific_pairing)} pairs")

        # Check for precomputed_d_values_pth
        if hasattr(cfg.experiment, 'precomputed_d_values_pth'):
            precomputed_d_values_path = cfg.experiment.precomputed_d_values_pth

            data = np.load(precomputed_d_values_path)
            precomputed_d_values = data['precomputed_d_values']
            sampler_kwargs['precomputed_d_values'] = precomputed_d_values
            logger.info(f"Loaded precomputed_d_values with shape {precomputed_d_values.shape}")

        sampler = hydra.utils.instantiate(sampling_config, dataset=dataset, **sampler_kwargs)

        if debug_logger is not None:
            debug_logger.log_memory("DATALOADER_START", "Creating DataLoader")
        dataloader = DataLoader(
            dataset,
            **base_dataloader_kwargs,
            sampler=sampler,
            shuffle=False if sampler is not None else True,
        )
        if debug_logger is not None:
            debug_logger.log_memory("DATALOADER_CREATED", f"DataLoader created with batch_size={cfg.experiment.batch_size}")
        
        # Create encoder
        if debug_logger is not None:
            debug_logger.log_memory("ENCODER_START", "Creating encoder")
        encoder = hydra.utils.instantiate(cfg.encoder)

        if debug_logger is not None:
            debug_logger.log_model_info("ENCODER_CREATED", "encoder", encoder)
        
        train_predictor_posthoc = False
        if hasattr(cfg, "experiment") and hasattr(cfg.experiment, "train_predictor_posthoc"):
            train_predictor_posthoc = bool(cfg.experiment.train_predictor_posthoc)

        # Instantiate predictor independently (no longer part of encoder)
        predictor = None
        if hasattr(cfg, "predictor") and cfg.predictor is not None:

            if debug_logger is not None:
                debug_logger.log_memory("PREDICTOR_START", "Creating predictor")

            predictor = hydra.utils.instantiate(cfg.predictor)
            if hasattr(encoder, "latent_act"):
                predictor.latent_act = encoder.latent_act
            else:
                predictor.latent_act = nn.SELU()

            if debug_logger is not None:
                debug_logger.log_model_info("PREDICTOR_CREATED", "predictor", predictor)


        if debug_logger is not None:
            debug_logger.log_memory("GENERATOR_START", "Creating generator")
        generator = hydra.utils.instantiate(cfg.generator)
        if debug_logger is not None:
            debug_logger.log_model_info("GENERATOR_CREATED", "generator", generator)
        
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

        if train_predictor_posthoc and cfg.loss._target_ != "loss.default.LossManager":
            raise ValueError("Cannot train predictor posthoc with non-default loss")
        
        loss_manager = hydra.utils.instantiate(cfg.loss)


        # Create trainer
        trainer = hydra.utils.instantiate(cfg.training)
        
        # GPU Transfer Check
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        gpu_start = time.time()
        if debug_logger is not None:
            debug_logger.log_memory("GPU_TRANSFER_START", "Moving models to GPU")
        encoder = encoder.to(device)
        if debug_logger is not None:
            debug_logger.log_memory("ENCODER_GPU", f"Encoder moved to {device}")
        generator = generator.to(device)
        if debug_logger is not None:
            debug_logger.log_memory("GENERATOR_GPU", f"Generator moved to {device}")
            debug_logger.reset_peak_memory()
        
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