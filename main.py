import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset
import logging
import wandb
import os
import numpy as np
import time

import torch.nn as nn


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
    
    # Set up detailed debug logging to separate file
    debug_logger = logging.getLogger('debug_performance')
    debug_logger.setLevel(logging.DEBUG)
    
    # Create file handler for debug logging in base directory
    original_cwd = hydra.utils.get_original_cwd()
    debug_log_path = f'/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/debug_logging_seed{cfg.seed}.log'
    debug_handler = logging.FileHandler(debug_log_path, mode='w')  # Overwrite each run
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    debug_handler.setFormatter(debug_formatter)
    debug_logger.addHandler(debug_handler)
    
    # Also add console handler for debug logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(debug_formatter)
    debug_logger.addHandler(console_handler)
    
    start_time = time.time()
    debug_logger.info("=== MAIN FUNCTION STARTED ===")
    debug_logger.info(f"CUDA available: {torch.cuda.is_available()}")
    debug_logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        debug_logger.info(f"Current GPU: {torch.cuda.current_device()}")
        debug_logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Compute config hash for reproducibility
    debug_logger.info("Computing config hash...")
    config_hash = hash_utils.hash_config(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    debug_logger.info(f"Config hash computed: {config_hash} (took {time.time() - start_time:.2f}s)")
    
    # Check if we have already run this experiment
    debug_logger.info("Checking for existing experiments...")
    check_start = time.time()
    base_output_dir = os.path.join(original_cwd, "outputs")
    existing_dir = hash_utils.find_matching_output_dir(cfg, base_dir=base_output_dir)
    debug_logger.info(f"Existing experiment check completed (took {time.time() - check_start:.2f}s)")
    
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
        debug_logger.info("Creating dataset...")
        dataset_start = time.time()
        full_dataset = hydra.utils.instantiate(cfg.dataset)
        debug_logger.info(f"Dataset created with {len(full_dataset)} samples (took {time.time() - dataset_start:.2f}s)")

        # Create train-validation split
        debug_logger.info("Creating train-validation split...")
        split_start = time.time()
        validation_split = getattr(cfg, 'validation_split', 0.2)
        shuffle_before_split = getattr(cfg, 'shuffle_before_split', True)
        
        # TODO: are you being to hard on the model by doing the validation split this way? Maybe it is enough to just use the randomness of the subsamples in the dataloader rather than leaving out full time-point pairs?
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
        
        debug_logger.info(f"Train-validation split completed (took {time.time() - split_start:.2f}s)")

        # Improved DataLoader with parallel workers and pinned memory
        debug_logger.info("Setting up DataLoaders...")
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
                # Resolve optional time index file and time_scale
                time_index_path = getattr(cfg.experiment, 'time_index_path', None)
                time_scale = getattr(cfg.experiment, 'num_time_points', 1.0)


                train_sampler = CustomWeightedSampler(
                    dataset=train_dataset,
                    sampling_mode=sampling_config.mode,
                    num_samples=getattr(cfg.experiment, 'num_samples', None),
                    replacement=getattr(sampling_config, 'replacement', False),
                    const_weight=getattr(sampling_config, 'const_weight', 1.0),
                    time_index_path=time_index_path,
                    time_scale=time_scale,
                    cfg=cfg,
                )
                # Log weight statistics
                stats = train_sampler.get_weight_statistics()
                logger.info(f"Train sampler weight statistics: {stats}")
                
                # Debug print: Show non-zero weight information
                print(f"[DEBUG] Dataset sampling info:")
                print(f"[DEBUG] - Total dataset samples: {stats['total_samples']}")
                print(f"[DEBUG] - Samples with non-zero weights: {stats['num_nonzero']}")
                print(f"[DEBUG] - Percentage with non-zero weights: {stats['num_nonzero']/stats['total_samples']*100:.2f}%")
                print(f"[DEBUG] - Sampling mode: {stats['sampling_mode']}")
                
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
        
        debug_logger.info(f"DataLoaders created (took {time.time() - dataloader_start:.2f}s)")
        
        # Create encoder
        debug_logger.info("Creating encoder...")
        encoder_start = time.time()
        encoder = hydra.utils.instantiate(cfg.encoder)
        debug_logger.info(f"Encoder created (took {time.time() - encoder_start:.2f}s)")

        # TODO: make sure the predictor really has both options of being used during training or only trained during training. 
        # TODO: it would probably be good to re-implement the option to train the predictor after having trained everything else.
        debug_logger.info("Creating predictor...")
        predictor_start = time.time()
        if hasattr(cfg, "predictor"):
            predictor = hydra.utils.instantiate(cfg.predictor)
            if hasattr(encoder, "latent_act"):
                # SELU by default but adding this to make sure it's the same as the encoder
                predictor.latent_act = encoder.latent_act
            else:
                predictor.latent_act = nn.SELU()
            encoder.predictor = predictor
        debug_logger.info(f"Predictor created (took {time.time() - predictor_start:.2f}s)")

        # Create generator (with model already instantiated)
        debug_logger.info("Creating generator...")
        generator_start = time.time()
        generator = hydra.utils.instantiate(cfg.generator)
        debug_logger.info(f"Generator created (took {time.time() - generator_start:.2f}s)")
        
        # Get model parameters
        debug_logger.info("Setting up training components...")
        setup_start = time.time()
        model_parameters = list(encoder.parameters()) + list(generator.parameters())
        debug_logger.info(f"Total model parameters: {sum(p.numel() for p in model_parameters):,}")
        
        # Create optimizer and scheduler
        optimizer = hydra.utils.instantiate(cfg.optimizer)(params=model_parameters)
        scheduler = hydra.utils.instantiate(cfg.scheduler)(optimizer=optimizer)

        loss_manager = hydra.utils.instantiate(cfg.loss)

        # Create trainer
        trainer = hydra.utils.instantiate(cfg.training)
        debug_logger.info(f"Training components created (took {time.time() - setup_start:.2f}s)")
        
        # GPU Transfer Check
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        debug_logger.info(f"Moving models to device: {device}")
        gpu_start = time.time()
        encoder = encoder.to(device)
        generator = generator.to(device)
        debug_logger.info(f"Models moved to {device} (took {time.time() - gpu_start:.2f}s)")
        
        # Run training with the hash-based output directory
        debug_logger.info("Starting training...")
        train_start = time.time()
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
        
        debug_logger.info(f"Training completed (took {time.time() - train_start:.2f}s)")
        debug_logger.info(f"Total main function time: {time.time() - start_time:.2f}s")
        logger.info(f"Training completed. Best epoch: {stats['best_epoch']}")
        
                    
    
    finally:
        # Make sure to finish the W&B run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main() 