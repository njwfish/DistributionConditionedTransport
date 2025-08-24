import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import logging
import wandb
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
        debug_logger.log_memory("START", "Main predictor function started")
    
    # Create file handler for debug logging in base directory
    original_cwd = hydra.utils.get_original_cwd()
    
    start_time = time.time()

    # Compute config hash for reproducibility
    config_hash = hash_utils.hash_config(cfg)
    logger.info(f"Configuration hash: {config_hash}")
    
    # Set random seed
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)
        
    # Initialize W&B
    run = wandb.init(
        project=cfg.wandb.project + "_predictor_only",  # Different project name to distinguish
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        mode=cfg.wandb.mode
    )
    
    # Log config hash to W&B
    if wandb.run is not None:
        wandb.run.summary["config_hash"] = config_hash
    
    try:
        # Parse model path from config (added with +model_path=...)
        if not hasattr(cfg, "model_path") or cfg.model_path is None:
            raise ValueError("model_path must be specified via command line: +model_path=/path/to/best_model.pt")
        
        model_path = cfg.model_path
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
            
        logger.info(f"Loading trained model from: {model_path}")
        
        # Create the dataset
        if debug_logger is not None:
            debug_logger.log_memory("DATASET_START", "Creating dataset")
        dataset = hydra.utils.instantiate(cfg.dataset)
        if debug_logger is not None:
            debug_logger.log_memory("DATASET_LOADED", f"Dataset created with {len(dataset)} samples")

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
        
        # Create encoder (same architecture as trained model)
        if debug_logger is not None:
            debug_logger.log_memory("ENCODER_START", "Creating encoder")
        encoder = hydra.utils.instantiate(cfg.encoder)
        if debug_logger is not None:
            debug_logger.log_model_info("ENCODER_CREATED", "encoder", encoder)
        
        # Create generator (same architecture as trained model) 
        if debug_logger is not None:
            debug_logger.log_memory("GENERATOR_START", "Creating generator")
        generator = hydra.utils.instantiate(cfg.generator)
        if debug_logger is not None:
            debug_logger.log_model_info("GENERATOR_CREATED", "generator", generator)

        # Create predictor (fresh, not trained)
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
        else:
            raise ValueError("Predictor configuration is required for predictor-only training")
        
        # GPU Transfer Check
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load trained encoder and generator weights
        logger.info(f"Loading trained weights from {model_path}")
        best_checkpoint = torch.load(model_path, weights_only=False, map_location=device)
        
        if "encoder_state_dict" not in best_checkpoint:
            raise KeyError("'encoder_state_dict' missing in model checkpoint")
        if "generator_state_dict" not in best_checkpoint:
            raise KeyError("'generator_state_dict' missing in model checkpoint")
            
        encoder.load_state_dict(best_checkpoint["encoder_state_dict"])
        generator.load_state_dict(best_checkpoint["generator_state_dict"])
        
        # Move models to device
        if debug_logger is not None:
            debug_logger.log_memory("GPU_TRANSFER_START", "Moving models to GPU")
        encoder = encoder.to(device)
        generator = generator.to(device)
        predictor = predictor.to(device)
        
        if debug_logger is not None:
            debug_logger.log_memory("MODELS_GPU", f"All models moved to {device}")
            debug_logger.reset_peak_memory()
        
        # Set encoder and generator to eval mode and freeze their parameters
        encoder.eval()
        generator.eval()
        for param in encoder.parameters():
            param.requires_grad = False
        for param in generator.parameters():
            param.requires_grad = False
            
        logger.info("Encoder and generator loaded and frozen for predictor training")

        # Create optimizer for predictor only
        pred_optimizer = hydra.utils.instantiate(cfg.optimizer)(params=predictor.parameters())

        # Set up output directory (use model path directory + predictor suffix)
        model_dir = os.path.dirname(model_path)
        pred_output_dir = os.path.join(model_dir, "predictor_training")
        os.makedirs(pred_output_dir, exist_ok=True)

        # Simple predictor trainer using same training config as main training
        predictor_trainer = PredictorTrainer(
            num_epochs=cfg.training.num_epochs,
            log_interval=cfg.training.log_interval,
            save_interval=cfg.training.save_interval,
            eval_interval=cfg.training.eval_interval,
            early_stopping=cfg.training.early_stopping,
            patience=cfg.training.patience,
            use_tqdm=cfg.training.use_tqdm,
        )

        # Run predictor training
        train_start = time.time()
        final_output_dir, pred_stats = predictor_trainer.train(
            encoder=encoder,
            predictor=predictor,
            dataloader=dataloader,
            optimizer=pred_optimizer,
            device=device,
            output_dir=pred_output_dir,
        )
        
        logger.info(
            f"Predictor training completed. Best epoch: {pred_stats.get('best_epoch', 0)}"
        )
        logger.info(f"Predictor training took {time.time() - train_start:.2f} seconds")
        logger.info(f"Predictor training results saved in: {final_output_dir}")
        
        # Log final results to W&B
        if wandb.run is not None:
            wandb.run.summary["predictor_best_epoch"] = pred_stats.get('best_epoch', 0)
            wandb.run.summary["predictor_training_time"] = pred_stats.get('total_time', 0.0)
            wandb.run.summary["loaded_model_path"] = model_path
            wandb.run.summary["predictor_output_dir"] = final_output_dir
    
    finally:
        # Make sure to finish the W&B run
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
