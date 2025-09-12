#!/usr/bin/env python3
"""
Dataset Visualization Script

This script instantiates the Letters dataset using Hydra configuration
(just like main.py) and creates visualizations showing source and target
samples side by side for several dataset elements.

Usage:
    python visualize_dataset.py --ckpt_dir /abs/path/to/experiment_dir
"""

import hydra
import argparse
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import os

# Import our resolver for sum operations (same as main.py)
import utils.hash_utils as hash_utils

# Parsed from command line before Hydra runs
CLI_CKPT_SUBDIR = None

def main(ckpt_dir: str):
    logger = logging.getLogger(__name__)
    logger.info("Dataset Visualization Script")
    
    # Resolve and validate checkpoint directory
    experiment_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
    cfg_path = os.path.join(experiment_dir, 'config.yaml')
    ckpt_path = os.path.join(experiment_dir, 'best_model.pt')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"No config.yaml found at {cfg_path}")
    if not os.path.exists(ckpt_path):
        logger.warning(f"No best_model.pt found at {ckpt_path}. Will skip generator visualization.")
    
    # Load trained config and use it as the active config
    trained_cfg = OmegaConf.load(cfg_path)
    logger.info("Loaded trained config:\n" + OmegaConf.to_yaml(trained_cfg))
    
    # Set random seed for reproducibility
    if OmegaConf.select(trained_cfg, 'seed') is not None:
        torch.manual_seed(trained_cfg.seed)
        np.random.seed(trained_cfg.seed)
        
    # Create the dataset (same as main.py line 60)
    logger.info("Instantiating dataset...")
    dataset = hydra.utils.instantiate(trained_cfg.dataset)
    
    logger.info(f"Dataset created with {len(dataset)} elements")
    logger.info(f"Dataset parameters:")
    logger.info(f"  - Number of fonts: {dataset.num_fonts}")
    logger.info(f"  - Training letters: {dataset.train_letters}")
    logger.info(f"  - Set size: {dataset.set_size}")
    logger.info(f"  - Samples per letter-font: {dataset.samples_per_letter_font}")
    
    # Optionally load a trained encoder+generator to visualize generated samples
    trained_models_available = False
    encoder = None
    generator = None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Always try to use the provided ckpt_dir to load models
    try:
        logger.info("Instantiating encoder and generator from trained config...")
        encoder = hydra.utils.instantiate(trained_cfg.encoder).to(device)
        generator = hydra.utils.instantiate(trained_cfg.generator).to(device)
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'encoder_state_dict' not in checkpoint or 'generator_state_dict' not in checkpoint:
                logger.warning("Checkpoint missing encoder/generator weights. Skipping generator visualization.")
            else:
                encoder.load_state_dict(checkpoint['encoder_state_dict'])
                generator.load_state_dict(checkpoint['generator_state_dict'])
                encoder.eval()
                generator.eval()
                trained_models_available = True
        else:
            logger.warning(f"Checkpoint file not found at {ckpt_path}. Skipping generator visualization.")
    except Exception as e:
        logger.exception(f"Failed to load trained models: {e}")

    # Create visualization for several dataset elements
    num_examples = min(6, len(dataset))  # Show up to 6 examples

    # Create figure with subplots (add third row if trained models are available)
    num_rows = 3 if trained_models_available else 2
    fig_height = 12 if trained_models_available else 8
    fig, axes = plt.subplots(num_rows, num_examples, figsize=(4*num_examples, fig_height))
    if num_examples == 1:
        axes = axes.reshape(num_rows, 1)
    
    logger.info(f"Creating visualizations for {num_examples} examples...")
    
    for i in range(num_examples):
        # Get dataset item
        item = dataset[i]
        source_samples = item['source_samples']
        target_samples = item['target_samples']
        
        # Determine which letter and fonts we're looking at
        letter = dataset.train_letters[i // (dataset.num_fonts - 1)]
        font = i % (dataset.num_fonts - 1)
        
        logger.info(f"Example {i+1}: Letter '{letter}', Font {font} -> Font {font+1}")
        logger.info(f"  Source samples shape: {source_samples.shape}")
        logger.info(f"  Target samples shape: {target_samples.shape}")
        
        # Plot source samples
        axes[0, i].scatter(source_samples[:, 0], source_samples[:, 1], 
                          alpha=0.6, s=1, c='blue')
        axes[0, i].set_title(f"Source: Letter '{letter}' (Font {font})")
        axes[0, i].set_xlim(0, 1)
        axes[0, i].set_ylim(0, 1)
        axes[0, i].set_aspect('equal')
        axes[0, i].grid(True, alpha=0.3)
        axes[0, i].invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Plot target samples  
        axes[1, i].scatter(target_samples[:, 0], target_samples[:, 1], 
                          alpha=0.6, s=1, c='red')
        axes[1, i].set_title(f"Target: Letter '{letter}' (Font {font+1})")
        axes[1, i].set_xlim(0, 1)
        axes[1, i].set_ylim(0, 1)
        axes[1, i].set_aspect('equal')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].invert_yaxis()  # Invert y-axis to match image coordinates
        
        # Plot generated samples using trained encoder+generator if available
        if trained_models_available:
            try:
                with torch.no_grad():
                    # Prepare tensors
                    src_tensor = torch.from_numpy(source_samples).to(device)
                    tgt_tensor = torch.from_numpy(target_samples).to(device)
                    # Encode to latents (batch with single set)
                    src_latent = encoder(src_tensor.unsqueeze(0))  # [1, latent_dim]
                    tgt_latent = encoder(tgt_tensor.unsqueeze(0))  # [1, latent_dim]
                    # Generate from source samples conditioned on src/target latents
                    gen = generator.sample(src_tensor, src_latent, tgt_latent)  # [1, set_size, 2]
                    gen_np = gen.squeeze(0).detach().cpu().numpy()
                axes[2, i].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.6, s=1, c='green')
                axes[2, i].set_title(f"Generated from '{letter}' (Font {font}→{font+1})")
                axes[2, i].set_xlim(0, 1)
                axes[2, i].set_ylim(0, 1)
                axes[2, i].set_aspect('equal')
                axes[2, i].grid(True, alpha=0.3)
                axes[2, i].invert_yaxis()
            except Exception as e:
                logger.exception(f"Generation failed for example {i+1}: {e}")
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "/orcd/archive/abugoot/001/Projects/paolo/paper_branch_tde/dataset_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Visualization saved to: {output_path}")
    
    # Also show additional statistics
    logger.info("\nDataset Statistics:")
    logger.info(f"Total dataset size: {len(dataset)} pairs")
    
    # Sample a few more items to show statistics
    sample_source = []
    sample_target = []
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        sample_source.append(item['source_samples'])
        sample_target.append(item['target_samples'])
    
    sample_source = np.concatenate(sample_source, axis=0)
    sample_target = np.concatenate(sample_target, axis=0)
    
    logger.info(f"Sample statistics from first 10 pairs:")
    logger.info(f"  Source samples - mean: {np.mean(sample_source, axis=0)}, std: {np.std(sample_source, axis=0)}")
    logger.info(f"  Target samples - mean: {np.mean(sample_target, axis=0)}, std: {np.std(sample_target, axis=0)}")
    
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Visualization", add_help=True)
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Absolute path to experiment directory containing config.yaml and best_model.pt")
    args, _unknown = parser.parse_known_args()
    main(args.ckpt_dir)
