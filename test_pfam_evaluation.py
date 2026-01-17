#!/usr/bin/env python
"""
Evaluate a trained pfam model on a test .pt file.

For each element in the test .pt file:
- Use that element as the target (encode with encoder to get target_latent)
- Use the first element of source_pt_file as source (encode to get source_latent)
- Compute the generator loss

Usage:
    python test_pfam_evaluation.py --test_pt_file <path_to_test.pt> --output_dir <path_to_outputs_subdir>
    
Example:
    python test_pfam_evaluation.py \
        --test_pt_file data/pfam/pfam_test.pt \
        --output_dir outputs/pfam_28ad2d74a123e25b88f8d360d7d47170
"""

import argparse
import os
import sys
import numpy as np
import torch
import logging
from omegaconf import OmegaConf
from tqdm import tqdm

# Add the project root to path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(output_dir, device):
    """
    Load the encoder and generator from the best model checkpoint in the given output directory.
    """
    import hydra
    
    config_path = os.path.join(output_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(output_dir, checkpoints[-1])
        logger.info(f"Using latest checkpoint: {best_model_path}")
    else:
        logger.info(f"Using best model: {best_model_path}")
    
    encoder = hydra.utils.instantiate(config.encoder)
    generator = hydra.utils.instantiate(config.generator)
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', float('nan'))
    logger.info(f"Loaded model from epoch {epoch} with loss {loss:.6f}")
    
    encoder.to(device)
    generator.to(device)
    encoder.eval()
    generator.eval()
    
    return encoder, generator, config


def evaluate_single_target(encoder, generator, source_samples, target_samples, device):
    """
    Compute the generator loss for a single source-target pair.
    """
    with torch.no_grad():
        # Build full sample batches with all keys (encoder uses ESM, generator uses what it needs)
        source_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in source_samples.items()
        }
        target_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target_samples.items()
        }
        
        # Encode to get latents
        source_latent = encoder(source_batch)
        target_latent = encoder(target_batch)
        
        # Compute generator loss
        loss = generator.loss(source_batch, target_batch, source_latent, target_latent)
        
        return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Evaluate pfam model on test .pt file')
    parser.add_argument('--test_pt_file', type=str, required=True,
                        help='Path to the test .pt file created by PfamDataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the outputs subdirectory containing best_model.pt and config.yaml')
    parser.add_argument('--source_pt_file', type=str, default='data/pfam/pfam_tokenized_data.pt',
                        help='Path to the source .pt file (default: data/pfam/pfam_tokenized_data.pt)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print per-element results (default: only print summary statistics)')
    parser.add_argument('-n', '--num_samples', type=int, default=None,
                        help='Only evaluate the first N test elements (default: all)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Handle relative paths - make them relative to script directory
    if not os.path.isabs(args.test_pt_file):
        args.test_pt_file = os.path.join(SCRIPT_DIR, args.test_pt_file)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    if not os.path.isabs(args.source_pt_file):
        args.source_pt_file = os.path.join(SCRIPT_DIR, args.source_pt_file)
    
    # Verify paths exist
    if not os.path.exists(args.test_pt_file):
        logger.error(f"Test .pt file not found: {args.test_pt_file}")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)
    if not os.path.exists(args.source_pt_file):
        logger.error(f"Source .pt file not found: {args.source_pt_file}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from: {args.output_dir}")
    encoder, generator, config = load_model_from_checkpoint(args.output_dir, device)
    
    # Load source data (first element of training data)
    logger.info(f"Loading source data from: {args.source_pt_file}")
    source_data = torch.load(args.source_pt_file, weights_only=False)
    source_element = source_data[0]
    source_samples = source_element['samples']
    source_pfam = source_element.get('pfam', 'unknown')
    logger.info(f"Source family: {source_pfam}, num sequences: {source_samples['esm_input_ids'].shape[0]}")
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_pt_file}")
    test_data = torch.load(args.test_pt_file, weights_only=False)
    logger.info(f"Test data contains {len(test_data)} families")
    
    # Limit to first N elements if specified
    if args.num_samples is not None:
        test_data = test_data[:args.num_samples]
        logger.info(f"Limiting evaluation to first {len(test_data)} families")
    
    # Evaluate each target family
    all_losses = []
    
    for idx, target_element in enumerate(tqdm(test_data, desc="Evaluating", disable=args.verbose)):
        target_pfam = target_element.get('pfam', f'family_{idx}')
        target_samples = target_element['samples']
        num_target_seqs = target_samples['esm_input_ids'].shape[0]
        
        try:
            loss = evaluate_single_target(
                encoder, generator, source_samples, target_samples, device
            )
            all_losses.append(loss)
            
            if args.verbose:
                logger.info(f"[{idx+1}/{len(test_data)}] Family: {target_pfam}, "
                           f"Loss: {loss:.6f}, Num seqs: {num_target_seqs}")
                print(f"!!!!!!!!!!!!!!!!!!!!!!!!!! [{idx+1}/{len(test_data)}] Family: {target_pfam}, "
                           f"Loss: {loss:.6f}, Num seqs: {num_target_seqs}")
            
        except Exception as e:
            logger.error(f"Error evaluating family {target_pfam}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary statistics
    if len(all_losses) > 0:
        losses_array = np.array(all_losses)
        
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Families evaluated: {len(all_losses)}/{len(test_data)}")
        print("-" * 60)
        print(f"Mean loss:         {losses_array.mean():.6f}")
        print(f"Median loss:       {np.median(losses_array):.6f}")
        print(f"Std loss:          {losses_array.std():.6f}")
        print(f"Min loss:          {losses_array.min():.6f}")
        print(f"Max loss:          {losses_array.max():.6f}")
        print("=" * 60)
    else:
        logger.warning("No families were successfully evaluated!")


if __name__ == '__main__':
    main()
