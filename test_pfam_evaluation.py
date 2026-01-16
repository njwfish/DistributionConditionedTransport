#!/usr/bin/env python
"""
Evaluate a trained pfam model on a test .pt file.

For each element in the test .pt file:
- Use that element as the target (encode with encoder to get target_latent)
- Use the first element of data/pfam/pfam_tokenized_data.pt as source (encode to get source_latent)
- Compute the generator loss

Results are saved incrementally to allow resuming if interrupted.

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
import json
import torch
import logging
from datetime import datetime
from omegaconf import OmegaConf

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
    
    Args:
        output_dir: Path to the outputs subdirectory containing best_model.pt and config.yaml
        device: torch device to load model to
    
    Returns:
        encoder, generator, config
    """
    import hydra
    
    # Load config
    config_path = os.path.join(output_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Find best model checkpoint
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        # Try to find the latest checkpoint
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        # Sort by epoch number and get the latest
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(output_dir, checkpoints[-1])
        logger.info(f"Using latest checkpoint: {best_model_path}")
    else:
        logger.info(f"Using best model: {best_model_path}")
    
    # Instantiate encoder and generator using hydra, exactly as in main.py
    encoder = hydra.utils.instantiate(config.encoder)
    generator = hydra.utils.instantiate(config.generator)
    
    # Load checkpoint
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
    
    Args:
        encoder: The trained encoder
        generator: The trained generator
        source_samples: Source samples dict (from first element of training data)
                        Contains: esm_input_ids, esm_attention_mask, progen_input_ids, progen_attention_mask
                        Each has shape [num_seqs, seq_len]
        target_samples: Target samples dict (from test data element)
                        Same structure as source_samples
        device: torch device
        
    Returns:
        loss value (float)
    """
    with torch.no_grad():
        # Prepare samples - need to add batch dimension
        # source_samples['esm_input_ids'] has shape [num_seqs, seq_len]
        # We need [batch_size=1, num_seqs, seq_len]
        
        # Build full sample batches with all keys (encoder uses ESM, generator uses what it needs)
        source_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in source_samples.items()
        }
        target_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target_samples.items()
        }
        
        # Encode to get latents (encoder uses esm_input_ids and esm_attention_mask)
        source_latent = encoder(source_batch)  # [1, latent_dim]
        target_latent = encoder(target_batch)  # [1, latent_dim]
        
        # Compute generator loss
        # Generator picks the keys it needs (Progen2 uses progen_*, DFM uses esm_*)
        loss = generator.loss(source_batch, target_batch, source_latent, target_latent)
        
        return loss.item()


def load_results(results_path):
    """Load existing results from file if it exists."""
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            return json.load(f)
    return {}


def save_results(results, results_path):
    """Save results to file."""
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Evaluate pfam model on test .pt file')
    parser.add_argument('--test_pt_file', type=str, required=True,
                        help='Path to the test .pt file created by PfamDataset')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the outputs subdirectory containing best_model.pt and config.yaml')
    parser.add_argument('--source_pt_file', type=str, default='data/pfam/pfam_tokenized_data.pt',
                        help='Path to the source .pt file (default: data/pfam/pfam_tokenized_data.pt)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to save results (default: <output_dir>/eval_results_<test_pt_name>.json)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    
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
    
    # Determine results file path
    if args.results_file is None:
        test_pt_name = os.path.splitext(os.path.basename(args.test_pt_file))[0]
        args.results_file = os.path.join(args.output_dir, f'eval_results_{test_pt_name}.json')
    
    logger.info(f"Results will be saved to: {args.results_file}")
    
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
    
    # Load existing results for resumption
    results = load_results(args.results_file)
    if 'metadata' not in results:
        results['metadata'] = {
            'test_pt_file': args.test_pt_file,
            'source_pt_file': args.source_pt_file,
            'output_dir': args.output_dir,
            'source_pfam': source_pfam,
            'num_test_families': len(test_data),
            'started_at': datetime.now().isoformat(),
            'device': str(device)
        }
    
    if 'losses' not in results:
        results['losses'] = {}
    
    completed = set(results['losses'].keys())
    logger.info(f"Already completed: {len(completed)}/{len(test_data)} families")
    
    # Evaluate each target family
    total_loss = 0.0
    count = 0
    
    for idx, target_element in enumerate(test_data):
        target_pfam = target_element.get('pfam', f'family_{idx}')
        
        # Skip if already computed
        if target_pfam in completed:
            loss = results['losses'][target_pfam]['loss']
            total_loss += loss
            count += 1
            continue
        
        target_samples = target_element['samples']
        num_target_seqs = target_samples['esm_input_ids'].shape[0]
        
        try:
            loss = evaluate_single_target(
                encoder, generator, source_samples, target_samples, device
            )
            
            # Save result
            results['losses'][target_pfam] = {
                'loss': loss,
                'num_sequences': num_target_seqs,
                'idx': idx
            }
            
            total_loss += loss
            count += 1
            
            # Print progress
            avg_loss = total_loss / count
            logger.info(f"[{count}/{len(test_data)}] Family: {target_pfam}, "
                       f"Loss: {loss:.6f}, Running avg: {avg_loss:.6f}, "
                       f"Num seqs: {num_target_seqs}")
            
            # Save intermediate results every family (for safety)
            results['metadata']['last_updated'] = datetime.now().isoformat()
            results['metadata']['completed_count'] = count
            results['summary'] = {
                'average_loss': avg_loss,
                'total_loss': total_loss,
                'num_evaluated': count
            }
            save_results(results, args.results_file)
            
        except Exception as e:
            logger.error(f"Error evaluating family {target_pfam}: {e}")
            results['losses'][target_pfam] = {
                'error': str(e),
                'idx': idx
            }
            save_results(results, args.results_file)
            continue
    
    # Final summary
    if count > 0:
        final_avg_loss = total_loss / count
        results['summary'] = {
            'average_loss': final_avg_loss,
            'total_loss': total_loss,
            'num_evaluated': count,
            'completed': True
        }
        results['metadata']['completed_at'] = datetime.now().isoformat()
        save_results(results, args.results_file)
        
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Total families evaluated: {count}")
        logger.info(f"Average loss: {final_avg_loss:.6f}")
        logger.info(f"Results saved to: {args.results_file}")
        logger.info("=" * 60)
        
        # Print per-family losses sorted by loss value
        logger.info("\nPer-family losses (sorted by loss):")
        family_losses = [(k, v['loss']) for k, v in results['losses'].items() 
                        if 'loss' in v]
        family_losses.sort(key=lambda x: x[1])
        for pfam, loss in family_losses:
            logger.info(f"  {pfam}: {loss:.6f}")
    else:
        logger.warning("No families were successfully evaluated!")


if __name__ == '__main__':
    main()
