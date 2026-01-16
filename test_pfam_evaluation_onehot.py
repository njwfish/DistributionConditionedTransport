#!/usr/bin/env python
"""
Evaluate a trained pfam model (with one-hot encoder) on a test .pt file.

For the one-hot encoder, we need to map test elements to training elements since
the one-hot encoding is based on training data indices. We do this by:
1. Computing mean-pooled ESM embeddings for all train and test elements
2. For each test element, finding the closest training element by cosine similarity
3. Using that training element's index as the "one-hot" encoding

Results are saved incrementally to allow resuming if interrupted.

Usage:
    python test_pfam_evaluation_onehot.py \
        --test_pt_file data/pfam/pfam_test.pt \
        --train_pt_file data/pfam/pfam_tokenized_data.pt \
        --output_dir outputs/pfam_onehot_<hash>
"""

import argparse
import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import logging
from datetime import datetime
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


def get_esm_model_and_device(device):
    """Load ESM model for computing embeddings."""
    from transformers import EsmModel, EsmTokenizer
    from utils.hf_local import resolve_local_or_repo
    
    model_name = 'facebook/esm2_t6_8M_UR50D'
    resolved_name = resolve_local_or_repo(model_name)
    
    model = EsmModel.from_pretrained(resolved_name)
    model.to(device)
    model.eval()
    
    return model


def compute_mean_pooled_embedding(esm_model, esm_input_ids, esm_attention_mask, device):
    """
    Compute mean-pooled ESM embedding for a set of sequences.
    
    Args:
        esm_model: ESM model
        esm_input_ids: [num_seqs, seq_len] tensor of token IDs
        esm_attention_mask: [num_seqs, seq_len] attention mask
        device: torch device
        
    Returns:
        [hidden_dim] tensor - mean-pooled embedding averaged over all sequences
    """
    esm_input_ids = esm_input_ids.to(device)
    esm_attention_mask = esm_attention_mask.to(device)
    
    with torch.no_grad():
        # Get ESM hidden states
        outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
        hidden_states = outputs.last_hidden_state  # [num_seqs, seq_len, hidden_dim]
        
        # Mean pool across residues (sequence positions) for each sequence
        mask = esm_attention_mask.unsqueeze(-1).float()  # [num_seqs, seq_len, 1]
        seq_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # [num_seqs, hidden_dim]
        
        # Average across all sequences in the set
        set_embedding = seq_embeddings.mean(dim=0)  # [hidden_dim]
        
    return set_embedding


def compute_and_cache_embeddings(data, cache_path, esm_model, device, desc="Computing embeddings"):
    """
    Compute mean-pooled ESM embeddings for all elements in data and cache them.
    
    Args:
        data: List of dicts with 'samples' containing 'esm_input_ids' and 'esm_attention_mask'
        cache_path: Path to save/load cached embeddings
        esm_model: ESM model
        device: torch device
        desc: Description for progress bar
        
    Returns:
        [num_elements, hidden_dim] tensor of embeddings
    """
    if os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        embeddings = torch.load(cache_path, weights_only=False)
        logger.info(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
        return embeddings
    
    logger.info(f"{desc} for {len(data)} elements...")
    embeddings = []
    
    for element in tqdm(data, desc=desc):
        samples = element['samples']
        emb = compute_mean_pooled_embedding(
            esm_model,
            samples['esm_input_ids'],
            samples['esm_attention_mask'],
            device
        )
        embeddings.append(emb.cpu())
    
    embeddings = torch.stack(embeddings, dim=0)  # [num_elements, hidden_dim]
    
    # Save cache
    logger.info(f"Saving embeddings to {cache_path}")
    torch.save(embeddings, cache_path)
    
    return embeddings


def find_closest_train_indices(test_embeddings, train_embeddings):
    """
    For each test embedding, find the closest training embedding by cosine similarity.
    
    Args:
        test_embeddings: [num_test, hidden_dim] tensor
        train_embeddings: [num_train, hidden_dim] tensor
        
    Returns:
        [num_test] tensor of indices into train_embeddings
    """
    # Normalize embeddings for cosine similarity
    test_norm = F.normalize(test_embeddings, p=2, dim=1)
    train_norm = F.normalize(train_embeddings, p=2, dim=1)
    
    # Compute cosine similarity matrix: [num_test, num_train]
    similarity = torch.mm(test_norm, train_norm.t())
    
    # Find closest training element for each test element
    closest_indices = similarity.argmax(dim=1)
    
    return closest_indices, similarity


def load_model_from_checkpoint(output_dir, device):
    """
    Load the encoder and generator from the best model checkpoint.
    
    Note: For one-hot encoder experiments, we only need to load the generator
    since the encoder is deterministic (one-hot based on index).
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
        checkpoints = [f for f in os.listdir(output_dir) if f.startswith('checkpoint_epoch_')]
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {output_dir}")
        checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        best_model_path = os.path.join(output_dir, checkpoints[-1])
        logger.info(f"Using latest checkpoint: {best_model_path}")
    else:
        logger.info(f"Using best model: {best_model_path}")
    
    # Instantiate encoder and generator
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


def evaluate_single_target(encoder, generator, source_samples, target_samples, 
                          source_train_idx, target_train_idx, device):
    """
    Compute the generator loss for a single source-target pair using one-hot encoding.
    
    Args:
        encoder: IndexOneHotEncoder
        generator: Generator model
        source_samples: Source samples dict
        target_samples: Target samples dict
        source_train_idx: Index of closest training element for source
        target_train_idx: Index of closest training element for target
        device: torch device
        
    Returns:
        loss value (float)
    """
    with torch.no_grad():
        # Get one-hot encodings from indices
        source_idx = torch.tensor([source_train_idx], device=device)
        target_idx = torch.tensor([target_train_idx], device=device)
        
        source_latent = encoder(source_idx)  # [1, n_unique_sets]
        target_latent = encoder(target_idx)  # [1, n_unique_sets]
        
        # Build full sample batches with all keys
        source_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in source_samples.items()
        }
        target_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target_samples.items()
        }
        
        # Compute generator loss
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
    parser = argparse.ArgumentParser(description='Evaluate pfam model (one-hot encoder) on test .pt file')
    parser.add_argument('--test_pt_file', type=str, required=True,
                        help='Path to the test .pt file created by PfamDataset')
    parser.add_argument('--train_pt_file', type=str, required=True,
                        help='Path to the training .pt file (e.g., data/pfam/pfam_tokenized_data.pt)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the outputs subdirectory containing best_model.pt and config.yaml')
    parser.add_argument('--source_idx', type=int, default=0,
                        help='Index of training element to use as source (default: 0)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Path to save results (default: <output_dir>/eval_results_onehot_<test_pt_name>.json)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--random_mapping', action='store_true',
                        help='Use random training indices instead of closest ESM embeddings')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random mapping (default: 42)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Handle relative paths
    if not os.path.isabs(args.test_pt_file):
        args.test_pt_file = os.path.join(SCRIPT_DIR, args.test_pt_file)
    if not os.path.isabs(args.train_pt_file):
        args.train_pt_file = os.path.join(SCRIPT_DIR, args.train_pt_file)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    
    # Verify paths exist
    if not os.path.exists(args.test_pt_file):
        logger.error(f"Test .pt file not found: {args.test_pt_file}")
        sys.exit(1)
    if not os.path.exists(args.train_pt_file):
        logger.error(f"Train .pt file not found: {args.train_pt_file}")
        sys.exit(1)
    if not os.path.exists(args.output_dir):
        logger.error(f"Output directory not found: {args.output_dir}")
        sys.exit(1)
    
    # Determine results file path
    if args.results_file is None:
        test_pt_name = os.path.splitext(os.path.basename(args.test_pt_file))[0]
        mapping_type = 'random' if args.random_mapping else 'onehot'
        args.results_file = os.path.join(args.output_dir, f'eval_results_{mapping_type}_{test_pt_name}.json')
    
    logger.info(f"Results will be saved to: {args.results_file}")
    logger.info(f"Mapping type: {'RANDOM' if args.random_mapping else 'SIMILARITY-BASED (closest ESM embedding)'}")
    
    # Load training data
    logger.info(f"Loading training data from: {args.train_pt_file}")
    train_data = torch.load(args.train_pt_file, weights_only=False)
    logger.info(f"Training data contains {len(train_data)} families")
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_pt_file}")
    test_data = torch.load(args.test_pt_file, weights_only=False)
    logger.info(f"Test data contains {len(test_data)} families")
    
    if args.random_mapping:
        # Random mapping: assign random training indices to test elements
        logger.info(f"Using random mapping with seed {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        # Generate random training indices for each test element
        random_train_indices = torch.tensor([
            random.randint(0, len(train_data) - 1) for _ in range(len(test_data))
        ])
        # No similarity scores for random mapping
        similarities = None
        closest_train_indices = random_train_indices
        
        # Log mapping statistics
        logger.info(f"Random training indices range: {closest_train_indices.min().item()} - {closest_train_indices.max().item()}")
        unique_indices = closest_train_indices.unique()
        logger.info(f"Number of unique training indices used: {len(unique_indices)}/{len(train_data)}")
    else:
        # Similarity-based mapping: use closest ESM embeddings
        # Load ESM model for computing embeddings
        logger.info("Loading ESM model for embedding computation...")
        esm_model = get_esm_model_and_device(device)
        
        # Compute/load cached embeddings
        train_cache_path = args.train_pt_file.replace('.pt', '_esm_embeddings.pt')
        test_cache_path = args.test_pt_file.replace('.pt', '_esm_embeddings.pt')
        
        train_embeddings = compute_and_cache_embeddings(
            train_data, train_cache_path, esm_model, device, 
            desc="Computing train embeddings"
        )
        test_embeddings = compute_and_cache_embeddings(
            test_data, test_cache_path, esm_model, device,
            desc="Computing test embeddings"
        )
        
        # Find closest training element for each test element
        logger.info("Finding closest training elements for test data...")
        closest_train_indices, similarities = find_closest_train_indices(test_embeddings, train_embeddings)
        
        # Log mapping statistics
        logger.info(f"Closest training indices range: {closest_train_indices.min().item()} - {closest_train_indices.max().item()}")
        unique_indices = closest_train_indices.unique()
        logger.info(f"Number of unique training indices used: {len(unique_indices)}/{len(train_data)}")
        
        # Free ESM model memory
        del esm_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Load trained model
    logger.info(f"Loading model from: {args.output_dir}")
    encoder, generator, config = load_model_from_checkpoint(args.output_dir, device)
    
    # Get source samples and its closest training index
    source_train_idx = args.source_idx
    source_element = train_data[source_train_idx]
    source_samples = source_element['samples']
    source_pfam = source_element.get('pfam', 'unknown')
    logger.info(f"Source: training element {source_train_idx} (family: {source_pfam})")
    
    # Load existing results for resumption
    results = load_results(args.results_file)
    if 'metadata' not in results:
        results['metadata'] = {
            'test_pt_file': args.test_pt_file,
            'train_pt_file': args.train_pt_file,
            'output_dir': args.output_dir,
            'source_train_idx': source_train_idx,
            'source_pfam': source_pfam,
            'num_test_families': len(test_data),
            'num_train_families': len(train_data),
            'mapping_type': 'random' if args.random_mapping else 'similarity',
            'random_seed': args.seed if args.random_mapping else None,
            'started_at': datetime.now().isoformat(),
            'device': str(device)
        }
    
    if 'losses' not in results:
        results['losses'] = {}
    
    if 'mappings' not in results:
        results['mappings'] = {}
    
    completed = set(results['losses'].keys())
    logger.info(f"Already completed: {len(completed)}/{len(test_data)} families")
    
    # Evaluate each target family
    total_loss = 0.0
    count = 0
    
    for idx, target_element in enumerate(test_data):
        target_pfam = target_element.get('pfam', f'family_{idx}')
        
        # Skip if already computed
        if target_pfam in completed:
            if 'loss' in results['losses'][target_pfam]:
                loss = results['losses'][target_pfam]['loss']
                total_loss += loss
                count += 1
            continue
        
        target_samples = target_element['samples']
        num_target_seqs = target_samples['esm_input_ids'].shape[0]
        
        # Get the mapped training index for this test element
        target_train_idx = closest_train_indices[idx].item()
        target_train_pfam = train_data[target_train_idx].get('pfam', f'train_{target_train_idx}')
        similarity = similarities[idx, target_train_idx].item() if similarities is not None else None
        
        try:
            loss = evaluate_single_target(
                encoder, generator, source_samples, target_samples,
                source_train_idx, target_train_idx, device
            )
            
            # Save result
            results['losses'][target_pfam] = {
                'loss': loss,
                'num_sequences': num_target_seqs,
                'idx': idx,
                'mapped_to_train_idx': target_train_idx,
                'mapped_to_train_pfam': target_train_pfam,
                'similarity': similarity
            }
            
            results['mappings'][target_pfam] = {
                'train_idx': target_train_idx,
                'train_pfam': target_train_pfam,
                'similarity': similarity
            }
            
            total_loss += loss
            count += 1
            
            # Print progress
            avg_loss = total_loss / count
            if similarity is not None:
                logger.info(f"[{count}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} (idx={target_train_idx}, sim={similarity:.4f}), "
                           f"Loss: {loss:.6f}, Running avg: {avg_loss:.6f}")
            else:
                logger.info(f"[{count}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} (idx={target_train_idx}, random), "
                           f"Loss: {loss:.6f}, Running avg: {avg_loss:.6f}")
            
            # Save intermediate results
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
            import traceback
            traceback.print_exc()
            results['losses'][target_pfam] = {
                'error': str(e),
                'idx': idx,
                'mapped_to_train_idx': target_train_idx
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
        family_losses = [(k, v['loss'], v.get('mapped_to_train_pfam', '?')) 
                        for k, v in results['losses'].items() if 'loss' in v]
        family_losses.sort(key=lambda x: x[1])
        for pfam, loss, mapped_pfam in family_losses:
            logger.info(f"  {pfam} -> {mapped_pfam}: {loss:.6f}")
    else:
        logger.warning("No families were successfully evaluated!")


if __name__ == '__main__':
    main()
