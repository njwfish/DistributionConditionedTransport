#!/usr/bin/env python
"""
Unified evaluation script for pfam clan models.

Supports models trained with any of the following experiment configs:
- experiment=pfam_onehot_clan (index_onehot encoder + progen2 generator)
- experiment=pfam_dfm_onehot_clan (index_onehot encoder + dfm generator)
- experiment=pfam_clan (esm encoder + progen2 generator)
- experiment=pfam_dfm_clan (esm encoder + dfm generator)

The script automatically detects:
- Encoder type (index_onehot vs esm) from the saved config
- Generator type (progen2 vs dfm) and uses the corresponding loss computation

For one-hot encoder models, we map test elements to training elements by:
1. Computing mean-pooled ESM embeddings for all train and test elements
2. For each test element, finding the closest training element by cosine similarity or MSE distance
3. Using that training element's index as the "one-hot" encoding

Usage:
    python test_pfam_evaluation_clan.py \
        --output_dir outputs/pfam_onehot_clan_<hash>

    # With explicit data files:
    python test_pfam_evaluation_clan.py \
        --train_pt_file data/pfam/pfam_tokenized_data_clan.pt \
        --test_pt_file data/pfam/pfam_tokenized_data_clan_eval.pt \
        --output_dir outputs/pfam_clan_<hash>
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import torch.nn.functional as F
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

# Default data files for clan experiments
DEFAULT_TRAIN_FILE = 'data/pfam/pfam_tokenized_data_clan.pt'
DEFAULT_TEST_FILE = 'data/pfam/pfam_tokenized_data_clan_eval.pt'


def get_esm_model_and_device(device):
    """Load ESM model for computing embeddings."""
    from transformers import EsmModel
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
        outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
        hidden_states = outputs.last_hidden_state
        
        mask = esm_attention_mask.unsqueeze(-1).float()
        seq_embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        set_embedding = seq_embeddings.mean(dim=0)
        
    return set_embedding


def compute_and_cache_embeddings(data, cache_path, esm_model, device, desc="Computing embeddings"):
    """
    Compute mean-pooled ESM embeddings for all elements in data and cache them.
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
    
    embeddings = torch.stack(embeddings, dim=0)
    
    logger.info(f"Saving embeddings to {cache_path}")
    torch.save(embeddings, cache_path)
    
    return embeddings


def find_closest_train_indices(test_embeddings, train_embeddings, method='cosine'):
    """
    For each test embedding, find the closest training embedding.
    
    Args:
        test_embeddings: [num_test, embed_dim] tensor
        train_embeddings: [num_train, embed_dim] tensor
        method: 'cosine' for cosine similarity (higher is closer) or 
                'mse' for mean squared error (lower is closer)
    
    Returns:
        closest_indices: [num_test] tensor of indices into train_embeddings
        scores: [num_test, num_train] tensor of similarity/distance scores
    """
    if method == 'cosine':
        # Cosine similarity: higher is better
        test_norm = F.normalize(test_embeddings, p=2, dim=1)
        train_norm = F.normalize(train_embeddings, p=2, dim=1)
        
        scores = torch.mm(test_norm, train_norm.t())
        closest_indices = scores.argmax(dim=1)
    elif method == 'mse':
        # Mean squared error: lower is better
        # Compute pairwise squared distances efficiently
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        test_sq = (test_embeddings ** 2).sum(dim=1, keepdim=True)  # [num_test, 1]
        train_sq = (train_embeddings ** 2).sum(dim=1, keepdim=True)  # [num_train, 1]
        
        # Pairwise dot products
        dot_product = torch.mm(test_embeddings, train_embeddings.t())  # [num_test, num_train]
        
        # MSE = (test_sq + train_sq.T - 2 * dot_product) / embed_dim
        embed_dim = test_embeddings.shape[1]
        scores = (test_sq + train_sq.t() - 2 * dot_product) / embed_dim
        
        closest_indices = scores.argmin(dim=1)  # Lower MSE is better
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cosine' or 'mse'.")
    
    return closest_indices, scores


def detect_model_types(config):
    """
    Detect encoder and generator types from the saved config.
    
    Returns:
        encoder_type: 'index_onehot' or 'esm'
        generator_type: 'progen2' or 'dfm'
    """
    # Detect encoder type
    encoder_target = config.encoder.get('_target_', '')
    if 'IndexOneHotEncoder' in encoder_target or 'index_onehot' in encoder_target.lower():
        encoder_type = 'index_onehot'
    elif 'ESM' in encoder_target or 'esm' in encoder_target.lower() or encoder_target == 'encoder.protein_encoders.ProteinSetEncoder':
        encoder_type = 'esm'
    else:
        raise ValueError(f"Unknown encoder type: {encoder_target}")
    
    # Detect generator type
    generator_target = config.generator.get('_target_', '')
    if 'DFM' in generator_target or 'dfm' in generator_target.lower() or 'ESM2_DFM' in generator_target:
        generator_type = 'dfm'
    elif 'Progen' in generator_target or 'progen' in generator_target.lower():
        generator_type = 'progen2'
    else:
        raise ValueError(f"Unknown generator type: {generator_target}")
    
    return encoder_type, generator_type


def load_model_from_checkpoint(output_dir, device):
    """
    Load the encoder and generator from the best model checkpoint.
    """
    import hydra
    
    config_path = os.path.join(output_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")
    
    config = OmegaConf.load(config_path)
    logger.info(f"Loaded config from {config_path}")
    
    # Detect model types
    encoder_type, generator_type = detect_model_types(config)
    logger.info(f"Detected encoder type: {encoder_type}")
    logger.info(f"Detected generator type: {generator_type}")
    
    best_model_path = os.path.join(output_dir, 'best_model.pt')
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
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
    
    return encoder, generator, config, encoder_type, generator_type


def evaluate_single_target_onehot_progen2(encoder, generator, source_samples, target_samples, 
                                          source_train_idx, target_train_idx, device):
    """
    Compute the generator loss for a single source-target pair using one-hot encoding
    with the progen2 generator.
    """
    with torch.no_grad():
        source_idx = torch.tensor([source_train_idx], device=device)
        target_idx = torch.tensor([target_train_idx], device=device)
        
        source_latent = encoder(source_idx)
        target_latent = encoder(target_idx)
        
        source_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in source_samples.items()
        }
        target_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target_samples.items()
        }
        
        loss = generator.loss(source_batch, target_batch, source_latent, target_latent)
        
        return loss.item()


def evaluate_single_target_onehot_dfm(encoder, generator, source_samples, target_samples, 
                                      source_train_idx, target_train_idx, device):
    """
    Compute the generator loss for a single source-target pair using one-hot encoding
    with the DFM generator.
    
    The DFM generator uses discrete flow-matching loss, which interpolates between
    source and target sequences.
    """
    with torch.no_grad():
        source_idx = torch.tensor([source_train_idx], device=device)
        target_idx = torch.tensor([target_train_idx], device=device)
        
        source_latent = encoder(source_idx)
        target_latent = encoder(target_idx)
        
        source_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in source_samples.items()
        }
        target_batch = {
            key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
            for key, value in target_samples.items()
        }
        
        loss = generator.loss(source_batch, target_batch, source_latent, target_latent)
        
        return loss.item()


def evaluate_single_target_esm_progen2(encoder, generator, source_samples, target_samples, device):
    """
    Compute the generator loss for a single source-target pair using ESM encoder
    with the progen2 generator.
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


def evaluate_single_target_esm_dfm(encoder, generator, source_samples, target_samples, device):
    """
    Compute the generator loss for a single source-target pair using ESM encoder
    with the DFM generator.
    """
    with torch.no_grad():
        # Build full sample batches with all keys
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
        
        # Compute generator loss using DFM loss
        loss = generator.loss(source_batch, target_batch, source_latent, target_latent)
        
        return loss.item()


def main():
    parser = argparse.ArgumentParser(description='Unified evaluation for pfam clan models')
    parser.add_argument('--test_pt_file', type=str, default=DEFAULT_TEST_FILE,
                        help=f'Path to the test/validation .pt file (default: {DEFAULT_TEST_FILE})')
    parser.add_argument('--train_pt_file', type=str, default=DEFAULT_TRAIN_FILE,
                        help=f'Path to the training .pt file (default: {DEFAULT_TRAIN_FILE})')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the outputs subdirectory containing best_model.pt and config.yaml')
    parser.add_argument('--source_idx', type=int, default=0,
                        help='Index of training element to use as source (default: 0)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--random_mapping', action='store_true',
                        help='Use random training indices instead of closest ESM embeddings (for one-hot encoder only)')
    parser.add_argument('--similarity_method', type=str, default='cosine', choices=['cosine', 'mse'],
                        help='Method for finding closest training elements: "cosine" (cosine similarity) '
                             'or "mse" (mean squared error distance). Default: cosine')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for random mapping (default: 42)')
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
    
    # Load trained model and detect types
    logger.info(f"Loading model from: {args.output_dir}")
    encoder, generator, config, encoder_type, generator_type = load_model_from_checkpoint(args.output_dir, device)
    
    # Log configuration
    logger.info(f"Encoder type: {encoder_type}")
    logger.info(f"Generator type: {generator_type}")
    
    if encoder_type == 'index_onehot':
        if args.random_mapping:
            logger.info("Mapping type: RANDOM")
        else:
            method_desc = "cosine similarity" if args.similarity_method == 'cosine' else "MSE distance"
            logger.info(f"Mapping type: SIMILARITY-BASED ({method_desc})")
    
    # Load training data
    logger.info(f"Loading training data from: {args.train_pt_file}")
    train_data = torch.load(args.train_pt_file, weights_only=False)
    logger.info(f"Training data contains {len(train_data)} families")
    
    # Load test data
    logger.info(f"Loading test data from: {args.test_pt_file}")
    test_data = torch.load(args.test_pt_file, weights_only=False)
    logger.info(f"Test data contains {len(test_data)} families")
    
    # For one-hot encoder, we need to map test elements to training elements
    closest_train_indices = None
    scores = None
    
    if encoder_type == 'index_onehot':
        if args.random_mapping:
            # Random mapping: assign random training indices to test elements
            logger.info(f"Using random mapping with seed {args.seed}")
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            
            random_train_indices = torch.tensor([
                random.randint(0, len(train_data) - 1) for _ in range(len(test_data))
            ])
            scores = None
            closest_train_indices = random_train_indices
            
            logger.info(f"Random training indices range: {closest_train_indices.min().item()} - {closest_train_indices.max().item()}")
            unique_indices = closest_train_indices.unique()
            logger.info(f"Number of unique training indices used: {len(unique_indices)}/{len(train_data)}")
        else:
            # Similarity-based mapping: use closest ESM embeddings
            logger.info("Loading ESM model for embedding computation...")
            esm_model = get_esm_model_and_device(device)
            
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
            
            logger.info(f"Finding closest training elements for test data using {args.similarity_method} method...")
            closest_train_indices, scores = find_closest_train_indices(
                test_embeddings, train_embeddings, method=args.similarity_method
            )
            
            logger.info(f"Closest training indices range: {closest_train_indices.min().item()} - {closest_train_indices.max().item()}")
            unique_indices = closest_train_indices.unique()
            logger.info(f"Number of unique training indices used: {len(unique_indices)}/{len(train_data)}")
            
            # Free ESM model memory
            del esm_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Limit to first N elements if specified
    if args.num_samples is not None:
        test_data = test_data[:args.num_samples]
        if closest_train_indices is not None:
            closest_train_indices = closest_train_indices[:args.num_samples]
        if scores is not None:
            scores = scores[:args.num_samples]
        logger.info(f"Limiting evaluation to first {len(test_data)} families")
    
    # Get source samples (for one-hot encoder, use training data; for ESM encoder, use test data element 0)
    source_train_idx = args.source_idx
    if encoder_type == 'index_onehot':
        source_element = train_data[source_train_idx]
    else:
        # For ESM encoder, the source comes from the train data by default
        source_element = train_data[source_train_idx]
    source_samples = source_element['samples']
    source_pfam = source_element.get('pfam', source_element.get('clan', 'unknown'))
    logger.info(f"Source: training element {source_train_idx} (family/clan: {source_pfam})")
    
    # Select the appropriate evaluation function based on encoder and generator types
    if encoder_type == 'index_onehot':
        if generator_type == 'progen2':
            evaluate_fn = evaluate_single_target_onehot_progen2
        else:  # dfm
            evaluate_fn = evaluate_single_target_onehot_dfm
    else:  # esm encoder
        if generator_type == 'progen2':
            evaluate_fn = evaluate_single_target_esm_progen2
        else:  # dfm
            evaluate_fn = evaluate_single_target_esm_dfm
    
    # Evaluate each target family
    all_losses = []
    
    for idx, target_element in enumerate(tqdm(test_data, desc="Evaluating", disable=args.verbose)):
        target_pfam = target_element.get('pfam', target_element.get('clan', f'family_{idx}'))
        target_samples = target_element['samples']
        
        try:
            if encoder_type == 'index_onehot':
                target_train_idx = closest_train_indices[idx].item()
                target_train_pfam = train_data[target_train_idx].get('pfam', 
                                     train_data[target_train_idx].get('clan', f'train_{target_train_idx}'))
                score = scores[idx, target_train_idx].item() if scores is not None else None
                
                loss = evaluate_fn(
                    encoder, generator, source_samples, target_samples,
                    source_train_idx, target_train_idx, device
                )
                
                if args.verbose:
                    if score is not None:
                        score_label = "sim" if args.similarity_method == 'cosine' else "mse"
                        logger.info(f"[{idx+1}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} "
                                   f"(idx={target_train_idx}, {score_label}={score:.4f}), Loss: {loss:.6f}")
                    else:
                        logger.info(f"[{idx+1}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} "
                                   f"(idx={target_train_idx}, random), Loss: {loss:.6f}")
            else:
                # ESM encoder: directly encode target samples
                loss = evaluate_fn(
                    encoder, generator, source_samples, target_samples, device
                )
                
                if args.verbose:
                    num_target_seqs = target_samples['esm_input_ids'].shape[0]
                    logger.info(f"[{idx+1}/{len(test_data)}] Family: {target_pfam}, "
                               f"Loss: {loss:.6f}, Num seqs: {num_target_seqs}")
            
            all_losses.append(loss)
            
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
        print(f"Encoder type:       {encoder_type}")
        print(f"Generator type:     {generator_type}")
        if encoder_type == 'index_onehot':
            if args.random_mapping:
                print(f"Mapping type:       Random")
            else:
                method_desc = "Cosine similarity" if args.similarity_method == 'cosine' else "MSE distance"
                print(f"Mapping type:       {method_desc}")
        print(f"Families evaluated: {len(all_losses)}/{len(test_data)}")
        print("-" * 60)
        print(f"Mean loss:          {losses_array.mean():.6f}")
        print(f"Median loss:        {np.median(losses_array):.6f}")
        print(f"Std loss:           {losses_array.std():.6f}")
        print(f"Min loss:           {losses_array.min():.6f}")
        print(f"Max loss:           {losses_array.max():.6f}")
        print("=" * 60)
    else:
        logger.warning("No families were successfully evaluated!")


if __name__ == '__main__':
    main()
