#!/usr/bin/env python
"""
Evaluate a trained pfam model (with one-hot encoder) on a test .pt file.

For the one-hot encoder, we need to map test elements to training elements since
the one-hot encoding is based on training data indices. We do this by:
1. Computing mean-pooled ESM embeddings for all train and test elements
2. For each test element, finding the closest training element by cosine similarity
3. Using that training element's index as the "one-hot" encoding

Alternatively, use --random_mapping to randomly assign training indices (baseline).

Usage:
    python test_pfam_evaluation_onehot.py \
        --test_pt_file data/pfam/pfam_test.pt \
        --train_pt_file data/pfam/pfam_tokenized_data.pt \
        --output_dir outputs/pfam_onehot_<hash>
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


def find_closest_train_indices(test_embeddings, train_embeddings):
    """
    For each test embedding, find the closest training embedding by cosine similarity.
    """
    test_norm = F.normalize(test_embeddings, p=2, dim=1)
    train_norm = F.normalize(train_embeddings, p=2, dim=1)
    
    similarity = torch.mm(test_norm, train_norm.t())
    closest_indices = similarity.argmax(dim=1)
    
    return closest_indices, similarity


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


def compute_oracle_loss(generator, source_samples, target_samples, oracle_samples, device):
    """
    Compute the oracle loss: what the loss would be if the generator perfectly
    reproduced the oracle (training) sequences.
    
    This bypasses the generator forward pass and instead uses one-hot logits
    derived from the oracle sequences to compare against the actual target sequences.
    
    Args:
        generator: The generator model (used for tokenizer and concatenation logic)
        source_samples: Source sequence samples (dict with progen_input_ids, etc.)
        target_samples: Target sequence samples to evaluate against (test element)
        oracle_samples: The "perfect" predictions - typically from the closest training element
        device: torch device
    
    Returns:
        Loss value (float)
    """
    import torch.nn as nn
    
    source_batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in source_samples.items()
    }
    target_batch = {
        key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
        for key, value in target_samples.items()
    }
    
    # Use the generator's concatenation logic to get target_start_idx
    combined_input_ids, combined_attention_mask, target_start_idx = generator._concatenate_source_target(
        source_batch, target_batch
    )
    
    # Handle potential 3D tensor shapes (batch_size, set_size, seq_len)
    if len(combined_input_ids.shape) == 3:
        batch_size, set_size, seq_len = combined_input_ids.shape
        combined_input_ids = combined_input_ids.view(batch_size * set_size, seq_len)
    
    # Get the target labels (the actual target token IDs from test element)
    target_labels = combined_input_ids[:, target_start_idx:]  # Target token IDs
    target_seq_len = target_labels.shape[1]
    
    # Get oracle token IDs (from closest training element)
    oracle_target_ids = oracle_samples['progen_input_ids'].unsqueeze(0).to(device)
    
    # Handle 3D oracle shape
    if len(oracle_target_ids.shape) == 3:
        batch_size, set_size, seq_len = oracle_target_ids.shape
        oracle_target_ids = oracle_target_ids.view(batch_size * set_size, seq_len)
    
    # Get vocab size from the generator's tokenizer
    vocab_size = generator.tokenizer.vocab_size
    pad_token_id = generator.tokenizer.pad_token_id
    
    # Adjust oracle_target_ids length to match target_labels length
    oracle_seq_len = oracle_target_ids.shape[1]
    if oracle_seq_len > target_seq_len:
        # Truncate oracle to match target length
        oracle_target_ids = oracle_target_ids[:, :target_seq_len]
    elif oracle_seq_len < target_seq_len:
        # Pad oracle with pad tokens to match target length
        pad_length = target_seq_len - oracle_seq_len
        pad_tokens = torch.full(
            (oracle_target_ids.shape[0], pad_length),
            pad_token_id,
            dtype=oracle_target_ids.dtype,
            device=oracle_target_ids.device
        )
        oracle_target_ids = torch.cat([oracle_target_ids, pad_tokens], dim=1)
    
    # Create one-hot logits from oracle_target_ids
    # Use large positive value for the oracle token to simulate confident predictions
    # Shape: [batch_size, target_seq_len, vocab_size]
    target_logits = F.one_hot(oracle_target_ids, num_classes=vocab_size).float()
    # Scale to make them behave like confident logits
    # Using 100.0 makes the softmax output essentially a one-hot
    target_logits = target_logits * 100.0
    
    # Calculate loss only on target tokens
    loss_fct = nn.CrossEntropyLoss(reduction='mean', ignore_index=pad_token_id)
    loss = loss_fct(
        target_logits.reshape(-1, target_logits.size(-1)),
        target_labels.reshape(-1)
    )
    
    return loss.item()


def evaluate_single_target(encoder, generator, source_samples, target_samples, 
                          source_train_idx, target_train_idx, device,
                          oracle_samples=None):
    """
    Compute the generator loss for a single source-target pair using one-hot encoding.
    
    Args:
        encoder: The encoder model
        generator: The generator model
        source_samples: Source sequence samples (dict with progen_input_ids, etc.)
        target_samples: Target sequence samples to evaluate against (test element)
        source_train_idx: Index of training element to use as source
        target_train_idx: Index of training element to use for target one-hot encoding
        device: torch device
        oracle_samples: Optional. If provided, use these samples' token IDs as the "perfect"
            predictions instead of running the generator. This tests the ideal case where
            the model perfectly learned to generate these oracle sequences based on the
            one-hot encoding. Typically these would be the samples from the closest
            training element.
    
    Returns:
        Loss value (float)
    """
    with torch.no_grad():
        # If oracle mode, compute oracle loss instead
        if oracle_samples is not None:
            return compute_oracle_loss(generator, source_samples, target_samples, 
                                       oracle_samples, device)
        
        # Normal mode: use the generator
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
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (default: cuda if available, else cpu)')
    parser.add_argument('--random_mapping', action='store_true',
                        help='Use random training indices instead of closest ESM embeddings')
    parser.add_argument('--oracle_mode', action='store_true',
                        help='Use oracle mode: instead of generator predictions, use one-hot logits '
                             'from the closest training element. This tests the ideal case where '
                             'the model perfectly learned to generate training sequences for each '
                             'one-hot encoding.')
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
    
    logger.info(f"Mapping type: {'RANDOM' if args.random_mapping else 'SIMILARITY-BASED (closest ESM embedding)'}")
    logger.info(f"Oracle mode: {'ENABLED' if args.oracle_mode else 'DISABLED'}")
    if args.oracle_mode:
        logger.info("  (Using one-hot logits from closest training element instead of generator predictions)")
    
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
        
        random_train_indices = torch.tensor([
            random.randint(0, len(train_data) - 1) for _ in range(len(test_data))
        ])
        similarities = None
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
        
        logger.info("Finding closest training elements for test data...")
        closest_train_indices, similarities = find_closest_train_indices(test_embeddings, train_embeddings)
        
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
        closest_train_indices = closest_train_indices[:args.num_samples]
        if similarities is not None:
            similarities = similarities[:args.num_samples]
        logger.info(f"Limiting evaluation to first {len(test_data)} families")
    
    # Load trained model
    logger.info(f"Loading model from: {args.output_dir}")
    encoder, generator, config = load_model_from_checkpoint(args.output_dir, device)
    
    # Get source samples
    source_train_idx = args.source_idx
    source_element = train_data[source_train_idx]
    source_samples = source_element['samples']
    source_pfam = source_element.get('pfam', 'unknown')
    logger.info(f"Source: training element {source_train_idx} (family: {source_pfam})")
    
    # Evaluate each target family
    all_losses = []
    
    for idx, target_element in enumerate(tqdm(test_data, desc="Evaluating", disable=args.verbose)):
        target_pfam = target_element.get('pfam', f'family_{idx}')
        target_samples = target_element['samples']
        
        target_train_idx = closest_train_indices[idx].item()
        target_train_pfam = train_data[target_train_idx].get('pfam', f'train_{target_train_idx}')
        similarity = similarities[idx, target_train_idx].item() if similarities is not None else None
        
        # Get oracle samples from closest training element if oracle mode is enabled
        oracle_samples = None
        if args.oracle_mode:
            oracle_samples = train_data[target_train_idx]['samples']
        
        try:
            loss = evaluate_single_target(
                encoder, generator, source_samples, target_samples,
                source_train_idx, target_train_idx, device,
                oracle_samples=oracle_samples
            )
            all_losses.append(loss)
            
            if args.verbose:
                if similarity is not None:
                    logger.info(f"[{idx+1}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} "
                               f"(idx={target_train_idx}, sim={similarity:.4f}), Loss: {loss:.6f}")
                else:
                    logger.info(f"[{idx+1}/{len(test_data)}] Test: {target_pfam} -> Train: {target_train_pfam} "
                               f"(idx={target_train_idx}, random), Loss: {loss:.6f}")
            
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
        print(f"Mapping type:      {'Random' if args.random_mapping else 'Similarity-based'}")
        print(f"Oracle mode:       {'Enabled' if args.oracle_mode else 'Disabled'}")
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
