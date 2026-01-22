#!/usr/bin/env python
"""
Evaluation script for model performance.

Usage:
    python evaluate.py --experiment_dir outputs/synth_discrete_esm_dfm_64ea7ffca915ffdbbff95450012c9e99
"""

import argparse
import os
import sys
import torch
import numpy as np
import random
from omegaconf import OmegaConf


def load_config(experiment_dir: str):
    """Load the config.yaml from the experiment directory."""
    config_path = os.path.join(experiment_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    return OmegaConf.load(config_path)


def load_data(cfg, base_dir: str):
    """Load the train and test data files that were saved by the dataset class."""
    data_dir = cfg.dataset.data_dir
    data_file = cfg.dataset.data_file
    
    train_data_path = os.path.join(base_dir, data_dir, data_file)
    test_data_path = os.path.join(base_dir, data_dir, 'eval_' + data_file)
    
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Train data not found at {train_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found at {test_data_path}")
    
    train_data = torch.load(train_data_path, weights_only=False)
    test_data = torch.load(test_data_path, weights_only=False)
    
    print(f"Loaded train data: {len(train_data)} distributions")
    print(f"Loaded test data: {len(test_data)} distributions")
    
    return train_data, test_data


def instantiate_encoder(cfg, device):
    """Instantiate the encoder from config."""
    # Get the encoder parameters from the resolved config
    exp = cfg.experiment
    
    from encoder.protein_encoders import ProteinSetEncoder
    
    encoder = ProteinSetEncoder(
        esm_model_name=exp.esm_model_name,
        latent_dim=exp.latent_dim,
        hidden_dim=exp.hidden_dim,
        pooling=cfg.encoder.pooling,
        freeze=cfg.encoder.freeze,
        dist_type=cfg.encoder.dist_type,
        layers=cfg.encoder.layers,
        heads=cfg.encoder.heads,
    )
    return encoder.to(device)


def instantiate_generator(cfg, device):
    """Instantiate the generator from config."""
    exp = cfg.experiment
    
    from generator.dfm import ESM2_DFM_Generator
    
    generator = ESM2_DFM_Generator(
        model_name=exp.esm_model_name,
        latent_dim=exp.latent_dim,
        condition_dim=exp.condition_dim,
        freeze_esm2=exp.freeze_esm2,
        temperature=exp.temperature,
    )
    return generator.to(device)


def load_model_weights(encoder, generator, experiment_dir, device):
    """Load weights from best_model.pt."""
    best_model_path = os.path.join(experiment_dir, "best_model.pt")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    
    print(f"Loaded model weights from epoch {checkpoint['epoch']} (loss: {checkpoint['loss']:.6f})")
    
    return encoder, generator


def get_sample_batch(data, indices, set_size, device):
    """
    Create a batch from the given data at specified indices.
    
    Returns a dict with the same structure as the dataset's __getitem__ output.
    """
    source_idx, target_idx = indices
    item_source = data[source_idx]
    item_target = data[target_idx]
    
    # Select set_size samples from each distribution
    n_source = len(item_source['samples']['esm_input_ids'])
    n_target = len(item_target['samples']['esm_input_ids'])
    
    source_subset_indices = np.random.choice(
        n_source, size=min(set_size, n_source), replace=False
    )
    target_subset_indices = np.random.choice(
        n_target, size=min(set_size, n_target), replace=False
    )
    
    # Build source samples dict
    source_samples = {
        'esm_input_ids': item_source['samples']['esm_input_ids'][source_subset_indices].unsqueeze(0).to(device),
        'esm_attention_mask': item_source['samples']['esm_attention_mask'][source_subset_indices].unsqueeze(0).to(device),
        'raw_texts': [item_source['raw_texts'][i] for i in source_subset_indices],
        'canonical_kmer': item_source['canonical_kmer'],
    }
    
    # Build target samples dict
    target_samples = {
        'esm_input_ids': item_target['samples']['esm_input_ids'][target_subset_indices].unsqueeze(0).to(device),
        'esm_attention_mask': item_target['samples']['esm_attention_mask'][target_subset_indices].unsqueeze(0).to(device),
        'raw_texts': [item_target['raw_texts'][i] for i in target_subset_indices],
        'canonical_kmer': item_target['canonical_kmer'],
    }
    
    return source_samples, target_samples


def sample_and_print(encoder, generator, source_samples, target_samples, scenario_name, num_samples=1):
    """
    Use the generator to sample sequences and print them alongside the true targets.
    """
    print(f"\n{'='*60}")
    print(f"Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    # Get latent representations
    with torch.no_grad():
        latent_source = encoder(source_samples)
        latent_target = encoder(target_samples)
    
    # Sample sequences
    with torch.no_grad():
        _, sampled_texts = generator.sample(
            source_samples, 
            latent_source, 
            latent_target, 
            num_samples=num_samples, 
            return_texts=True
        )
    
    print(f"\nSource k-mer: {source_samples['canonical_kmer']}")
    print(f"Target k-mer: {target_samples['canonical_kmer']}")
    
    print(f"\n--- Source sequences (input) ---")
    for i, text in enumerate(source_samples['raw_texts'][:3]):  # Show first 3
        print(f"  {i+1}. {text}")
    if len(source_samples['raw_texts']) > 3:
        print(f"  ... ({len(source_samples['raw_texts'])} total)")
    
    print(f"\n--- Target sequences (ground truth) ---")
    for i, text in enumerate(target_samples['raw_texts'][:3]):  # Show first 3
        print(f"  {i+1}. {text}")
    if len(target_samples['raw_texts']) > 3:
        print(f"  ... ({len(target_samples['raw_texts'])} total)")
    
    print(f"\n--- Sampled sequences (generated) ---")
    # sampled_texts is [batch][num_samples], batch size is 1
    for i, text in enumerate(sampled_texts[0]):
        print(f"  {i+1}. {text}")
    
    return sampled_texts


def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument(
        "--experiment_dir", 
        type=str, 
        required=True,
        help="Path to the experiment directory containing config.yaml and best_model.pt"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=4,
        help="Number of sequences to sample per scenario (default: 4)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Determine base directory (where the repo is)
    experiment_dir = os.path.abspath(args.experiment_dir)
    # Base dir is typically 2 levels up from the experiment directory
    # e.g., outputs/experiment_name/ -> base is parent of outputs
    base_dir = os.path.dirname(os.path.dirname(experiment_dir))
    
    # If experiment_dir is directly in outputs/ (not nested), adjust
    if os.path.basename(os.path.dirname(experiment_dir)) == "outputs":
        base_dir = os.path.dirname(os.path.dirname(experiment_dir))
    else:
        # Fallback: use current working directory
        base_dir = os.getcwd()
    
    print(f"Experiment directory: {experiment_dir}")
    print(f"Base directory: {base_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load config
    print("\nLoading configuration...")
    cfg = load_config(experiment_dir)
    
    # Load data
    print("\nLoading train and test data...")
    train_data, test_data = load_data(cfg, base_dir)
    
    # Instantiate models
    print("\nInstantiating encoder...")
    encoder = instantiate_encoder(cfg, device)
    
    print("Instantiating generator...")
    generator = instantiate_generator(cfg, device)
    
    # Load weights
    print("\nLoading model weights...")
    encoder, generator = load_model_weights(encoder, generator, experiment_dir, device)
    
    # Set to eval mode
    encoder.eval()
    generator.eval()
    
    # Get set size from config
    set_size = cfg.experiment.set_size
    
    # Prepare indices for each scenario (2 source-target pairs each)
    # We need at least 2 different distributions for train and test
    n_train = len(train_data)
    n_test = len(test_data)
    
    print(f"\nGenerating samples for 4 scenarios with {args.num_samples} samples each...")
    
    # 1. Train -> Train (2 pairs)
    print("\n" + "="*80)
    print("TRAIN -> TRAIN")
    print("="*80)
    for pair_idx in range(2):
        # Pick random source and target from train data
        src_idx = np.random.randint(0, n_train)
        tgt_idx = np.random.randint(0, n_train)
        
        source_samples, target_samples = get_sample_batch(
            train_data, (src_idx, tgt_idx), set_size, device
        )
        sample_and_print(
            encoder, generator, source_samples, target_samples,
            f"Train->Train (pair {pair_idx+1})", 
            num_samples=args.num_samples
        )
    
    # 2. Train -> Test (2 pairs)
    print("\n" + "="*80)
    print("TRAIN -> TEST")
    print("="*80)
    for pair_idx in range(2):
        src_idx = np.random.randint(0, n_train)
        tgt_idx = np.random.randint(0, n_test)
        
        source_samples, target_samples = get_sample_batch(
            train_data, (src_idx, src_idx), set_size, device  # source from train
        )
        _, target_samples = get_sample_batch(
            test_data, (tgt_idx, tgt_idx), set_size, device  # target from test
        )
        sample_and_print(
            encoder, generator, source_samples, target_samples,
            f"Train->Test (pair {pair_idx+1})", 
            num_samples=args.num_samples
        )
    
    # 3. Test -> Train (2 pairs)
    print("\n" + "="*80)
    print("TEST -> TRAIN")
    print("="*80)
    for pair_idx in range(2):
        src_idx = np.random.randint(0, n_test)
        tgt_idx = np.random.randint(0, n_train)
        
        source_samples, _ = get_sample_batch(
            test_data, (src_idx, src_idx), set_size, device  # source from test
        )
        _, target_samples = get_sample_batch(
            train_data, (tgt_idx, tgt_idx), set_size, device  # target from train
        )
        sample_and_print(
            encoder, generator, source_samples, target_samples,
            f"Test->Train (pair {pair_idx+1})", 
            num_samples=args.num_samples
        )
    
    # 4. Test -> Test (2 pairs)
    print("\n" + "="*80)
    print("TEST -> TEST")
    print("="*80)
    for pair_idx in range(2):
        src_idx = np.random.randint(0, n_test)
        tgt_idx = np.random.randint(0, n_test)
        
        source_samples, target_samples = get_sample_batch(
            test_data, (src_idx, tgt_idx), set_size, device
        )
        sample_and_print(
            encoder, generator, source_samples, target_samples,
            f"Test->Test (pair {pair_idx+1})", 
            num_samples=args.num_samples
        )
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)


if __name__ == "__main__":
    main()
