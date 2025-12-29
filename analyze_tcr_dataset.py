#!/usr/bin/env python3
"""
Script to analyze and visualize the TCRDataset class structure.
This script samples elements from the dataset and provides detailed information
about shapes, lengths, and structure of the data.
"""

import torch
import numpy as np
from datasets.tcr import TCRDataset
import sys

def print_separator(title=""):
    """Print a decorative separator"""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")
    else:
        print(f"{'='*80}")

def analyze_sample_dict(sample_dict, label="Sample"):
    """Analyze and print details of a sample dictionary (source or target)"""
    print(f"\n  {label} Structure:")
    print(f"  {'-'*76}")
    
    # Encoder inputs
    encoder_inputs = sample_dict['encoder_inputs']
    print(f"  encoder_inputs:")
    print(f"    - Type: {type(encoder_inputs)}")
    print(f"    - Shape: {encoder_inputs.shape}")
    print(f"    - Dtype: {encoder_inputs.dtype}")
    print(f"    - Min/Max values: {encoder_inputs.min():.4f} / {encoder_inputs.max():.4f}")
    print(f"    - Device: {encoder_inputs.device}")
    
    # HyenaDNA input IDs
    hyena_input_ids = sample_dict['hyena_input_ids']
    print(f"\n  hyena_input_ids:")
    print(f"    - Type: {type(hyena_input_ids)}")
    print(f"    - Shape: {hyena_input_ids.shape}")
    print(f"    - Dtype: {hyena_input_ids.dtype}")
    print(f"    - Unique token IDs: {torch.unique(hyena_input_ids).tolist()}")
    print(f"    - Device: {hyena_input_ids.device}")
    
    # Show first sequence's token IDs in detail
    first_seq_ids = hyena_input_ids[0]
    print(f"    - First sequence token IDs (first 5): {first_seq_ids[:100].tolist()}")
    print(f"    - First sequence token IDs (last 5):  {first_seq_ids[-100:].tolist()}")
    
    # HyenaDNA attention mask
    hyena_attention_mask = sample_dict['hyena_attention_mask']
    print(f"\n  hyena_attention_mask:")
    print(f"    - Type: {type(hyena_attention_mask)}")
    print(f"    - Shape: {hyena_attention_mask.shape}")
    print(f"    - Dtype: {hyena_attention_mask.dtype}")
    print(f"    - Non-zero positions (first sequence): {hyena_attention_mask[0].sum().item()}/{hyena_attention_mask[0].shape[0]}")
    print(f"    - Device: {hyena_attention_mask.device}")
    
    # Raw texts
    raw_texts = sample_dict['raw_texts']
    print(f"\n  raw_texts:")
    print(f"    - Type: {type(raw_texts)}")
    print(f"    - Number of sequences: {len(raw_texts)}")
    print(f"    - Sequence lengths: {[len(seq) for seq in raw_texts[:5]]} {'...' if len(raw_texts) > 5 else ''}")
    print(f"    - First sequence (first 80 chars): {raw_texts[0][:80]}...")
    
    # Metadata
    metadata_key = 'source_metadata' if 'source_metadata' in sample_dict else 'target_metadata'
    metadata = sample_dict[metadata_key]
    print(f"\n  {metadata_key}:")
    print(f"    - Type: {type(metadata)}")
    print(f"    - Value: {metadata}")
    print(f"    - Subject ID: {metadata[0]}")
    print(f"    - Timepoint: {metadata[1]}")


def main():
    print_separator("TCR Dataset Analysis")
    
    # Initialize the dataset
    print("\n1. Initializing TCRDataset...")
    try:
        dataset = TCRDataset(seed=42, set_size=32)
        print(f"   ✓ Dataset initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing dataset: {e}")
        sys.exit(1)
    
    # Basic dataset information
    print_separator("Dataset Overview")
    
    # Dataset length calculation
    num_repertoires = len(dataset.data)
    dataset_length = len(dataset)
    expected_length = num_repertoires * (num_repertoires - 1)
    
    print(f"\nDataset Length Information:")
    print(f"  Number of repertoires (n): {num_repertoires}")
    print(f"  Dataset length: {dataset_length}")
    print(f"  Formula: n × (n - 1) = {num_repertoires} × {num_repertoires - 1} = {expected_length}")
    print(f"  Explanation: All possible ordered pairs of repertoires (source → target)")
    print(f"               Each repertoire can be paired with any other repertoire")
    print(f"               excluding self-pairing (i ≠ j)")
    
    print(f"\nDataset Configuration:")
    print(f"  Set size (sequences per sample): {dataset.set_size}")
    print(f"  Max sequence length: {dataset.max_seq_length}")
    print(f"  DNA vocabulary size: {dataset.vocab_size}")
    print(f"  DNA vocabulary: {dataset.dna_vocab}")
    
    # HyenaDNA tokenizer information
    print(f"\nHyenaDNA Tokenizer:")
    print(f"  Tokenizer type: {type(dataset.hyena_tokenizer)}")
    print(f"  Model max length: {dataset.hyena_tokenizer.model_max_length}")
    
    # Get the token-to-id mapping
    if hasattr(dataset.hyena_tokenizer, 'get_vocab'):
        vocab = dataset.hyena_tokenizer.get_vocab()
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Token-to-ID mapping:")
        # Sort by ID for better readability
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            # Display special characters and tokens clearly
            token_display = repr(token) if token in ['\n', '\t', ' '] or len(token) == 0 else token
            print(f"    {token_display:20s} -> {token_id}")
    elif hasattr(dataset.hyena_tokenizer, 'vocab'):
        vocab = dataset.hyena_tokenizer.vocab
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Token-to-ID mapping:")
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        for token, token_id in sorted_vocab:
            token_display = repr(token) if token in ['\n', '\t', ' '] or len(token) == 0 else token
            print(f"    {token_display:20s} -> {token_id}")
    else:
        print(f"  Could not access vocabulary mapping")
    
    # Check for special tokens
    if hasattr(dataset.hyena_tokenizer, 'special_tokens_map'):
        print(f"\n  Special tokens:")
        for key, value in dataset.hyena_tokenizer.special_tokens_map.items():
            print(f"    {key}: {value}")
    
    # Print repertoire information
    print(f"\nRepertoire metadata:")
    for i, (subject_id, timepoint) in enumerate(dataset.metadata[:10]):
        num_sequences = len(dataset.data[i])
        print(f"  [{i}] Subject: {subject_id}, Timepoint: {timepoint}, "
              f"Total sequences: {num_sequences}")
    if len(dataset.metadata) > 10:
        print(f"  ... and {len(dataset.metadata) - 10} more repertoires")
    
    # Sample and analyze multiple elements
    num_samples = 1
    sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
    
    for sample_num, idx in enumerate(sample_indices, 1):
        print_separator(f"Sample #{sample_num} (Index: {idx})")
        
        try:
            item = dataset[idx]
            print(f"\n✓ Successfully retrieved item at index {idx}")
            
            # Top-level structure
            print(f"\nTop-level keys: {list(item.keys())}")
            
            # Analyze source samples
            print_separator(f"SOURCE SAMPLES (Sample #{sample_num})")
            analyze_sample_dict(item['source_samples'], label="Source")
            
            # Analyze target samples
            print_separator(f"TARGET SAMPLES (Sample #{sample_num})")
            analyze_sample_dict(item['target_samples'], label="Target")
            
            # Train predictor bool
            print(f"\n  train_predictor_bool:")
            print(f"    - Type: {type(item['train_predictor_bool'])}")
            print(f"    - Value: {item['train_predictor_bool']}")
            print(f"    - Meaning: {'Same patient, consecutive timepoints' if item['train_predictor_bool'] else 'Different patients or non-consecutive timepoints'}")
            
        except Exception as e:
            print(f"\n✗ Error retrieving item at index {idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Memory analysis
    print_separator("Memory Analysis")
    
    # Get a sample to calculate memory
    sample = dataset[0]
    
    def get_tensor_size_mb(tensor):
        """Calculate tensor size in MB"""
        return tensor.element_size() * tensor.nelement() / (1024 ** 2)
    
    source_encoder_mb = get_tensor_size_mb(sample['source_samples']['encoder_inputs'])
    source_hyena_ids_mb = get_tensor_size_mb(sample['source_samples']['hyena_input_ids'])
    source_hyena_mask_mb = get_tensor_size_mb(sample['source_samples']['hyena_attention_mask'])
    
    target_encoder_mb = get_tensor_size_mb(sample['target_samples']['encoder_inputs'])
    target_hyena_ids_mb = get_tensor_size_mb(sample['target_samples']['hyena_input_ids'])
    target_hyena_mask_mb = get_tensor_size_mb(sample['target_samples']['hyena_attention_mask'])
    
    total_mb = (source_encoder_mb + source_hyena_ids_mb + source_hyena_mask_mb + 
                target_encoder_mb + target_hyena_ids_mb + target_hyena_mask_mb)
    
    print(f"\nMemory per sample:")
    print(f"  Source samples:")
    print(f"    - encoder_inputs: {source_encoder_mb:.4f} MB")
    print(f"    - hyena_input_ids: {source_hyena_ids_mb:.4f} MB")
    print(f"    - hyena_attention_mask: {source_hyena_mask_mb:.4f} MB")
    print(f"  Target samples:")
    print(f"    - encoder_inputs: {target_encoder_mb:.4f} MB")
    print(f"    - hyena_input_ids: {target_hyena_ids_mb:.4f} MB")
    print(f"    - hyena_attention_mask: {target_hyena_mask_mb:.4f} MB")
    print(f"  Total per sample: {total_mb:.4f} MB")
    
    # Statistical summary
    print_separator("Statistical Summary")
    
    # Sample multiple items to get statistics
    print("\nSampling 10 random items for statistical analysis...")
    train_predictor_counts = {'True': 0, 'False': 0}
    sequence_lengths = []
    
    for i in range(min(10, len(dataset))):
        item = dataset[i]
        train_predictor_counts[str(item['train_predictor_bool'])] += 1
        
        # Get sequence lengths from raw texts
        for seq in item['source_samples']['raw_texts']:
            sequence_lengths.append(len(seq))
        for seq in item['target_samples']['raw_texts']:
            sequence_lengths.append(len(seq))

    
    print(f"\nTrain predictor bool distribution (from {sum(train_predictor_counts.values())} samples):")
    print(f"  True (consecutive timepoints): {train_predictor_counts['True']}")
    print(f"  False (non-consecutive): {train_predictor_counts['False']}")
    
    if sequence_lengths:
        sequence_lengths = np.array(sequence_lengths)
        print(f"\nSequence length statistics (from {len(sequence_lengths)} sequences):")
        print(f"  Min: {sequence_lengths.min()}")
        print(f"  Max: {sequence_lengths.max()}")
        print(f"  Mean: {sequence_lengths.mean():.2f}")
        print(f"  Median: {np.median(sequence_lengths):.2f}")
        print(f"  Std: {sequence_lengths.std():.2f}")
    
    print_separator("Analysis Complete")
    print()


if __name__ == "__main__":
    main()

