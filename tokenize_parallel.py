#!/usr/bin/env python
"""
Parallel tokenization script for TCR data.
Run with SLURM job array: each task tokenizes a subset of repertoires.

Usage:
    python tokenize_parallel.py --task_id 0 --num_tasks 10
    
Or via SLURM:
    sbatch run_tokenize.sh  (uses SLURM_ARRAY_TASK_ID)
"""

import argparse
import os
import torch
import pandas as pd
import logging
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# The 20 standard amino acids
STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def is_valid_sequence(seq: str) -> bool:
    """Check if sequence contains only standard amino acids."""
    return all(aa in STANDARD_AMINO_ACIDS for aa in seq.upper())


def tokenize_repertoire(row, base_dir, esm_tokenizer, progen_tokenizer, max_length, set_size):
    """Tokenize a single repertoire."""
    subject_id = row['subject_id']
    timepoint = row['timepoint']
    
    # Construct path to full_data_unit.tsv
    data_unit_path = os.path.join(
        base_dir, 
        'tcr_dataset', 
        'tcr_data', 
        f'subject={subject_id}', 
        f'time={timepoint}', 
        'full_data_unit.tsv'
    )
    
    if not os.path.exists(data_unit_path):
        logger.warning(f'File not found: {data_unit_path}')
        return None, None
    
    # Load the data unit and extract sequences
    data_unit = pd.read_csv(data_unit_path, sep='\t')
    raw_sequences = data_unit['junction_aa'].tolist()
    
    # Filter sequences: keep only those with standard amino acids
    sequences = [seq for seq in raw_sequences if isinstance(seq, str) and is_valid_sequence(seq)]
    
    n_filtered = len(raw_sequences) - len(sequences)
    if n_filtered > 0:
        logger.info(f'Repertoire {subject_id}/time={timepoint}: filtered {n_filtered} sequences with non-standard amino acids')
    
    # Skip repertoires that don't have enough sequences
    if len(sequences) < set_size:
        logger.warning(f'Repertoire {subject_id}/time={timepoint}: only {len(sequences)} valid sequences (< set_size={set_size}), skipping')
        return None, None
    
    # Tokenize all sequences for this repertoire
    all_esm_input_ids = []
    all_esm_attention_mask = []
    all_progen_input_ids = []
    all_progen_attention_mask = []
    all_texts = []
    
    for seq in sequences:
        # ESM tokenization
        esm_tokens = esm_tokenizer(
            seq.strip(), 
            padding='max_length', 
            truncation=True, 
            max_length=max_length,
            add_special_tokens=True,
            return_tensors='pt'
        )
        all_esm_input_ids.append(esm_tokens.input_ids[0])
        all_esm_attention_mask.append(esm_tokens.attention_mask[0])
        
        # Progen tokenization
        progen_seq = progen_tokenizer.bos_token + seq.strip() + progen_tokenizer.eos_token
        progen_tokens = progen_tokenizer(
            progen_seq, 
            padding='max_length', 
            truncation=True, 
            max_length=max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )
        all_progen_input_ids.append(progen_tokens.input_ids[0])
        all_progen_attention_mask.append(progen_tokens.attention_mask[0])
        
        all_texts.append(seq[:max_length])
    
    # Stack into tensors
    tokenized_data = {
        'samples': {
            'esm_input_ids': torch.stack(all_esm_input_ids),
            'esm_attention_mask': torch.stack(all_esm_attention_mask),
            'progen_input_ids': torch.stack(all_progen_input_ids),
            'progen_attention_mask': torch.stack(all_progen_attention_mask),
        },
        'raw_texts': all_texts,
    }
    metadata = (subject_id, timepoint)
    
    logger.info(f'Tokenized repertoire {subject_id}/time={timepoint}: {len(sequences)} sequences')
    return tokenized_data, metadata


def main():
    parser = argparse.ArgumentParser(description='Parallel TCR tokenization')
    parser.add_argument('--task_id', type=int, default=None, 
                        help='Task ID (0-indexed). If not provided, uses SLURM_ARRAY_TASK_ID')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='Total number of parallel tasks')
    parser.add_argument('--max_length', type=int, default=18,
                        help='Max sequence length for tokenization')
    parser.add_argument('--set_size', type=int, default=1024,
                        help='Minimum sequences required per repertoire')
    parser.add_argument('--esm_name', type=str, default='facebook/esm2_t6_8M_UR50D')
    parser.add_argument('--progen_name', type=str, default='hugohrban/progen2-small')
    parser.add_argument('--output_dir', type=str, default='data/tcr/tokenized_parts')
    args = parser.parse_args()
    
    # Get task ID from argument or SLURM environment
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load repertoire index
    repertoire_index_path = os.path.join(base_dir, 'tcr_dataset', 'repertoire_index.tsv')
    repertoire_index = pd.read_csv(repertoire_index_path, sep='\t')
    
    # Determine which repertoires this task should process
    total_repertoires = len(repertoire_index)
    repertoires_per_task = (total_repertoires + args.num_tasks - 1) // args.num_tasks
    start_idx = task_id * repertoires_per_task
    end_idx = min(start_idx + repertoires_per_task, total_repertoires)
    
    logger.info(f'Task {task_id}/{args.num_tasks}: processing repertoires {start_idx} to {end_idx-1}')
    
    if start_idx >= total_repertoires:
        logger.info(f'Task {task_id}: no repertoires to process')
        return
    
    # Initialize tokenizers
    logger.info('Loading tokenizers...')
    esm_tokenizer = AutoTokenizer.from_pretrained(args.esm_name, trust_remote_code=True)
    progen_tokenizer = AutoTokenizer.from_pretrained(args.progen_name, trust_remote_code=True)
    progen_tokenizer.pad_token = '<|pad|>'
    progen_tokenizer.bos_token = '<|bos|>'
    progen_tokenizer.eos_token = '<|eos|>'
    
    # Process assigned repertoires
    tokenized_data = []
    metadata = []
    
    for idx in range(start_idx, end_idx):
        row = repertoire_index.iloc[idx]
        data, meta = tokenize_repertoire(
            row, base_dir, esm_tokenizer, progen_tokenizer, 
            args.max_length, args.set_size
        )
        if data is not None:
            tokenized_data.append(data)
            metadata.append(meta)
    
    # Save this task's output
    output_file = os.path.join(output_dir, f'tokenized_part_{task_id:03d}.pt')
    torch.save({'data': tokenized_data, 'metadata': metadata}, output_file)
    logger.info(f'Task {task_id}: saved {len(tokenized_data)} repertoires to {output_file}')


if __name__ == '__main__':
    main()
