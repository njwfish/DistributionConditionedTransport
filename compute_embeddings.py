#!/usr/bin/env python
"""
Compute ESM2 embeddings for tokenized TCR sequences.
Run with SLURM job array: each task processes one tokenized part file.

Usage:
    python compute_embeddings.py --task_id 0 --num_tasks 10
    
Or via SLURM:
    sbatch run_embeddings.sh  (uses SLURM_ARRAY_TASK_ID)
"""

import argparse
import os
import torch
import logging
from transformers import AutoModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_mean_pooled_embeddings(model, input_ids, attention_mask, batch_size=64, device='cuda'):
    """
    Compute mean-pooled embeddings from ESM2 last layer.
    
    Args:
        model: ESM2 model
        input_ids: tensor of shape (N, seq_len)
        attention_mask: tensor of shape (N, seq_len)
        batch_size: batch size for inference
        device: device to run on
        
    Returns:
        embeddings: tensor of shape (N, hidden_dim)
    """
    model.eval()
    all_embeddings = []
    
    n_samples = input_ids.shape[0]
    
    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing embeddings"):
            batch_input_ids = input_ids[i:i+batch_size].to(device)
            batch_attention_mask = attention_mask[i:i+batch_size].to(device)
            
            # Get last hidden state
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            
            # Last layer hidden states: (batch, seq_len, hidden_dim)
            last_hidden = outputs.last_hidden_state
            
            # Mean pool over sequence length (excluding padding)
            # Expand attention mask for broadcasting: (batch, seq_len, 1)
            mask_expanded = batch_attention_mask.unsqueeze(-1).float()
            
            # Sum of hidden states weighted by mask
            sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
            
            # Count of non-padding tokens
            token_counts = mask_expanded.sum(dim=1).clamp(min=1)  # avoid div by zero
            
            # Mean pooled embeddings: (batch, hidden_dim)
            mean_pooled = sum_hidden / token_counts
            
            all_embeddings.append(mean_pooled.cpu())
    
    return torch.cat(all_embeddings, dim=0)


def main():
    parser = argparse.ArgumentParser(description='Compute ESM2 embeddings for TCR sequences')
    parser.add_argument('--task_id', type=int, default=None,
                        help='Task ID (0-indexed). If not provided, uses SLURM_ARRAY_TASK_ID')
    parser.add_argument('--num_tasks', type=int, default=10,
                        help='Total number of parallel tasks')
    parser.add_argument('--esm_name', type=str, default='facebook/esm2_t6_8M_UR50D')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for inference')
    parser.add_argument('--input_dir', type=str, default='data/tcr/tokenized_parts')
    parser.add_argument('--output_dir', type=str, default='data/tcr/embedding_parts')
    args = parser.parse_args()
    
    # Get task ID from argument or SLURM environment
    task_id = args.task_id
    if task_id is None:
        task_id = int(os.environ.get('SLURM_ARRAY_TASK_ID', 0))
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Input and output paths
    input_dir = os.path.join(base_dir, args.input_dir)
    output_dir = os.path.join(base_dir, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    input_file = os.path.join(input_dir, f'tokenized_part_{task_id:03d}.pt')
    output_file = os.path.join(output_dir, f'embeddings_part_{task_id:03d}.pt')
    
    if not os.path.exists(input_file):
        logger.warning(f'Input file not found: {input_file}')
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model
    logger.info(f'Loading ESM2 model: {args.esm_name}')
    model = AutoModel.from_pretrained(args.esm_name, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    
    # Load tokenized data
    logger.info(f'Loading tokenized data from {input_file}')
    data = torch.load(input_file)
    
    # Process each repertoire
    results = []
    
    for rep_idx, repertoire in enumerate(data['data']):
        logger.info(f'Processing repertoire {rep_idx + 1}/{len(data["data"])}')
        
        input_ids = repertoire['samples']['esm_input_ids']
        attention_mask = repertoire['samples']['esm_attention_mask']
        
        logger.info(f'  Sequences: {input_ids.shape[0]}')
        
        # Compute embeddings
        embeddings = compute_mean_pooled_embeddings(
            model, input_ids, attention_mask,
            batch_size=args.batch_size, device=device
        )
        
        results.append({
            'embeddings': embeddings,  # (N, hidden_dim)
            'raw_texts': repertoire['raw_texts'],
        })
        
        logger.info(f'  Embedding shape: {embeddings.shape}')
    
    # Save results with metadata
    output_data = {
        'embeddings': results,
        'metadata': data['metadata'],
        'model_name': args.esm_name,
    }
    
    torch.save(output_data, output_file)
    logger.info(f'Saved embeddings to {output_file}')
    
    # Summary
    total_sequences = sum(r['embeddings'].shape[0] for r in results)
    logger.info(f'Task {task_id} complete: {len(results)} repertoires, {total_sequences:,} sequences')


if __name__ == '__main__':
    main()
