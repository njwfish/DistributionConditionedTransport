#!/usr/bin/env python
"""
Merge embedding parts into a single file.
Run after all parallel embedding tasks complete.

Usage:
    python merge_embeddings.py
"""

import os
import torch
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parts_dir = os.path.join(base_dir, 'data/tcr/embedding_parts')
    output_file = os.path.join(base_dir, 'data/tcr/tcr_esm_embeddings.pt')
    
    # Find all part files
    part_files = sorted(glob.glob(os.path.join(parts_dir, 'embeddings_part_*.pt')))
    
    if not part_files:
        logger.error(f'No part files found in {parts_dir}')
        return
    
    logger.info(f'Found {len(part_files)} part files')
    
    # Merge all parts
    all_embeddings = []
    all_metadata = []
    model_name = None
    
    for part_file in part_files:
        logger.info(f'Loading {part_file}')
        part = torch.load(part_file)
        all_embeddings.extend(part['embeddings'])
        all_metadata.extend(part['metadata'])
        if model_name is None:
            model_name = part.get('model_name', 'unknown')
    
    logger.info(f'Total repertoires: {len(all_embeddings)}')
    
    # Count total sequences and get embedding dim
    total_sequences = sum(e['embeddings'].shape[0] for e in all_embeddings)
    embedding_dim = all_embeddings[0]['embeddings'].shape[1] if all_embeddings else 0
    
    logger.info(f'Total sequences: {total_sequences:,}')
    logger.info(f'Embedding dimension: {embedding_dim}')
    
    # Save merged file
    output_data = {
        'embeddings': all_embeddings,
        'metadata': all_metadata,
        'model_name': model_name,
        'embedding_dim': embedding_dim,
        'total_sequences': total_sequences,
    }
    
    torch.save(output_data, output_file)
    logger.info(f'Saved merged embeddings to {output_file}')


if __name__ == '__main__':
    main()
