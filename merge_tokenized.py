#!/usr/bin/env python
"""
Merge tokenized parts into a single file.
Run after all parallel tokenization tasks complete.

Usage:
    python merge_tokenized.py
"""

import os
import torch
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parts_dir = os.path.join(base_dir, 'data/tcr/tokenized_parts')
    output_file = os.path.join(base_dir, 'data/tcr/tcr_tokenized_data.pt')
    
    # Find all part files
    part_files = sorted(glob.glob(os.path.join(parts_dir, 'tokenized_part_*.pt')))
    
    if not part_files:
        logger.error(f'No part files found in {parts_dir}')
        return
    
    logger.info(f'Found {len(part_files)} part files')
    
    # Merge all parts
    all_data = []
    all_metadata = []
    
    for part_file in part_files:
        logger.info(f'Loading {part_file}')
        part = torch.load(part_file)
        all_data.extend(part['data'])
        all_metadata.extend(part['metadata'])
    
    logger.info(f'Total repertoires: {len(all_data)}')
    
    # Count total sequences
    total_sequences = sum(len(d['raw_texts']) for d in all_data)
    logger.info(f'Total sequences: {total_sequences:,}')
    
    # Save merged file
    torch.save({'data': all_data, 'metadata': all_metadata}, output_file)
    logger.info(f'Saved merged data to {output_file}')
    
    # Optionally clean up part files
    # for part_file in part_files:
    #     os.remove(part_file)
    # logger.info('Cleaned up part files')


if __name__ == '__main__':
    main()
