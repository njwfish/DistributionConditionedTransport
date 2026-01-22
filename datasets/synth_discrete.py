from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from typing import Optional
import gzip
import os
import logging
import time
from hydra.core.global_hydra import GlobalHydra
import hydra

logger = logging.getLogger(__name__)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class SyntheticProteinDataset(Dataset):
    """
    A synthetic where each source/target distribution
    consists of N identical copies of a random k-mer amino acid sequence.
    """
    
    def __init__(self,
                 data_dir: str = 'data/pfam',
                 data_file: str = "synthetic_pfam_tokenized_data.pt",
                 set_size: int = 16,
                 n_copies: int = 16,  # Number of identical copies per distribution
                 kmer_length: int = 4,  # Length of the random amino acid sequence
                 num_distributions: int = 4,  # Number of unique distributions
                 esm_name: str = 'facebook/esm2_t6_8M_UR50D',
                 progen_name: str = 'hugohrban/progen2-small',
                 max_length: int = 128,
                 seed: Optional[int] = 212121,
                 dataset_length: int = 1000,
                 tokenize: bool = False,
                 test_split: float = 0.25,
                 base_dir: Optional[str] = None,
                 start_line: int = 0,
                 lines_to_read: int = 10**12):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.data_file = data_file
        self.set_size = set_size
        self.n_copies = n_copies
        self.kmer_length = kmer_length
        self.num_distributions = num_distributions
        self.max_length = max_length
        self.dataset_length = dataset_length
        
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            self.base_dir = hydra.utils.get_original_cwd()
        else:
            self.base_dir = os.getcwd()
        if base_dir is not None:
            self.base_dir = base_dir
        
        # Ensure data directory exists
        os.makedirs(os.path.join(self.base_dir, self.data_dir), exist_ok=True)
        
        self.tokenized_data_file = os.path.join(self.base_dir, self.data_dir, self.data_file)
        
        # Initialize tokenizers
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'
        
        # Generate or load synthetic data
        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._generate_and_save_synthetic_data(test_split=test_split)
        
        self.data = torch.load(self.tokenized_data_file)
        logger.info(f"Loaded {len(self.data)} synthetic distributions from {self.tokenized_data_file}")

    def _generate_random_kmer(self) -> str:
        """Generate a random k-mer amino acid sequence."""
        return ''.join(random.choices(AMINO_ACIDS, k=self.kmer_length))

    def _generate_and_save_synthetic_data(self, test_split: float = 0.25):
        """Generate synthetic tokenized data with identical sequences per distribution and split train/test."""
        logger.info(f"Generating {self.num_distributions} synthetic distributions...")
        
        # Generate all unique kmers
        all_kmers = set()
        while len(all_kmers) < self.num_distributions:
            kmer = self._generate_random_kmer()
            all_kmers.add(kmer)
        
        all_kmers = list(all_kmers)
        
        # Split kmers into train and test sets (ensuring no overlap)
        np.random.shuffle(all_kmers)
        n_test = max(1, int(len(all_kmers) * test_split))
        test_kmers = set(all_kmers[:n_test])
        train_kmers = set(all_kmers[n_test:])
        
        logger.info(f'Split {len(all_kmers)} unique kmers into {len(train_kmers)} train and {len(test_kmers)} test')
        
        train_data = []
        test_data = []
        
        for i, kmer in enumerate(all_kmers):
            # Create n_copies of this k-mer
            seqs = [kmer] * self.n_copies
            
            # Tokenize all copies
            all_esm_input_ids = []
            all_esm_attention_mask = []
            all_progen_input_ids = []
            all_progen_attention_mask = []
            
            for seq in seqs:
                esm = self._tokenize_for_esm(seq)
                all_esm_input_ids.append(esm[0])
                all_esm_attention_mask.append(esm[1])
                
                pg2 = self._tokenize_for_progen(seq)
                all_progen_input_ids.append(pg2[0])
                all_progen_attention_mask.append(pg2[1])
            
            # Stack into tensors
            family_entry = {
                'samples': {
                    'esm_input_ids': torch.stack(all_esm_input_ids),
                    'esm_attention_mask': torch.stack(all_esm_attention_mask),
                    'progen_input_ids': torch.stack(all_progen_input_ids),
                    'progen_attention_mask': torch.stack(all_progen_attention_mask),
                },
                'pfam': f'SYNTH_{i:05d}',
                'raw_texts': seqs,
                'canonical_kmer': kmer,  # Store the canonical k-mer for reference
            }
            
            # Add to appropriate split
            if kmer in test_kmers:
                test_data.append(family_entry)
            else:
                train_data.append(family_entry)
        
        # Save train data to the main file
        torch.save(train_data, self.tokenized_data_file)
        logger.info(f"Train tokenized data ({len(train_data)} distributions) saved to {self.tokenized_data_file}")
        
        # Save test data to eval file
        eval_tokenized_data_file = self._get_eval_data_file()
        torch.save(test_data, eval_tokenized_data_file)
        logger.info(f"Test tokenized data ({len(test_data)} distributions) saved to {eval_tokenized_data_file}")

    def _get_eval_data_file(self):
        """Get the path to the eval data file based on the train data file."""
        dir_path = os.path.dirname(self.tokenized_data_file)
        base_name = os.path.basename(self.tokenized_data_file)
        return os.path.join(dir_path, 'eval_' + base_name)

    def _tokenize_for_esm(self, sequence):
        """Tokenize a protein sequence for ESM."""
        sequence = sequence.strip()
        tokens = self.esm_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True,
            return_tensors='pt'
        )
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _tokenize_for_progen(self, sequence):
        """Tokenize a protein sequence for Progen."""
        sequence = sequence.strip()
        bos_token = self.progen_tokenizer.bos_token
        eos_token = self.progen_tokenizer.eos_token
        
        if bos_token and not sequence.startswith(bos_token):
            sequence = bos_token + sequence
        if eos_token and not sequence.endswith(eos_token):
            sequence = sequence + eos_token
        
        tokens = self.progen_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=False,
            return_tensors='pt'
        )
        return tokens.input_ids[0], tokens.attention_mask[0]

    def __len__(self):
        return self.dataset_length
    
    def __getitem__(self, idx):
        n = len(self.data)
        source_idx = np.random.randint(0, n)
        target_idx = np.random.randint(0, n)
        item_source = self.data[source_idx]
        item_target = self.data[target_idx]
        
        # Since all copies are identical, we can just select any set_size of them
        source_subset_indices = np.random.choice(
            len(item_source['samples']['esm_input_ids']), 
            size=min(self.set_size, len(item_source['samples']['esm_input_ids'])), 
            replace=False
        )
        target_subset_indices = np.random.choice(
            len(item_target['samples']['esm_input_ids']), 
            size=min(self.set_size, len(item_target['samples']['esm_input_ids'])), 
            replace=False
        )
        
        esm_input_ids_source = item_source['samples']['esm_input_ids'][source_subset_indices]
        esm_attention_mask_source = item_source['samples']['esm_attention_mask'][source_subset_indices]
        progen_input_ids_source = item_source['samples']['progen_input_ids'][source_subset_indices]
        progen_attention_mask_source = item_source['samples']['progen_attention_mask'][source_subset_indices]

        esm_input_ids_target = item_target['samples']['esm_input_ids'][target_subset_indices]
        esm_attention_mask_target = item_target['samples']['esm_attention_mask'][target_subset_indices]
        progen_input_ids_target = item_target['samples']['progen_input_ids'][target_subset_indices]
        progen_attention_mask_target = item_target['samples']['progen_attention_mask'][target_subset_indices]

        return {
            'source_samples': {
                'esm_input_ids': esm_input_ids_source,
                'esm_attention_mask': esm_attention_mask_source,
                'progen_input_ids': progen_input_ids_source,
                'progen_attention_mask': progen_attention_mask_source,
                'pfam': item_source['pfam'],
                'raw_texts': [item_source['raw_texts'][i] for i in source_subset_indices],
                'canonical_kmer': item_source['canonical_kmer'],
            },
            'target_samples': {
                'esm_input_ids': esm_input_ids_target,
                'esm_attention_mask': esm_attention_mask_target,
                'progen_input_ids': progen_input_ids_target,
                'progen_attention_mask': progen_attention_mask_target,
                'pfam': item_target['pfam'],
                'raw_texts': [item_target['raw_texts'][i] for i in target_subset_indices],
                'canonical_kmer': item_target['canonical_kmer'],
            },
            'source_idx': source_idx,
            'target_idx': target_idx,
            'train_predictor_bool': self.get_train_predictor_bool(source_idx, target_idx)
        }
    
    def get_train_predictor_bool(self, source_idx, target_idx):
        return False