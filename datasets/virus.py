from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.hf_local import resolve_local_or_repo
from torch.utils.data import Dataset
import torch
import numpy as np
import random
from typing import Optional
import os
import logging
from Bio import SeqIO
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

logger = logging.getLogger(__name__)

class ViralDataset(Dataset):
    def __init__(self,
                 data_dir: str = 'data/spikeprot0430',
                 data_file: str = 'virus_tokenized_data_for_tde.pt',
                 set_size: int = 10,
                 esm_name: str = 'facebook/esm2_t6_8M_UR50D',
                 progen_name: str = 'hugohrban/progen2-medium',
                 max_length: int = 1200,
                 seed: Optional[int] = 212121,
                 tokenize: bool = False,
                 lines_to_read: int = 10**8,
                 max_sets_per_fam: int = 10,
                 include_location: bool = False,
                 max_draws_per_epoch: int = 1000):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.data_file = data_file
        self.set_size = set_size
        self.max_length = max_length
        self.max_sets_per_fam = max_sets_per_fam
        self.max_draws_per_epoch = max_draws_per_epoch
        self.include_location = include_location
        
        self.esm_tokenizer = AutoTokenizer.from_pretrained(resolve_local_or_repo(esm_name), trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(resolve_local_or_repo(progen_name), trust_remote_code=True)
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'

        self.tokenized_data_file = f'{self.data_dir}/{self.data_file}' #virus_tokenized_data_for_tde.pt'

        # NOTE: Important, hold out last time point for forecasting benchmarks.
        self.data = torch.load(self.tokenized_data_file)[:-1]
        #with open("auxillary_log.log", "a") as f:
        #    f.write(f"len(self.data): {len(self.data)}\n")
        # Build index pairs after data is loaded
        self.index_pairs = np.array(
            [
                (i, j)
                for i in range(len(self.data))
                for j in range(len(self.data))
                if i != j
            ]
        )

    
    def _parse_time_loc(self, time_loc_str):
        """
        Parse time-loc string to extract yyyy-mm date portion.
        
        Args:
            time_loc_str: String in format "yyyy-mm" or "yyyy-mm-location"
            
        Returns:
            datetime object representing the year and month
        """
        # Extract just the yyyy-mm portion (first 7 characters)
        date_str = time_loc_str[:7]

        return datetime.strptime(date_str, "%Y-%m")


    def _calculate_month_difference(self, time_loc_1, time_loc_2):
        """
        Calculate the difference in months between two time-loc strings.
        
        Args:
            time_loc_1: First time-loc string (source)
            time_loc_2: Second time-loc string (target)
            
        Returns:
            Integer representing the difference in months (target - source)
            Positive values mean target is later than source
            Negative values mean target is earlier than source
            Returns 0 if either date cannot be parsed
        """
        date1 = self._parse_time_loc(time_loc_1)
        date2 = self._parse_time_loc(time_loc_2)
        
        if date1 is None or date2 is None:
            logger.warning(f"Date parsing failed for time_loc_1='{time_loc_1}' or time_loc_2='{time_loc_2}', returning 0 month difference")
            return 0  # Return 0 instead of None to avoid DataLoader collation errors
        
        # Calculate month difference (target - source)
        month_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
        return month_diff

    def __len__(self):
        # TODO: not sure how to do this correctly since we want to pair subsets of the data with each other at random. For now, just doing all the pairs.
        # TODO: it is probably better in the long run to sample random pairs rather than doing all pairs, 
        ### but for that we need to change the sampler setup to correctly weigh pairs.
        n = len(self.data)
        return n * (n - 1)
    
    def __getitem__(self, idx):
        # TODO: need to change this if you ever want to sample pairs from identical time-points.
        # TODO: make sure this is correct.
        #n = len(self.data)
        #i = idx // (n - 1)
        #j = idx % (n - 1)
        #if j >= i:
        #    j += 1
        #source_idx, target_idx = i, j
        source_idx, target_idx = self.index_pairs[idx]
        
        item_source = self.data[source_idx]
        item_target = self.data[target_idx]
        
        # TODO: the way I'm doing things here might be a bit inefficient.
        esm_input_ids_source = item_source['samples']['esm_input_ids']
        esm_attention_mask_source = item_source['samples']['esm_attention_mask']
        progen_input_ids_source = item_source['samples']['progen_input_ids']
        progen_attention_mask_source = item_source['samples']['progen_attention_mask']
        
        esm_input_ids_target = item_target['samples']['esm_input_ids']
        esm_attention_mask_target = item_target['samples']['esm_attention_mask']
        progen_input_ids_target = item_target['samples']['progen_input_ids']
        progen_attention_mask_target = item_target['samples']['progen_attention_mask']

        # TODO: implement optimal pairing here.
        subset_indices_source = np.random.choice(esm_input_ids_source.shape[0], size=self.set_size, replace=False)

        esm_input_ids_source = esm_input_ids_source[subset_indices_source]
        esm_attention_mask_source = esm_attention_mask_source[subset_indices_source]
        progen_input_ids_source = progen_input_ids_source[subset_indices_source]
        progen_attention_mask_source = progen_attention_mask_source[subset_indices_source]

        subset_indices_target = np.random.choice(esm_input_ids_target.shape[0], size=self.set_size, replace=False)

        esm_input_ids_target = esm_input_ids_target[subset_indices_target]
        esm_attention_mask_target = esm_attention_mask_target[subset_indices_target]
        progen_input_ids_target = progen_input_ids_target[subset_indices_target]
        progen_attention_mask_target = progen_attention_mask_target[subset_indices_target]

        # Calculate month difference between source and target times
        month_difference = self._calculate_month_difference(
            item_source['time'], 
            item_target['time']
        )
        
        # TODO: in it's current version sometimes the source and target samples seem to have different batch sizes... not sure right now why, just had a weird bug.
        return { 'source_samples' : {
            'esm_input_ids': esm_input_ids_source,
            'esm_attention_mask': esm_attention_mask_source,
            'progen_input_ids': progen_input_ids_source,
            'progen_attention_mask': progen_attention_mask_source,
            },
            'target_samples' : {
                'esm_input_ids': esm_input_ids_target,
                'esm_attention_mask': esm_attention_mask_target,
                'progen_input_ids': progen_input_ids_target,
                'progen_attention_mask': progen_attention_mask_target,
            },
            'source_idx': source_idx,
            'target_idx': target_idx,
            'd': month_difference, 
            }