from transformers import AutoModelForCausalLM, AutoTokenizer
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
                 set_size: int = 10,
                 esm_name: str = 'facebook/esm2_t6_8M_UR50D',
                 progen_name: str = 'hugohrban/progen2-medium',
                 max_length: int = 1200,
                 seed: Optional[int] = 212121,
                 tokenize: bool = False,
                 lines_to_read: int = 10**8,
                 max_sets_per_fam: int = 10,
                 include_location: bool = False,
                 max_draws_per_epoch: int = 10000):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.set_size = set_size
        self.max_length = max_length
        self.max_sets_per_fam = max_sets_per_fam
        self.max_draws_per_epoch = max_draws_per_epoch
        self.include_location = include_location
        
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'

        self.tokenized_data_file = f'{self.data_dir}/virus_tokenized_data.pt'
        self.index_pairs = np.array([(i, j) for i in range(self.data.shape[0]) for j in range(self.data.shape[0]) if i != j])

        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data(lines_to_read=lines_to_read)
        self.data = torch.load(self.tokenized_data_file)

    def _tokenize_data(self, lines_to_read=10**8):

        fn = self.data_dir+'/spikeprot0430.fasta'
        seqs_by_monthloc = defaultdict(list)
        max_per_monthloc = self.set_size  # cap per group

        logger.info('building dict')

        record_iterator = SeqIO.parse(fn, "fasta")

        for _ in tqdm(range(lines_to_read)):
            try:
                record = next(record_iterator)
                fields = record.description.split("|")
                (gene, isolate, date, iso_id, passage,
                type_loc, host, o_lab, s_lab,
                submitter, location) = (fields + ["?"] * 11)[:11]

                virus_type, state = type_loc.split("^^") if "^^" in type_loc else (type_loc, "?")

                if date[5:7] != '00' and date[-2:] != '00' and date[4] == '-':
                    if self.include_location:
                        key = date[:7] + '-' + location  # yyyy-mm-location
                    else:
                        key = date[:7]
                    if len(seqs_by_monthloc[key]) < max_per_monthloc:
                        seqs_by_monthloc[key].append(str(record.seq))
            except:
                continue

        tokenized_data = []

        for timeloc, seqs in tqdm(seqs_by_monthloc.items()):
            if len(seqs) != self.set_size:
                continue
            
            print("LEN",len(seqs))
            n_sets = min(len(seqs) // self.set_size, self.max_sets_per_fam)
            np.random.shuffle(seqs)

            for i in range(n_sets):
                batch = seqs[i*self.set_size : (i+1)*self.set_size]

                esm_input_ids = []
                esm_attention_mask = []
                progen_input_ids = []
                progen_attention_mask = []
                texts = []

                for seq in batch:

                    pg2 = self._tokenize_for_progen(seq)
                    progen_input_ids.append(pg2[0])
                    progen_attention_mask.append(pg2[1])
                    esm = self._tokenize_for_esm(seq)
                    esm_input_ids.append(esm[0])
                    esm_attention_mask.append(esm[1])
                    texts.append(seq[:self.max_length]) # note truncated


                esm_input_ids = torch.stack(esm_input_ids)
                esm_attention_mask = torch.stack(esm_attention_mask)
                progen_input_ids = torch.stack(progen_input_ids)
                progen_attention_mask = torch.stack(progen_attention_mask)

                tokenized_data.append({
                    'samples' : {
                    'esm_input_ids': esm_input_ids,
                    'esm_attention_mask': esm_attention_mask,
                    'progen_input_ids': progen_input_ids,
                    'progen_attention_mask': progen_attention_mask,},
                    'time-loc': timeloc,
                    'raw_texts': texts
                })
        torch.save(tokenized_data, self.tokenized_data_file)
        logger.info(f"tokenized data saved to {self.tokenized_data_file}")

    
    def _tokenize_for_esm(self, sequence):
        """
        Tokenize a protein sequence for ESM.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # ESM tokenizer requires starting with <cls> token
        # Ensure the sequence is not modified with extra spaces or newlines
        sequence = sequence.strip()
        
        # Tokenize with appropriate settings and explicitly add special tokens
        tokens = self.esm_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=True,  # This will add the CLS token
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _tokenize_for_progen(self, sequence):
        """
        Tokenize a protein sequence for Progen.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Clean the sequence
        sequence = sequence.strip()
        
        # Since the tokenizer isn't automatically adding special tokens,
        # we'll manually add BOS and EOS tokens
        bos_token = self.progen_tokenizer.bos_token
        eos_token = self.progen_tokenizer.eos_token
        
        # Ensure sequence starts with BOS and ends with EOS
        if bos_token and not sequence.startswith(bos_token):
            sequence = bos_token + sequence
        
        if eos_token and not sequence.endswith(eos_token):
            sequence = sequence + eos_token
        
        # Tokenize with appropriate settings
        # Set add_special_tokens=False since we've manually added them
        tokens = self.progen_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length,
            add_special_tokens=False,  # Don't add again since we did it manually
            return_tensors='pt'
        )
        
        # Log the first sequence's token IDs for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Progen tokenized sequence: {tokens.input_ids[0]}")
            logger.debug(f"Progen BOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(bos_token)}")
            logger.debug(f"Progen EOS token ID: {self.progen_tokenizer.convert_tokens_to_ids(eos_token)}")
        
        return tokens.input_ids[0], tokens.attention_mask[0]

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
        try:
            return datetime.strptime(date_str, "%Y-%m")
        except ValueError:
            logger.warning(f"Could not parse date from time_loc: {time_loc_str}")
            return None

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
        """
        date1 = self._parse_time_loc(time_loc_1)
        date2 = self._parse_time_loc(time_loc_2)
        
        if date1 is None or date2 is None:
            return None
        
        # Calculate month difference (target - source)
        month_diff = (date2.year - date1.year) * 12 + (date2.month - date1.month)
        return month_diff

    def __len__(self):
        # TODO: not sure how to do this correctly since we want to pair subsets of the data with each other at random. For now, just setting a shorter length.
        return self.max_draws_per_epoch
    
    def __getitem__(self, idx):
        source_idx, target_idx = np.random.choice(np.arange(len(self.data)), size=2, replace=False)

        item_source = self.data[source_idx]
        item_target = self.data[target_idx]

        esm_input_ids_source = item_source['samples']['esm_input_ids']
        esm_attention_mask_source = item_source['samples']['esm_attention_mask']
        progen_input_ids_source = item_source['samples']['progen_input_ids']
        progen_attention_mask_source = item_source['samples']['progen_attention_mask']
        
        esm_input_ids_target = item_target['samples']['esm_input_ids']
        esm_attention_mask_target = item_target['samples']['esm_attention_mask']
        progen_input_ids_target = item_target['samples']['progen_input_ids']
        progen_attention_mask_target = item_target['samples']['progen_attention_mask']

        # Calculate month difference between source and target times
        month_difference = self._calculate_month_difference(
            item_source['time-loc'], 
            item_target['time-loc']
        )
        
        return { 'samples_source' : {
            'esm_input_ids_source': esm_input_ids_source,
            'esm_attention_mask_source': esm_attention_mask_source,
            'progen_input_ids_source': progen_input_ids_source,
            'progen_attention_mask_source': progen_attention_mask_source,
            },
            'samples_target' : {
                'esm_input_ids_target': esm_input_ids_target,
                'esm_attention_mask_target': esm_attention_mask_target,
                'progen_input_ids_target': progen_input_ids_target,
                'progen_attention_mask_target': progen_attention_mask_target,
            },
            'time_source': item_source['time-loc'],
            'time_target': item_target['time-loc'],
            'dt': month_difference,
            'raw_texts_source': tuple(item_source['raw_texts']),
            'raw_texts_target': tuple(item_target['raw_texts'])
        }