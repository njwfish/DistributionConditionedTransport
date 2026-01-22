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

class PfamDataset(Dataset):
    def __init__(self,
                data_dir: str = 'data/pfam',
                data_file: str = "pfam_tokenized_data.pt",
                raw_data_file: str = "Pfam-A.fasta.gz",
                set_size: int = 16,
                esm_name: str = 'facebook/esm2_t6_8M_UR50D',
                progen_name: str = 'hugohrban/progen2-small',
                max_length: int = 512,
                seed: Optional[int] = 212121,
                tokenize: bool = False,
                start_line: int = 0,
                lines_to_read: int = 10**12,
                max_seqs_per_fam: int = 16,
                max_fam_size: int = 100,
                max_pfams: Optional[int] = None,
                test_split: float = 0.25,
                base_dir: Optional[str] = None):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.data_file = data_file
        self.raw_data_file = raw_data_file
        self.set_size = set_size
        self.max_length = max_length
        self.max_seqs_per_fam = max_seqs_per_fam
        self.max_fam_size = max_fam_size
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)

        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            self.base_dir = hydra.utils.get_original_cwd()
        else:
            self.base_dir = os.getcwd()
        if base_dir is not None:
            self.base_dir = base_dir
            
        self.tokenized_data_file = os.path.join(self.base_dir, self.data_dir, self.data_file)

        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data(start_line=start_line, lines_to_read=lines_to_read, max_pfams=max_pfams, test_split=test_split)
        self.data = torch.load(self.tokenized_data_file)


    def _tokenize_data(self, start_line=0, lines_to_read=10**10, max_pfams=None, test_split=0.25):
        f = gzip.open(os.path.join(self.base_dir, self.data_dir, self.raw_data_file), 'rt')
        d = {}
        i = 0
        validated_count = 0  # Count of families confirmed to have enough sequences
        current_fam = None   # Family currently being read

        logger.info(f'building pfam dict from line {start_line} to line {start_line + lines_to_read}')
        # collect seqs per pfam, filtering on-the-fly (pfams are grouped in the file)
        for line in f:
            # Skip lines until we reach start_line
            if i < start_line:
                i += 1
                continue
            
            # Stop reading after we've read lines_to_read lines from start_line
            if i >= start_line + lines_to_read:
                break
                
            if line.startswith('>'):
                new_fam = line.split()[-1].split(';')[0]
                
                # When we encounter a new family, validate the previous one
                if current_fam is not None and current_fam != new_fam:
                    if len(d[current_fam]) >= self.set_size and len(d[current_fam]) <= self.max_fam_size:
                        validated_count += 1
                        # Only stop when we have enough VALIDATED families
                        if max_pfams is not None and validated_count >= max_pfams:
                            break
                    else:
                        logger.info(f"Removing family {current_fam}: only {len(d[current_fam])} sequences (should be between {self.set_size} and {self.max_fam_size})")
                        del d[current_fam]
                
                # Start collecting the new family (if we haven't seen it before)
                if new_fam not in d:
                    d[new_fam] = []
                current_fam = new_fam
            elif current_fam is not None:
                # Only append sequence if we've seen a header
                d[current_fam].append(line.strip())
            # else: skip orphan sequence lines before the first header
            i += 1

        # Validate the last family (wasn't validated in the loop if we hit the line limit)
        if current_fam is not None and current_fam in d and len(d[current_fam]) < self.set_size:
            logger.info(f"Removing family {current_fam}: only {len(d[current_fam])} sequences (< set_size={self.set_size})")
            del d[current_fam]

        f.close()

        # Split families into train and test sets
        all_fams = list(d.keys())
        np.random.shuffle(all_fams)
        n_test = max(1, int(len(all_fams) * test_split))
        test_fams = set(all_fams[:n_test])
        train_fams = set(all_fams[n_test:])
        
        logger.info(f'Split {len(all_fams)} families into {len(train_fams)} train and {len(test_fams)} test')

        train_tokenized_data = []
        test_tokenized_data = []
        
        logger.info('tokenizing pfam data')
        for fam, seqs in d.items():
            # Shuffle and limit to max_seqs_per_fam
            np.random.shuffle(seqs)
            seqs = seqs[:self.max_seqs_per_fam]
            
            # Accumulate all tokenized sequences for this family
            all_esm_input_ids = []
            all_esm_attention_mask = []
            all_progen_input_ids = []
            all_progen_attention_mask = []
            all_texts = []
            
            # Process in batches for efficiency
            for i in range(0, len(seqs), self.set_size):
                batch = seqs[i:i + self.set_size]
                for seq in batch:
                    pg2 = self._tokenize_for_progen(seq)
                    all_progen_input_ids.append(pg2[0])
                    all_progen_attention_mask.append(pg2[1])
                    esm = self._tokenize_for_esm(seq)
                    all_esm_input_ids.append(esm[0])
                    all_esm_attention_mask.append(esm[1])
                    all_texts.append(seq[:self.max_length]) # note truncated
            
            # Stack all sequences for this family into single tensors
            esm_input_ids = torch.stack(all_esm_input_ids)
            esm_attention_mask = torch.stack(all_esm_attention_mask)
            progen_input_ids = torch.stack(all_progen_input_ids)
            progen_attention_mask = torch.stack(all_progen_attention_mask)
            
            # Create the entry for this family
            family_entry = {
                'samples' : {
                'esm_input_ids': esm_input_ids,
                'esm_attention_mask': esm_attention_mask,
                'progen_input_ids': progen_input_ids,
                'progen_attention_mask': progen_attention_mask,},
                'pfam': fam,
                'raw_texts': all_texts
            }
            
            # Add to appropriate split
            if fam in test_fams:
                test_tokenized_data.append(family_entry)
            else:
                train_tokenized_data.append(family_entry)
        
        # Save train data to the main file
        torch.save(train_tokenized_data, self.tokenized_data_file)
        logger.info(f"Train tokenized data ({len(train_tokenized_data)} families) saved to {self.tokenized_data_file}")
        
        # Save test data to eval file
        eval_tokenized_data_file = self._get_eval_data_file()
        torch.save(test_tokenized_data, eval_tokenized_data_file)
        logger.info(f"Test tokenized data ({len(test_tokenized_data)} families) saved to {eval_tokenized_data_file}")

    def _get_eval_data_file(self):
        """Get the path to the eval data file based on the train data file."""
        dir_path = os.path.dirname(self.tokenized_data_file)
        base_name = os.path.basename(self.tokenized_data_file)
        return os.path.join(dir_path, 'eval_' + base_name)

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

    def __len__(self):
        # Cap at 2**24 - 1 to avoid torch.multinomial limit in WeightedRandomSampler
        return 1000
    
    def __getitem__(self, idx):

        n = len(self.data)
        source_idx = np.random.randint(0, n)
        target_idx = np.random.randint(0, n)
        item_source = self.data[source_idx]
        item_target = self.data[target_idx]
        
        source_subset_indices = np.random.choice(len(item_source['samples']['esm_input_ids']), size=self.set_size, replace=False)
        target_subset_indices = np.random.choice(len(item_target['samples']['esm_input_ids']), size=self.set_size, replace=False)
        
        # Use torch.tensor() to create new tensors with resizable storage (needed for DataLoader collation)
        # .clone() is insufficient as it may preserve non-resizable storage from torch.load
        esm_input_ids_source = item_source['samples']['esm_input_ids'][source_subset_indices]
        esm_attention_mask_source = item_source['samples']['esm_attention_mask'][source_subset_indices]
        progen_input_ids_source = item_source['samples']['progen_input_ids'][source_subset_indices]
        progen_attention_mask_source = item_source['samples']['progen_attention_mask'][source_subset_indices]

        esm_input_ids_target = item_target['samples']['esm_input_ids'][target_subset_indices]
        esm_attention_mask_target = item_target['samples']['esm_attention_mask'][target_subset_indices]
        progen_input_ids_target = item_target['samples']['progen_input_ids'][target_subset_indices]
        progen_attention_mask_target = item_target['samples']['progen_attention_mask'][target_subset_indices]   


        return { 'source_samples' : {
                'esm_input_ids': esm_input_ids_source,
                'esm_attention_mask': esm_attention_mask_source,
                'progen_input_ids': progen_input_ids_source,
                'progen_attention_mask': progen_attention_mask_source,
                'pfam': item_source['pfam'],
                'raw_texts': [item_source['raw_texts'][i] for i in source_subset_indices],
            },
            'target_samples' : {
                'esm_input_ids': esm_input_ids_target,
                'esm_attention_mask': esm_attention_mask_target,
                'progen_input_ids': progen_input_ids_target,
                'progen_attention_mask': progen_attention_mask_target,
                'pfam': item_target['pfam'],
                'raw_texts': [item_target['raw_texts'][i] for i in target_subset_indices],
            }, 
            'source_idx': source_idx, 
            'target_idx': target_idx, 
            'train_predictor_bool': self.get_train_predictor_bool(source_idx, target_idx)     
        }
    
    def get_train_predictor_bool(self, source_idx, target_idx):
        return False