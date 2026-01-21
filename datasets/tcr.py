import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import os
import logging
import hydra
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# The 20 standard amino acids
STANDARD_AMINO_ACIDS = set('ACDEFGHIKLMNPQRSTVWY')


def is_valid_sequence(seq: str) -> bool:
    """Check if sequence contains only standard amino acids."""
    return all(aa in STANDARD_AMINO_ACIDS for aa in seq.upper())


class TCRDataset(Dataset):
    def __init__(
            self,
            seed: Optional[int] = None,
            set_size: int = 32,
            data_dir: str = 'data/tcr',
            data_file: str = 'tcr_tokenized_data.pt',
            esm_name: str = 'facebook/esm2_t6_8M_UR50D',
            progen_name: str = 'hugohrban/progen2-small',
            max_length: int = 64,  # TCR junction sequences are typically short (~10-20 aa)
            tokenize: bool = False,
            within_patient: bool = True,  # If True, only learn within-patient transitions; else any-to-any
            split: str = 'train',  # 'train' or 'test'
            test_fraction: float = 0.2,  # Fraction of patients to hold out for testing
            **kwargs,  # absorb any extra keyword args without failing
            ):

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.set_size = set_size
        self.max_length = max_length
        self.data_dir = data_dir
        self.data_file = data_file
        self.within_patient = within_patient
        self.split = split
        self.test_fraction = test_fraction
        
        # Initialize tokenizers
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
        
        self.tokenized_data_file = os.path.join(self.base_dir, self.data_dir, self.data_file)
        
        # Load or create tokenized data
        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data()
        
        loaded = torch.load(self.tokenized_data_file)
        all_data = loaded['data']
        all_metadata = loaded['metadata']

        # Split patients into train and test sets
        self._split_data(all_data, all_metadata)

        # Build next timepoint map for train_predictor_bool
        self._build_next_timepoint_map()

        # Build valid pair indices based on within_patient setting
        self._build_pair_indices()

    def _split_data(self, all_data, all_metadata):
        """Split data by patient into train/test sets.

        Special handling:
        - When within_patient=False and split='train': Include first timepoint from ALL patients
          (including test patients) to ensure train set has anchor points from all patients.
        - When within_patient=True: First timepoint cannot be used as source (handled in _build_pair_indices).
        """
        # Get unique patients
        unique_patients = list(set(subject_id for subject_id, _ in all_metadata))
        unique_patients.sort()  # Sort for reproducibility

        # Determine train/test patients
        n_test = max(1, int(len(unique_patients) * self.test_fraction))
        test_patients = set(unique_patients[:n_test])
        train_patients = set(unique_patients[n_test:])

        logger.info(f'Split {len(unique_patients)} patients: {len(train_patients)} train, {len(test_patients)} test')
        logger.info(f'Test patients: {sorted(test_patients)}')

        # Find first timepoint for each patient
        first_timepoints = {}
        for subject_id, timepoint in all_metadata:
            if subject_id not in first_timepoints or timepoint < first_timepoints[subject_id]:
                first_timepoints[subject_id] = timepoint

        # Filter data based on split
        if self.split == 'train':
            target_patients = train_patients
        elif self.split == 'test':
            target_patients = test_patients
        else:
            raise ValueError(f"Unknown split: {self.split}. Must be 'train' or 'test'.")

        self.data = []
        self.metadata = []
        for data_item, (subject_id, timepoint) in zip(all_data, all_metadata):
            include = False

            if subject_id in target_patients:
                include = True
            # For train split with any-to-any, also include first timepoint from test patients
            elif (self.split == 'train' and not self.within_patient and
                  subject_id in test_patients and timepoint == first_timepoints[subject_id]):
                include = True
                logger.info(f'Including first timepoint from test patient {subject_id}/time={timepoint} in train')

            if include:
                self.data.append(data_item)
                self.metadata.append((subject_id, timepoint))

        logger.info(f'{self.split.capitalize()} split: {len(self.data)} repertoires')

    def _build_next_timepoint_map(self):
        """Build mapping from current timepoint to next timepoint for each patient."""
        self.next_timepoint_map = {}

        # Group timepoints by patient
        self.patient_timepoints = {}
        self.patient_indices = {}  # Map (subject_id) -> list of indices in metadata
        for idx, (subject_id, timepoint) in enumerate(self.metadata):
            if subject_id not in self.patient_timepoints:
                self.patient_timepoints[subject_id] = []
                self.patient_indices[subject_id] = []
            self.patient_timepoints[subject_id].append(timepoint)
            self.patient_indices[subject_id].append(idx)

        # For each patient, sort timepoints and create mapping
        for subject_id, timepoints in self.patient_timepoints.items():
            sorted_timepoints = sorted(timepoints)
            self.next_timepoint_map[subject_id] = {}
            for i in range(len(sorted_timepoints) - 1):
                current_time = sorted_timepoints[i]
                next_time = sorted_timepoints[i + 1]
                self.next_timepoint_map[subject_id][current_time] = next_time

    def _build_pair_indices(self):
        """Build list of valid (source_idx, target_idx) pairs based on within_patient setting."""
        self.pair_indices = []
        n = len(self.metadata)

        if self.within_patient:
            # Only pairs within the same patient (consecutive timepoints)
            for subject_id, indices in self.patient_indices.items():
                # Sort indices by timepoint
                idx_time_pairs = [(idx, self.metadata[idx][1]) for idx in indices]
                idx_time_pairs.sort(key=lambda x: x[1])  # Sort by timepoint

                # Create pairs of consecutive timepoints
                for i in range(len(idx_time_pairs) - 1):
                    source_idx = idx_time_pairs[i][0]
                    target_idx = idx_time_pairs[i + 1][0]
                    self.pair_indices.append((source_idx, target_idx))
        else:
            # Any-to-any: all pairs (i, j) where i != j
            for i in range(n):
                for j in range(n):
                    if i != j:
                        self.pair_indices.append((i, j))
    
    def _tokenize_data(self):
        """Load raw data, filter sequences, tokenize, and save."""
        logger.info('Loading and tokenizing TCR data...')
        
        # Load repertoire index
        repertoire_index_path = os.path.join(self.base_dir, 'tcr_dataset', 'repertoire_index.tsv')
        repertoire_index = pd.read_csv(repertoire_index_path, sep='\t')
        
        tokenized_data = []
        metadata = []
        
        # Load data for each repertoire
        for _, row in repertoire_index.iterrows():
            subject_id = row['subject_id']
            timepoint = row['timepoint']
            
            # Construct path to full_data_unit.tsv
            data_unit_path = os.path.join(
                self.base_dir, 
                'tcr_dataset', 
                'tcr_data', 
                f'subject={subject_id}', 
                f'time={timepoint}', 
                'full_data_unit.tsv'
            )
            
            # Load the data unit and extract sequences
            data_unit = pd.read_csv(data_unit_path, sep='\t')
            raw_sequences = data_unit['junction_aa'].tolist()
            
            # Filter sequences: keep only those with standard amino acids
            sequences = [seq for seq in raw_sequences if isinstance(seq, str) and is_valid_sequence(seq)]
            
            n_filtered = len(raw_sequences) - len(sequences)
            if n_filtered > 0:
                logger.info(f'Repertoire {subject_id}/time={timepoint}: filtered {n_filtered} sequences with non-standard amino acids')
            
            # Skip repertoires that don't have enough sequences
            if len(sequences) < self.set_size:
                logger.warning(f'Repertoire {subject_id}/time={timepoint}: only {len(sequences)} valid sequences (< set_size={self.set_size}), skipping')
                continue
            
            # Tokenize all sequences for this repertoire
            all_esm_input_ids = []
            all_esm_attention_mask = []
            all_progen_input_ids = []
            all_progen_attention_mask = []
            all_texts = []
            
            for seq in sequences:
                esm = self._tokenize_for_esm(seq)
                all_esm_input_ids.append(esm[0])
                all_esm_attention_mask.append(esm[1])
                
                pg = self._tokenize_for_progen(seq)
                all_progen_input_ids.append(pg[0])
                all_progen_attention_mask.append(pg[1])
                
                all_texts.append(seq[:self.max_length])
            
            # Stack into tensors
            tokenized_data.append({
                'samples': {
                    'esm_input_ids': torch.stack(all_esm_input_ids),
                    'esm_attention_mask': torch.stack(all_esm_attention_mask),
                    'progen_input_ids': torch.stack(all_progen_input_ids),
                    'progen_attention_mask': torch.stack(all_progen_attention_mask),
                },
                'raw_texts': all_texts,
            })
            metadata.append((subject_id, timepoint))
            
            logger.info(f'Tokenized repertoire {subject_id}/time={timepoint}: {len(sequences)} sequences')
        
        # Save tokenized data
        torch.save({'data': tokenized_data, 'metadata': metadata}, self.tokenized_data_file)
        logger.info(f'Tokenized data saved to {self.tokenized_data_file}')
    
    def _tokenize_for_esm(self, sequence: str):
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
    
    def _tokenize_for_progen(self, sequence: str):
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
    
    def get_train_predictor_bool(self, source_metadata, target_metadata):
        """Check if target is the next consecutive timepoint for source."""
        source_subject, source_time = source_metadata
        target_subject, target_time = target_metadata
        
        # Must be from the same patient
        if source_subject != target_subject:
            return False
        
        # Check if target is the next consecutive timepoint for this patient
        if source_subject in self.next_timepoint_map:
            expected_next_time = self.next_timepoint_map[source_subject].get(source_time)
            return expected_next_time == target_time
        
        return False
          
    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        source_idx, target_idx = self.pair_indices[idx]
        
        item_source = self.data[source_idx]
        item_target = self.data[target_idx]
        
        source_metadata = self.metadata[source_idx]
        target_metadata = self.metadata[target_idx]
        
        # Sample subset indices
        source_subset_indices = np.random.choice(
            len(item_source['samples']['esm_input_ids']), 
            size=self.set_size, 
            replace=False
        )
        target_subset_indices = np.random.choice(
            len(item_target['samples']['esm_input_ids']), 
            size=self.set_size, 
            replace=False
        )
        
        # Extract source samples
        esm_input_ids_source = item_source['samples']['esm_input_ids'][source_subset_indices]
        esm_attention_mask_source = item_source['samples']['esm_attention_mask'][source_subset_indices]
        progen_input_ids_source = item_source['samples']['progen_input_ids'][source_subset_indices]
        progen_attention_mask_source = item_source['samples']['progen_attention_mask'][source_subset_indices]
        
        # Extract target samples
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
                'subject_id': source_metadata[0],
                'timepoint': source_metadata[1],
                'raw_texts': [item_source['raw_texts'][i] for i in source_subset_indices],
            },
            'target_samples': {
                'esm_input_ids': esm_input_ids_target,
                'esm_attention_mask': esm_attention_mask_target,
                'progen_input_ids': progen_input_ids_target,
                'progen_attention_mask': progen_attention_mask_target,
                'subject_id': target_metadata[0],
                'timepoint': target_metadata[1],
                'raw_texts': [item_target['raw_texts'][i] for i in target_subset_indices],
            },
            'source_idx': source_idx,
            'target_idx': target_idx,
            'train_predictor_bool': self.get_train_predictor_bool(source_metadata, target_metadata),
        }