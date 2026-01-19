import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
import os
import hydra
from hydra.core.global_hydra import GlobalHydra
import pandas as pd
from Bio import pairwise2

class TCRDataset(Dataset):
    def __init__(
            self,
            seed: Optional[int] = None,
            set_size: int = 32,
            data_dir: str = 'tcr_dataset',
            max_seq_length: int = 301,  # Maximum sequence length for encoder and HyenaDNA
            **kwargs,  # absorb any extra keyword args without failing
            ):

        if seed is not None:
            np.random.seed(seed)
        
        self.set_size = set_size
        self.max_seq_length = max_seq_length
        
        # Create the DNA vocabulary for one-hot encoding
        self.dna_vocab = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
        self.vocab_size = len(self.dna_vocab)
        
        # Initialize HyenaDNA tokenizer
        self._init_hyena_tokenizer()
        
        # Resolve base directory robustly with or without Hydra
        if GlobalHydra.instance().is_initialized():
            base_dir = hydra.utils.get_original_cwd()
        else:
            base_dir = os.getcwd()
        
        # Load repertoire index
        repertoire_index_path = os.path.join(base_dir, data_dir, 'repertoire_index.tsv')
        repertoire_index = pd.read_csv(repertoire_index_path, sep='\t')
        
        # Initialize data and metadata lists
        self.data = []
        self.metadata = []
        
        # Load data for each repertoire
        for _, row in repertoire_index.iterrows():
            subject_id = row['subject_id']
            timepoint = row['timepoint']
            
            # Construct path to full_data_unit.tsv
            data_unit_path = os.path.join(
                base_dir, 
                data_dir, 
                'tcr_data', 
                f'subject={subject_id}', 
                f'time={timepoint}', 
                'full_data_unit.tsv'
            )
            
            # Load the data unit and extract sequences
            data_unit = pd.read_csv(data_unit_path, sep='\t')
            sequences = data_unit['sequence'].tolist()
            
            # Add to data and metadata
            self.data.append(sequences)
            self.metadata.append((subject_id, timepoint))
        
        # Build next timepoint map for train_predictor_bool
        self.next_timepoint_map = {}
        
        # Group timepoints by patient
        patient_timepoints = {}
        for subject_id, timepoint in self.metadata:
            if subject_id not in patient_timepoints:
                patient_timepoints[subject_id] = []
            patient_timepoints[subject_id].append(timepoint)
        
        # For each patient, sort timepoints and create mapping
        for subject_id, timepoints in patient_timepoints.items():
            sorted_timepoints = sorted(timepoints)
            self.next_timepoint_map[subject_id] = {}
            for i in range(len(sorted_timepoints) - 1):
                current_time = sorted_timepoints[i]
                next_time = sorted_timepoints[i + 1]
                self.next_timepoint_map[subject_id][current_time] = next_time
    
    def _init_hyena_tokenizer(self):
        """Initialize the HyenaDNA tokenizer."""
        from datasets.hyena_tokenizer import CharacterTokenizer
        
        # HyenaDNA uses a character-level tokenizer
        vocab = ["A", "C", "G", "T", "N"]
        self.hyena_tokenizer = CharacterTokenizer(characters=vocab, model_max_length=self.max_seq_length)
    
    def _encode_dna_sequence(self, sequence):
        """
        One-hot encode a DNA sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            One-hot encoded tensor
        """
        # Truncate if necessary (shouldn't be the case here because longest sequence is 301 bp)
        sequence = sequence[:self.max_seq_length].upper()
        
        # Convert to indices
        indices = [self.dna_vocab.get(base, self.dna_vocab["N"]) for base in sequence]
        
        # Pad to max_seq_length
        padded_indices = indices + [self.dna_vocab["N"]] * (self.max_seq_length - len(indices))
        
        # One-hot encode
        one_hot = torch.zeros(self.max_seq_length, self.vocab_size)
        for i, idx in enumerate(padded_indices):
            one_hot[i, idx] = 1.0
            
        return one_hot
    
    def _tokenize_for_hyena(self, sequence):
        """
        Tokenize a DNA sequence for HyenaDNA.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Tokenized tensor and attention mask
        """
        # Truncate if necessary, but ensure we keep enough space for special tokens
        effective_max_length = self.max_seq_length - 2  # -2 for [CLS] and [SEP]
        sequence = sequence[:effective_max_length].upper()
        
        # Tokenize with special tokens
        tokens = self.hyena_tokenizer(
            sequence, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_seq_length,
            add_special_tokens=True,
            return_tensors='pt'
        )
        
        return tokens.input_ids[0], tokens.attention_mask[0]
    
    def _process_sequences(self, sequences):
        """
        Process a list of sequences into encoder inputs and HyenaDNA tokens.
        
        Args:
            sequences: List of DNA sequence strings
            
        Returns:
            Dict with encoder_inputs, hyena_input_ids, and hyena_attention_mask
        """
        # One-hot encode sequences for the encoder
        encoder_inputs = torch.stack([self._encode_dna_sequence(seq) for seq in sequences])
        
        # Tokenize sequences for HyenaDNA
        hyena_input_ids = []
        hyena_attention_masks = []
        
        for seq in sequences:
            input_ids, attention_mask = self._tokenize_for_hyena(seq)
            hyena_input_ids.append(input_ids)
            hyena_attention_masks.append(attention_mask)
        
        hyena_input_ids = torch.stack(hyena_input_ids)
        hyena_attention_masks = torch.stack(hyena_attention_masks)
        
        return {
            'encoder_inputs': encoder_inputs,
            'hyena_input_ids': hyena_input_ids,
            'hyena_attention_mask': hyena_attention_masks
        }
    
    # NOTE: relevant for co-training updated that I (Paolo) need to push, ignore for now.
    def get_train_predictor_bool(self, source_metadata, target_metadata):
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
    
    # NOTE: current getitem method returns dictionary, not tensor, so this is (if I'm not mistaken) incompatible with the OTCollate class.
    def pairwise_distance(self, seq1, seq2):
        """
        Compute distance between two DNA sequences based on global alignment.
        Uses Bio.pairwise2 for global alignment and returns 1 - identity.
        
        Args:
            seq1: First DNA sequence (string of ACTG letters)
            seq2: Second DNA sequence (string of ACTG letters)
            
        Returns:
            Distance value between 0 and 1 (0 = identical, 1 = completely different)
        """
        score = pairwise2.align.globalxx(seq1, seq2, score_only=True)
        length = max(len(seq1), len(seq2))
        identity = score / length
        return 1 - identity
          
    def __len__(self):
        # NOTE: in practice this should be n = 69.
        n = len(self.metadata)
        return n * (n - 1)
    
    def __getitem__(self, idx):
        n = len(self.metadata)
        i = idx // (n - 1)
        j = idx % (n - 1)
        if j >= i:
            j += 1
        source_idx, target_idx = i, j
        
        source_samples = self.data[source_idx]
        target_samples = self.data[target_idx]
        
        source_metadata = self.metadata[source_idx]
        target_metadata = self.metadata[target_idx]
        
        source_subset_indices = np.random.choice(len(source_samples), size=self.set_size, replace=False)
        target_subset_indices = np.random.choice(len(target_samples), size=self.set_size, replace=False)
        
        source_sequences = [source_samples[subset_idx] for subset_idx in source_subset_indices]
        target_sequences = [target_samples[subset_idx] for subset_idx in target_subset_indices]
        
        # Process sequences into encoder inputs and HyenaDNA tokens
        source_processed = self._process_sequences(source_sequences)
        target_processed = self._process_sequences(target_sequences)
        
        return {
            'source_samples': {
                'encoder_inputs': source_processed['encoder_inputs'],
                'hyena_input_ids': source_processed['hyena_input_ids'],
                'hyena_attention_mask': source_processed['hyena_attention_mask'],
                'raw_texts': source_sequences,
                'source_metadata': source_metadata
            },
            'target_samples': {
                'encoder_inputs': target_processed['encoder_inputs'],
                'hyena_input_ids': target_processed['hyena_input_ids'],
                'hyena_attention_mask': target_processed['hyena_attention_mask'],
                'raw_texts': target_sequences,
                'target_metadata': target_metadata
            },              
            'train_predictor_bool': self.get_train_predictor_bool(source_metadata, target_metadata),
        }