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
                 pll_idx: int = 0,
                 num_workers: int = 1):
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.data_dir = data_dir
        self.set_size = set_size
        self.max_length = max_length
        # TODO: decide whether to remove this or not.
        self.max_sets_per_fam = max_sets_per_fam
        self.include_location = include_location
        self.pll_idx = pll_idx
        self.num_workers = num_workers
        
        
        self.esm_tokenizer = AutoTokenizer.from_pretrained(esm_name, trust_remote_code=True)
        self.progen_tokenizer = AutoTokenizer.from_pretrained(progen_name, trust_remote_code=True)
        
        self.progen_tokenizer.pad_token = '<|pad|>'
        self.progen_tokenizer.bos_token = '<|bos|>'
        self.progen_tokenizer.eos_token = '<|eos|>'

        self.tokenized_data_file = f'{self.data_dir}/tokenized_chunks/virus_tokenized_data_{self.pll_idx}_{self.num_workers}.pt'
        # Ensure data is prepared before building any indices
        if not os.path.exists(self.tokenized_data_file) or tokenize:
            self._tokenize_data() #(lines_to_read=lines_to_read)
        self.data = torch.load(self.tokenized_data_file)
        with open("auxillary_log.log", "a") as f:
            f.write(f"len(self.data): {len(self.data)}\n")
        # Build index pairs after data is loaded
        #self.index_pairs = np.array(
        #    [
        #        (i, j)
        #        for i in range(len(self.data))
        #        for j in range(len(self.data))
        #        if i != j
        #    ]
        #)

    def _tokenize_data(self, lines_to_read=2*10**7):

        fn = self.data_dir+'/spikeprot0430.fasta'
        seqs_by_monthloc = defaultdict(list)
        
        # TODO: should I keep this parameter this way? Might be good to not put too much weight on a single source. But then again, this is way to small if we bunch all locations together at once.
        max_per_monthloc = lines_to_read #self.set_size  # cap per group

        logger.info('building dict')
        
        # Debug log path for progress tracking
        debug_log_path = f"/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/tokenizer_logging/parse_time_loc_debug_{self.pll_idx}_{self.num_workers}.log"

        with open(debug_log_path, "a") as debug_file:
            debug_file.write(f"PARALLEL WORKER {self.pll_idx} from {self.num_workers} STARTING FASTA READING\n")
            debug_file.flush()
        
        # Open file with encoding error handling and pass to SeqIO.parse
        file_handle = open(fn, 'r', encoding='utf-8', errors='replace')
        
        try:
            record_iterator = SeqIO.parse(file_handle, "fasta")
            
            lines_per_worker = lines_to_read // self.num_workers + 1
            
            # Calculate the range of lines this worker should process
            start_line = self.pll_idx * lines_per_worker
            end_line = (self.pll_idx + 1) * lines_per_worker
            
            # Use enumerate to track actual line indices in the FASTA file
            for line_idx, record in enumerate(tqdm(record_iterator, desc=f"Worker {self.pll_idx}")):
                # Only process lines assigned to this worker
                if line_idx < start_line:
                    continue
                if line_idx >= end_line:
                    break
                    
                # Log progress every 10,000 lines
                if line_idx % 100000 == 0:
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"FASTA Reading Progress: {line_idx} lines processed\n")
                        debug_file.flush()
                
                try:
                    fields = record.description.split("|")
                    (gene, isolate, date, iso_id, passage,
                    type_loc, host, o_lab, s_lab,
                    submitter, location) = (fields + ["?"] * 11)[:11]

                    virus_type, state = type_loc.split("^^") if "^^" in type_loc else (type_loc, "?")
                    
                    # TODO: I need to add the filter for the case in which the month in the data is only a single integer.

                    if date[5:7] != '00' and date[-2:] != '00' and date[4] == '-' and date[6].isdigit():
                        if self.include_location:
                            key = date[:7] + '-' + location  # yyyy-mm-location
                        else:
                            key = date[:7]
                        if len(seqs_by_monthloc[key]) < max_per_monthloc:
                            seqs_by_monthloc[key].append(str(record.seq))
                        
                except UnicodeDecodeError as e:
                    # Log detailed Unicode decode error information
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"\n=== UNICODE DECODE ERROR ===\n")
                        debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        debug_file.write(f"FASTA line index: {line_idx}\n")
                        debug_file.write(f"Error type: {type(e).__name__}\n")
                        debug_file.write(f"Error message: {str(e)}\n")
                        debug_file.write(f"Error encoding: {e.encoding}\n")
                        debug_file.write(f"Error object: {e.object}\n")
                        debug_file.write(f"Error start position: {e.start}\n")
                        debug_file.write(f"Error end position: {e.end}\n")
                        debug_file.write(f"Error reason: {e.reason}\n")
                        debug_file.write(f"Problematic bytes around position: {e.object[max(0, e.start-50):e.end+50]}\n")
                        debug_file.write("=== END UNICODE ERROR DEBUG ===\n\n")
                        debug_file.flush()
                    
                    # Also print to console for immediate visibility
                    print(f"Unicode decode error at FASTA line {line_idx}! Check {debug_log_path} for details. Skipping this record.")
                    
                    # Continue to next record instead of crashing
                    continue
                    
                except StopIteration:
                    # End of file reached
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"End of FASTA file reached at line {line_idx}\n")
                        debug_file.flush()
                    #break
                    
                except Exception as e:
                    # Log any other unexpected errors
                    with open(debug_log_path, "a") as debug_file:
                        debug_file.write(f"\n=== UNEXPECTED ERROR ===\n")
                        debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                        debug_file.write(f"FASTA line index: {line_idx}\n")
                        debug_file.write(f"Error type: {type(e).__name__}\n")
                        debug_file.write(f"Error message: {str(e)}\n")
                        debug_file.write("=== END UNEXPECTED ERROR DEBUG ===\n\n")
                        debug_file.flush()
                    
                    print(f"Unexpected error at FASTA line {line_idx}! Check {debug_log_path} for details. Skipping this record.")
                    continue

            # Close file handle immediately after reading sequences to free memory
            file_handle.close()
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"FASTA file reading completed, file handle closed. Starting tokenization.\n")
                debug_file.flush()

            # Initialize tokenized data list - we'll save incrementally to avoid OOM
            tokenized_data = []
            save_every_n_sequences = 10000  # Save intermediate data every N sequences to free memory
            total_sequences_processed = 0
            save_counter = 0
            
            total_timelocs = len(seqs_by_monthloc)
            total_sequences = sum(len(seqs) for seqs in seqs_by_monthloc.values())
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"Starting tokenization of {total_timelocs} time-location groups with {total_sequences} total sequences\n")
                debug_file.write(f"Will save incrementally every {save_every_n_sequences} sequences to manage memory\n")
                debug_file.flush()

            for timeloc_idx, (timeloc, seqs) in enumerate(tqdm(seqs_by_monthloc.items())):
                # Log every iteration of time-location processing
                with open(debug_log_path, "a") as debug_file:
                    debug_file.write(f"Processing time-location {timeloc_idx + 1}/{total_timelocs}: '{timeloc}' with {len(seqs)} sequences\n")
                    debug_file.flush()
                ## TODO: I suppose this is because with max_per_monthloc = self.set_size, if this doesn't match it would be an indicator of a monthloc with too little data.
                ## TODO: if I tokenize every sequence, should I just skip this? After all, it means that we'll also have smaller sets 
                #if len(seqs) < self.set_size:
                #    continue
                #
                ## TODO: why was this set to 1 before? I guess because everything was trimmed to not exceed set size. But should I keep this? 
                ## TODO: And what should I choose for max sets per fam if I even keep it?
                ## TODO: for now I am not adding the +1 because I want to leave out smaller sets.
                #if len(seqs) % self.set_size != 0:
                #    n_sets = len(seqs) // self.set_size #+ 1
                #else:
                #    n_sets = len(seqs) // self.set_size 
                    
                np.random.shuffle(seqs)

                for seq_idx, seq in enumerate(seqs):
                    # Log progress every 100th sequence within each time-location group
                    if total_sequences_processed % 1000 == 0:
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"  Tokenizing sequence {total_sequences_processed + 1}/{total_sequences} (in time-location '{timeloc}')\n")
                            debug_file.flush()

                    # Tokenize individual sequence
                    pg2 = self._tokenize_for_progen(seq)
                    esm = self._tokenize_for_esm(seq)
                    
                    # Create individual entry for this sequence
                    sequence_data = {
                        'samples' : {
                            'esm_input_ids': esm[0].unsqueeze(0),  # Add batch dimension
                            'esm_attention_mask': esm[1].unsqueeze(0),
                            'progen_input_ids': pg2[0].unsqueeze(0),
                            'progen_attention_mask': pg2[1].unsqueeze(0),
                        },
                        'time-loc': timeloc,
                        'raw_texts': [seq[:self.max_length]]  # Single sequence in list
                    }
                    
                    tokenized_data.append(sequence_data)
                    total_sequences_processed += 1
                    
                    # Explicit memory cleanup for individual sequence tensors
                    del pg2, esm, sequence_data
                    
                    # Save incrementally every N sequences to avoid OOM
                    if total_sequences_processed % save_every_n_sequences == 0:
                        # Create intermediate filename
                        intermediate_file = f'{self.tokenized_data_file}.part_{save_counter}'
                        torch.save(tokenized_data, intermediate_file)
                        
                        with open(debug_log_path, "a") as debug_file:
                            debug_file.write(f"Saved intermediate data: {len(tokenized_data)} sequences to {intermediate_file} (total processed: {total_sequences_processed})\n")
                            debug_file.flush()
                        
                        # Clear tokenized_data to free memory
                        del tokenized_data
                        tokenized_data = []
                        save_counter += 1
                
                # Free the raw sequence data for this group
                del seqs
            
            # Save any remaining sequences that haven't been saved yet
            if len(tokenized_data) > 0:
                intermediate_file = f'{self.tokenized_data_file}.part_{save_counter}'
                torch.save(tokenized_data, intermediate_file)
                with open(debug_log_path, "a") as debug_file:
                    debug_file.write(f"Saved final intermediate data: {len(tokenized_data)} sequences to {intermediate_file}\n")
                    debug_file.flush()
                del tokenized_data
                save_counter += 1
            
            # Free the sequence dictionary to save memory
            del seqs_by_monthloc
            
            # TERMINATING HERE to avoid OOM issues during combining step
            # Intermediate files will be combined separately with a high-memory script
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"Tokenization completed successfully! Created {save_counter} intermediate files.\n")
                debug_file.write(f"Intermediate files pattern: {self.tokenized_data_file}.part_*\n")
                debug_file.write(f"Use separate high-memory script to combine these into final output.\n")
                debug_file.flush()
            
            logger.info(f"Tokenization phase complete! Created {save_counter} intermediate files: {self.tokenized_data_file}.part_*")
            logger.info(f"Run separate combining script with high memory to create final {self.tokenized_data_file}")
            
            # NOTE: Final combining step moved to separate script to avoid OOM issues
            sys.exit()
            return  # Exit here without combining
            
        except Exception as e:
            # If any error occurs, make sure file handle is closed
            if 'file_handle' in locals() and not file_handle.closed:
                file_handle.close()
            raise  # Re-raise the exception

    
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
        return len(self.data)
    
    def __getitem__(self, idx):


        item_source = self.data[idx]

        esm_input_ids_source = item_source['samples']['esm_input_ids']
        esm_attention_mask_source = item_source['samples']['esm_attention_mask']
        progen_input_ids_source = item_source['samples']['progen_input_ids']
        progen_attention_mask_source = item_source['samples']['progen_attention_mask']
        
        # Parse time-loc with detailed error handling and debugging
        debug_log_path = f"/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/tokenizer_logging/parse_time_loc_debug_{self.pll_idx}_{self.num_workers}.log"
        
        try:
            # Log successful access for debugging
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"SUCCESS: Dataset index {idx}, time-loc: '{item_source['time-loc']}'\n")
                debug_file.flush()
            
            additional_info = self._parse_time_loc(item_source['time-loc'])
        except Exception as e:
            # Write detailed error information to log file
            with open(debug_log_path, "a") as debug_file:
                debug_file.write(f"\n=== ERROR in _parse_time_loc ===\n")
                debug_file.write(f"Timestamp: {datetime.now().isoformat()}\n")
                debug_file.write(f"Dataset index: {idx}\n")
                debug_file.write(f"Error type: {type(e).__name__}\n")
                debug_file.write(f"Error message: {str(e)}\n")
                debug_file.write(f"item_source['time-loc'] value: '{item_source['time-loc']}'\n")
                debug_file.write(f"item_source['time-loc'] type: {type(item_source['time-loc'])}\n")
                debug_file.write(f"item_source['time-loc'] length: {len(item_source['time-loc']) if hasattr(item_source['time-loc'], '__len__') else 'N/A'}\n")
                debug_file.write(f"item_source['time-loc'] repr: {repr(item_source['time-loc'])}\n")
                debug_file.write(f"Full item_source keys: {list(item_source.keys())}\n")
                debug_file.write(f"Full item_source: {item_source}\n")
                debug_file.write("=== END ERROR DEBUG ===\n\n")
                debug_file.flush()  # Ensure it's written immediately
            
            # Also print to console for immediate visibility
            print(f"ERROR in _parse_time_loc at index {idx}! Check {debug_log_path} for details.")
            
            # Re-raise the exception so the error isn't silently ignored
            additional_info = None
        
        # TODO: in it's current version sometimes the source and target samples seem to have different batch sizes... not sure right now why, just had a weird bug.
        return { 'source_samples' : {
            'esm_input_ids': esm_input_ids_source,
            'esm_attention_mask': esm_attention_mask_source,
            'progen_input_ids': progen_input_ids_source,
            'progen_attention_mask': progen_attention_mask_source,
            },
            'source_time': item_source['time-loc'],
            'additional_info': additional_info
        }