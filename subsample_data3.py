import torch
import random
import numpy as np
import gc
import sys
import argparse
from typing import Dict, List, Optional, Sequence
from transformers import EsmTokenizer, AutoTokenizer, PreTrainedTokenizerBase

def standard_aa_ids(tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
    aa_tokens = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]
    aa_ids = [int(tokenizer.convert_tokens_to_ids(a)) for a in aa_tokens]
    return torch.tensor(sorted(set(aa_ids)), dtype=torch.long)


def x_id(tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    vocab = tokenizer.get_vocab()
    xid = vocab.get("X", None)
    if xid is None:
        val = tokenizer.convert_tokens_to_ids("X")
        xid = int(val) if val is not None else None
    return xid


def tensor_isin(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """Boolean mask of values membership in allowed (torch >=1.10 supports torch.isin)."""
    try:
        return torch.isin(values, allowed)
    except AttributeError:
        if allowed.numel() == 0:
            return torch.zeros_like(values, dtype=torch.bool)
        out = values == allowed[0]
        for k in range(1, allowed.numel()):
            out |= values == allowed[k]
        return out


def progen_bos_token_id(tokenizer: PreTrainedTokenizerBase) -> Optional[int]:
    """Return the id of the literal '<|bos|>' token if present; otherwise fall back to tokenizer.bos_token_id/cls_token_id."""
    try:
        vocab = tokenizer.get_vocab()
        if "<|bos|>" in vocab:
            return int(vocab["<|bos|>"])
    except Exception:
        pass
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        return int(bos_id)
    cls_id = getattr(tokenizer, "cls_token_id", None)
    return int(cls_id) if cls_id is not None else None


def filter_indices_for_esm(
    esm_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [N] where True means keep the sequence.

    ESM exclusion criteria:
      - Must have CLS at index 0 and EOS at index -1
      - All inner tokens (1..L-2) must be one of 20 AAs only (no 'X' allowed)
      - Any sequence with 'X' tokens is excluded (more stringent filtering)
    """
    assert esm_ids.ndim == 2, f"Expected 2D esm_input_ids, got {esm_ids.shape}"
    N, L = esm_ids.shape
    if L < 2:
        # Cannot satisfy BOS/EOS layout; drop all
        return torch.zeros(N, dtype=torch.bool)

    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    if bos_id is None or eos_id is None:
        raise ValueError("ESM tokenizer must provide cls_token_id and eos_token_id")

    # Rule 1: CLS at 0 and EOS at -1
    has_bos = esm_ids[:, 0] == bos_id
    has_eos = esm_ids[:, -1] == eos_id
    ok_bos_eos = has_bos & has_eos

    # Build allowed set for inner positions (20 AA only, no X)
    allowed = standard_aa_ids(tokenizer).tolist()
    xid = x_id(tokenizer)
    # Note: We don't add X to allowed list since we're filtering it out completely
    allowed_ids = torch.tensor(sorted(set(allowed)), dtype=torch.long)

    if L >= 3:
        inner = esm_ids[:, 1:-1]
        inner_is_allowed = tensor_isin(inner, allowed_ids)
        # Rule 2: any disallowed token inside -> drop
        inner_all_allowed = inner_is_allowed.all(dim=1)

        # Rule 3: No X tokens allowed (more stringent than threshold)
        if xid is None:
            x_fraction_ok = torch.ones(N, dtype=torch.bool)
        else:
            x_counts = (inner == int(xid)).sum(dim=1)
            x_fraction_ok = x_counts == 0  # No X tokens allowed at all
    else:
        # No inner positions when L==2
        inner_all_allowed = torch.ones(N, dtype=torch.bool)
        x_fraction_ok = torch.ones(N, dtype=torch.bool)

    keep = ok_bos_eos & inner_all_allowed & x_fraction_ok
    return keep


def filter_indices_for_progen(
    progen_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
) -> torch.Tensor:
    """
    Returns a boolean mask of shape [N] where True means keep the sequence.

    ProGen exclusion criteria:
      - Must have BOS at index 0
      - All tokens from index 1..end must be one of 20 AAs or 'X'
      - In particular, sequences should NOT end with EOS; last token must be AA or 'X'
    """
    assert progen_ids.ndim == 2, f"Expected 2D progen_input_ids, got {progen_ids.shape}"
    N, L = progen_ids.shape
    if L < 1:
        return torch.zeros(N, dtype=torch.bool)

    bos_id = progen_bos_token_id(tokenizer)
    if bos_id is None:
        # Many GPT-like tokenizers expose bos_token_id
        raise ValueError("ProGen tokenizer must provide '<|bos|>' or bos_token_id (or cls_token_id)")

    has_bos = progen_ids[:, 0] == bos_id

    # Allowed tokens for positions 1..end (20 AA + X)
    allowed = standard_aa_ids(tokenizer).tolist()
    xid = x_id(tokenizer)
    if xid is not None:
        allowed.append(int(xid))
    allowed_ids = torch.tensor(sorted(set(allowed)), dtype=torch.long)

    if L >= 2:
        tail = progen_ids[:, 1:]
        tail_is_allowed = tensor_isin(tail, allowed_ids)
        tail_all_allowed = tail_is_allowed.all(dim=1)
    else:
        # Sequence with only BOS -> not acceptable (must have at least one AA)
        tail_all_allowed = torch.zeros(N, dtype=torch.bool)

    keep = has_bos & tail_all_allowed
    return keep


def main():
    parser = argparse.ArgumentParser(description='Process subset of data files for parallel processing')
    parser.add_argument('--job_id', type=int, required=True, help='Job ID (0-based)')
    parser.add_argument('--total_jobs', type=int, required=True, help='Total number of jobs')
    parser.add_argument('--sample_fraction', type=float, default=1.0, help='Fraction of elements to sample from each data object (0 < fraction <= 1)')
    args = parser.parse_args()
    
    job_id = args.job_id
    total_jobs = args.total_jobs
    sample_fraction = args.sample_fraction
    
    # Validate sample_fraction
    if not (0 < sample_fraction <= 1):
        raise ValueError(f"sample_fraction must be between 0 and 1, got {sample_fraction}")
    
    print(f"Starting job {job_id} out of {total_jobs}")
    print(f"Sampling fraction: {sample_fraction}")
    
    # Set random seed for reproducibility
    random.seed(42 + job_id)  # Different seed per job but reproducible
    
    # Generate all (pll_idx, subset_idx) combinations
    all_combinations = []
    for pll_idx in range(35):
        for subset_idx in range(50):
            all_combinations.append((pll_idx, subset_idx))
    
    # Distribute combinations across jobs
    combinations_per_job = len(all_combinations) // total_jobs
    remainder = len(all_combinations) % total_jobs
    
    # Calculate start and end indices for this job
    start_idx = job_id * combinations_per_job + min(job_id, remainder)
    if job_id < remainder:
        end_idx = start_idx + combinations_per_job + 1
    else:
        end_idx = start_idx + combinations_per_job
    
    job_combinations = all_combinations[start_idx:end_idx]
    print(f"Job {job_id} processing combinations: {job_combinations}")
    
    # Load tokenizers using same defaults as filter_unusual_tokens.py
    esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    progen_tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-base", trust_remote_code=True)

    # Dictionary to aggregate filtered data by time-loc
    aggregated_data = {}

    # Process assigned combinations
    for pll_idx, subset_idx in job_combinations:
        print(f"Processing {pll_idx} {subset_idx}")
        with open(f"auxillary_log_job_{job_id}.log", "a") as f:
            f.write(f"Processing {pll_idx} {subset_idx}\n")
        input_file = f"data/spikeprot0430/tokenized_chunks_location_missing/virus_tokenized_data_{pll_idx}_40.pt.part_{subset_idx}"
        try:
            data = torch.load(input_file)
        except: 
            print(f"Could not load {input_file}, skipping...")
            continue

        # Sample a fraction of elements if sample_fraction < 1
        if sample_fraction < 1.0:
            num_elements = len(data)
            num_to_sample = max(1, int(num_elements * sample_fraction))
            sampled_indices = random.sample(range(num_elements), num_to_sample)
            data = [data[i] for i in sampled_indices]
            print(f"  Sampled {num_to_sample}/{num_elements} elements ({sample_fraction:.2%})")

        for eunmeration_idx, el in enumerate(data):

            samples = el["samples"]
            time_loc = el["time-loc"]
            raw_texts = el["raw_texts"]
            
            esm_ids = samples.get("esm_input_ids")
            progen_ids = samples.get("progen_input_ids")
            
            N = esm_ids.shape[0]
            keep_mask = torch.ones(N, dtype=torch.bool)
            
            # Apply ESM filtering
            if esm_ids is not None and esm_tokenizer is not None:
                esm_keep = filter_indices_for_esm(esm_ids, esm_tokenizer)
                keep_mask &= esm_keep
                del esm_keep  # Free memory

            # Apply ProGen filtering
            if progen_ids is not None and progen_tokenizer is not None:
                progen_keep = filter_indices_for_progen(progen_ids, progen_tokenizer)
                keep_mask &= progen_keep
                del progen_keep  # Free memory

            # Apply the filtering mask
            if keep_mask.sum() > 0:  # Only process if there are sequences to keep
                # Filter all tensors in samples
                filtered_samples = {}
                for key, tensor in samples.items():
                    filtered_samples[key] = tensor[keep_mask]
                
                # Filter raw_texts list using the mask
                filtered_raw_texts = [raw_texts[i] for i in range(len(raw_texts)) if keep_mask[i]]
                
                # Create filtered element
                filtered_element = {
                    'samples': filtered_samples,
                    'time-loc': time_loc,
                    'raw_texts': filtered_raw_texts
                }
                
                # Aggregate by time-loc
                if time_loc not in aggregated_data:
                    aggregated_data[time_loc] = []
                aggregated_data[time_loc].append(filtered_element)
                
                # Clean up intermediate variables
                del filtered_samples, filtered_raw_texts, filtered_element
            
            # Clean up variables for this element
            del samples, esm_ids, progen_ids, keep_mask, time_loc, raw_texts
        
        # Clean up the loaded data and force garbage collection
        del data
        gc.collect()
        
        # Optional: Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the partial aggregated data for this job
    output_file = f"data/spikeprot0430/partial_aggregated_data_job_{job_id}.pt"
    torch.save(aggregated_data, output_file)
    print(f"Job {job_id} completed. Saved partial aggregated data to {output_file}")
    print(f"Found {len(aggregated_data)} unique time-loc entries in this job")
    
    # Print summary statistics
    total_elements = sum(len(elements) for elements in aggregated_data.values())
    total_sequences = sum(sum(len(el['raw_texts']) for el in elements) for elements in aggregated_data.values())
    print(f"Job {job_id} summary: {total_elements} elements, {total_sequences} sequences")

if __name__ == "__main__":
    main()


