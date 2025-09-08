import torch
import random
import numpy as np
import gc
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


# Load tokenizers using same defaults as filter_unusual_tokens.py
esm_tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
progen_tokenizer = AutoTokenizer.from_pretrained("hugohrban/progen2-base", trust_remote_code=True)

# Dictionary to aggregate filtered data by time-loc
aggregated_data = {}

# Define file paths
for pll_idx in range(5):
    for subset_idx in range(5):
        print(f"Processing {pll_idx} {subset_idx}")
        input_file = f"data/spikeprot0430/tokenized_chunks/virus_tokenized_data_{pll_idx}_40.pt.part_{subset_idx}"
        try:
            data = torch.load(input_file)
        except: 
            break

        for el in data:
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

# Further aggregate data: concatenate raw_texts and stack tensors for each time-loc
print(f"Found {len(aggregated_data)} unique time-loc entries")
final_aggregated_data = {}

for time_loc, elements in aggregated_data.items():
    total_sequences = sum(len(el['raw_texts']) for el in elements)
    print(f"Time-loc '{time_loc}': {len(elements)} chunks, {total_sequences} sequences")
    
    if len(elements) == 0:
        continue
        
    # Concatenate all raw_texts lists
    all_raw_texts = []
    for el in elements:
        all_raw_texts.extend(el['raw_texts'])
    
    # Stack all tensors for each key in samples
    stacked_samples = {}
    sample_keys = elements[0]['samples'].keys()  # Get keys from first element
    
    for key in sample_keys:
        # Collect all tensors for this key across all elements
        tensors_to_stack = []
        for el in elements:
            tensors_to_stack.append(el['samples'][key])
        
        # Stack them along dimension 0 (batch dimension)
        stacked_samples[key] = torch.cat(tensors_to_stack, dim=0)
    
    # Create the final aggregated entry for this time-loc
    final_aggregated_data[time_loc] = {
        'samples': stacked_samples,
        'time-loc': time_loc,
        'raw_texts': all_raw_texts
    }
    
    print(f"  -> Final aggregated: {len(all_raw_texts)} sequences, tensor shapes: {stacked_samples['esm_input_ids'].shape}")

# Clean up intermediate aggregation data to free memory
del aggregated_data
gc.collect()

# Optional: Clear GPU cache if using CUDA
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Save the final aggregated data
output_file = "data/spikeprot0430/filtered_aggregated_data.pt"
torch.save(final_aggregated_data, output_file)
print(f"Saved final aggregated data to {output_file}")
print(f"Final structure: {len(final_aggregated_data)} time-loc entries")


