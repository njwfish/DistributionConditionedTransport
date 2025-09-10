#!/usr/bin/env python3
import argparse
import random
from typing import List, Tuple

import torch
from transformers import EsmTokenizer, AutoTokenizer


def load_dataset(data_path: str):
    """Load the filtered aggregated dataset saved as a list of dicts."""
    data = torch.load(data_path)
    if not isinstance(data, list):
        raise ValueError(f"Expected dataset to be a list, got {type(data)}")
    return data


def get_tokenizers() -> Tuple[EsmTokenizer, AutoTokenizer]:
    """Instantiate tokenizers exactly as used during dataset creation and generators."""
    # ESM tokenizer (used in generator/esm_dfm.py and subsample_data3.py)
    esm_tok = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    # ProGen tokenizer (used in subsample_data3.py and generator/protein_generator.py)
    progen_tok = AutoTokenizer.from_pretrained("hugohrban/progen2-base", trust_remote_code=True)
    # Ensure basic special tokens are set for ProGen
    if progen_tok.pad_token is None:
        progen_tok.pad_token = '<|pad|>'
    if progen_tok.bos_token is None:
        progen_tok.bos_token = '<|bos|>'
    if progen_tok.eos_token is None:
        progen_tok.eos_token = '<|eos|>'
    return esm_tok, progen_tok


def decode_esm(ids_1d: torch.Tensor, esm_tok: EsmTokenizer) -> str:
    """Replicate ESM decode behavior from ESM2_DFM_Generator: skip specials and remove spaces."""
    if ids_1d.dim() != 1:
        raise ValueError(f"decode_esm expects a 1D tensor, got shape {tuple(ids_1d.shape)}")
    text = esm_tok.decode(ids_1d.tolist(), skip_special_tokens=True)
    return text.replace(" ", "")


def decode_progen(ids_1d: torch.Tensor, progen_tok: AutoTokenizer) -> str:
    """Replicate Progen2Generator decode behavior: skip special tokens."""
    if ids_1d.dim() != 1:
        raise ValueError(f"decode_progen expects a 1D tensor, got shape {tuple(ids_1d.shape)}")
    return progen_tok.decode(ids_1d.tolist(), skip_special_tokens=True)


def check_group(entry: dict, esm_tok: EsmTokenizer, progen_tok: AutoTokenizer, num_samples: int, rng: random.Random) -> List[Tuple[int, bool, bool, int]]:
    """
    For a given time-group entry, check a subset of indices and return per-index match results.

    Returns list of tuples: (index, esm_match, progen_match, seq_len)
    """
    samples = entry.get('samples', {})
    raw_texts: List[str] = entry.get('raw_texts', [])

    if not raw_texts:
        return []

    esm_ids = samples.get('esm_input_ids', None)
    progen_ids = samples.get('progen_input_ids', None)

    if esm_ids is None and progen_ids is None:
        return []

    N = len(raw_texts)
    k = min(num_samples, N)
    indices = list(range(N))
    rng.shuffle(indices)
    indices = indices[:k]

    results = []
    for idx in indices:
        raw = raw_texts[idx].strip()
        esm_match = None
        progen_match = None

        if esm_ids is not None:
            decoded_esm = decode_esm(esm_ids[idx].cpu(), esm_tok)
            esm_match = (decoded_esm == raw)

        if progen_ids is not None:
            decoded_progen = decode_progen(progen_ids[idx].cpu(), progen_tok)
            progen_match = (decoded_progen == raw)

        results.append((idx, bool(esm_match) if esm_match is not None else False,
                        bool(progen_match) if progen_match is not None else False,
                        len(raw)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Verify token-to-sequence decoding for ESM and ProGen token IDs")
    parser.add_argument("--data_path", type=str,
                        default="data/spikeprot0430/filtered_aggregated_data_subsampled.pt",
                        help="Path to filtered aggregated dataset (.pt list)")
    parser.add_argument("--timepoints", type=int, default=2, help="Number of time groups to check")
    parser.add_argument("--num_samples", type=int, default=5, help="Samples per time group to verify")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for sampling indices")

    args = parser.parse_args()

    rng = random.Random(args.seed)
    data = load_dataset(args.data_path)
    esm_tok, progen_tok = get_tokenizers()

    total_esm_ok = 0
    total_esm_all = 0
    total_progen_ok = 0
    total_progen_all = 0

    groups_checked = 0
    for entry in data:
        if groups_checked >= args.timepoints:
            break
        time_label = entry.get('time', 'UNKNOWN')
        raw_texts = entry.get('raw_texts', [])
        samples = entry.get('samples', {})

        print(f"\nTime: {time_label} | sequences: {len(raw_texts)} | keys: {sorted(list(samples.keys()))}")
        results = check_group(entry, esm_tok, progen_tok, args.num_samples, rng)

        # Aggregate and print per-index results
        for (idx, esm_match, progen_match, seq_len) in results:
            msg_parts = [f"idx={idx}", f"len={seq_len}"]
            if 'esm_input_ids' in samples:
                msg_parts.append(f"ESM={'OK' if esm_match else 'MISMATCH'}")
                total_esm_all += 1
                if esm_match:
                    total_esm_ok += 1
            if 'progen_input_ids' in samples:
                msg_parts.append(f"ProGen={'OK' if progen_match else 'MISMATCH'}")
                total_progen_all += 1
                if progen_match:
                    total_progen_ok += 1
            print("  " + " | ".join(msg_parts))

        groups_checked += 1

    # Summary
    if total_esm_all > 0:
        print(f"\nESM decode matches: {total_esm_ok}/{total_esm_all} ({100.0*total_esm_ok/total_esm_all:.2f}%)")
    if total_progen_all > 0:
        print(f"ProGen decode matches: {total_progen_ok}/{total_progen_all} ({100.0*total_progen_ok/total_progen_all:.2f}%)")


if __name__ == "__main__":
    main()


