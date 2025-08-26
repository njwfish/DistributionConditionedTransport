#!/usr/bin/env python3
import os
import sys
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import torch
from transformers import EsmTokenizer

from utils.hf_local import resolve_local_or_repo


def load_dataset(path: str) -> List[dict]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected a list at root of {path}, got {type(data)}")
    # dataset produced by aggregate_by_month.py is a list of dicts with keys: 'samples', 'time', 'raw_texts'
    for i, el in enumerate(data):
        if not isinstance(el, dict) or "samples" not in el or "esm_input_ids" not in el["samples"]:
            raise ValueError(f"Element {i} missing expected structure: {el.keys() if isinstance(el, dict) else type(el)}")
    return data


def build_vocab_sets(tokenizer: EsmTokenizer) -> Dict[str, set]:
    # Standard amino acids per generator/esm2_dfm_uniform_seq2seq.py
    aa_tokens = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]
    aa_ids = {tokenizer.convert_tokens_to_ids(a) for a in aa_tokens}

    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    mask_id = tokenizer.mask_token_id

    return {
        "aa_ids": aa_ids,
        "bos_id": {bos_id} if bos_id is not None else set(),
        "eos_id": {eos_id} if eos_id is not None else set(),
        "pad_id": {pad_id} if pad_id is not None else set(),
        "mask_id": {mask_id} if mask_id is not None else set(),
    }


def count_tokens_and_find_anomalies(
    data: List[dict], tokenizer: EsmTokenizer
) -> Tuple[Counter, List[Tuple[int, int, List[int]]], Dict[str, int]]:
    """
    Returns:
      - token_counts: counts across all positions in esm_input_ids (ignoring attention mask)
      - anomalies: list of (month_index, seq_index, offending_token_ids)
      - summary: dict with counts of sequences with anomalies and totals
    """
    vocab_sets = build_vocab_sets(tokenizer)
    aa_ids = vocab_sets["aa_ids"]
    bos_id = next(iter(vocab_sets["bos_id"])) if vocab_sets["bos_id"] else None
    eos_id = next(iter(vocab_sets["eos_id"])) if vocab_sets["eos_id"] else None

    token_counts: Counter = Counter()
    anomalies: List[Tuple[int, int, List[int]]] = []
    num_sequences_total = 0
    num_sequences_with_anomaly = 0

    for month_idx, el in enumerate(data):
        samples = el["samples"]
        ids: torch.Tensor = samples["esm_input_ids"]  # [N, L]

        if ids.ndim != 2:
            raise ValueError(f"Expected 2D tensor for ids, got {ids.shape}")

        N, L = ids.shape
        num_sequences_total += N

        # Count all tokens regardless of attention mask
        token_counts.update(ids.view(-1).tolist())

        # Detect anomalies across all positions; only exempt BOS at idx 0 and EOS at last idx
        for i in range(N):
            row_ids = ids[i]
            offending = []
            for pos in range(L):
                tid = int(row_ids[pos].item())
                if pos == 0 and bos_id is not None and tid == bos_id:
                    continue
                if pos == L - 1 and eos_id is not None and tid == eos_id:
                    continue
                if tid not in aa_ids:
                    offending.append(tid)
            if offending:
                num_sequences_with_anomaly += 1
                anomalies.append((month_idx, i, offending))

    summary = {
        "num_sequences_total": num_sequences_total,
        "num_sequences_with_anomaly": num_sequences_with_anomaly,
    }
    return token_counts, anomalies, summary


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check ESM tokens and find non-standard amino acids in dataset")
    parser.add_argument("--data", default="data/spikeprot0430/virus_tokenized_data_for_tde.pt", help="Path to aggregated dataset .pt")
    parser.add_argument("--esm_name", default="facebook/esm2_t6_8M_UR50D", help="ESM model name or local path")
    parser.add_argument("--top_k", type=int, default=100, help="Print top-k most frequent tokens")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {data_path}")
    data = load_dataset(data_path)

    resolved = resolve_local_or_repo(args.esm_name)
    tokenizer = EsmTokenizer.from_pretrained(resolved)

    print("Building counts and checking for anomalies ...")
    token_counts, anomalies, summary = count_tokens_and_find_anomalies(data, tokenizer)

    # Print summary
    print("\nSummary:")
    print(f"  total sequences: {summary['num_sequences_total']}")
    print(f"  sequences with non-standard inner tokens: {summary['num_sequences_with_anomaly']}")

    # Report token counts with token strings
    print("\nTop token frequencies (by id -> token -> count):")
    most_common = token_counts.most_common(args.top_k)
    for tid, cnt in most_common:
        tok = tokenizer.convert_ids_to_tokens(int(tid))
        print(f"  {tid:>5} -> {tok:<10} -> {cnt}")

    # If anomalies present, print a small sample
    if anomalies:
        print("\nExamples of sequences with non-standard inner tokens (showing up to 10):")
        for month_idx, seq_idx, offending in anomalies[:10]:
            # Decode the sequence for quick inspection (skip_special_tokens False to see BOS/EOS)
            ids = data[month_idx]["samples"]["esm_input_ids"][seq_idx]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            print(f"  month_idx={month_idx} seq_idx={seq_idx} offending_ids={offending}")
            print(f"    decoded: {text}")
    else:
        print("\nNo non-standard inner tokens found.")


if __name__ == "__main__":
    main()


