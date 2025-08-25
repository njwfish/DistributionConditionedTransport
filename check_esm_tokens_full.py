#!/usr/bin/env python3
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import torch
from transformers import EsmTokenizer

from utils.hf_local import resolve_local_or_repo


def load_dataset(path: str) -> List[dict]:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected list at root of {path}, got {type(data)}")
    for i, el in enumerate(data):
        if not isinstance(el, dict) or "samples" not in el:
            raise ValueError(f"Element {i} missing 'samples'")
        samples = el["samples"]
        if "esm_input_ids" not in samples:
            raise ValueError(f"Element {i} samples missing 'esm_input_ids'")
    return data


def standard_aa_ids(tokenizer: EsmTokenizer) -> set:
    aa_tokens = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]
    return {tokenizer.convert_tokens_to_ids(a) for a in aa_tokens}


def analyze_dataset(data: List[dict], tokenizer: EsmTokenizer) -> Tuple[Counter, Dict[str, int], List[Tuple[int, int, List[int]]]]:
    aa_ids = standard_aa_ids(tokenizer)
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id

    counts_all: Counter = Counter()

    num_sequences_total = 0
    num_sequences_with_anomaly = 0
    anomalies: List[Tuple[int, int, List[int]]] = []

    for month_idx, el in enumerate(data):
        ids: torch.Tensor = el["samples"]["esm_input_ids"]  # [N, L]
        if ids.ndim != 2:
            raise ValueError(f"Expected 2D ids, got {ids.shape}")
        N, L = ids.shape
        num_sequences_total += N

        # Count all tokens regardless of attention mask
        counts_all.update(ids.view(-1).tolist())

        # Detect anomalies across all positions, only exempt BOS at index 0 and EOS at index L-1
        for i in range(N):
            row_ids = ids[i]
            offending: List[int] = []
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
    return counts_all, summary, anomalies


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check ESM tokens across dataset and detect non-standard AA tokens (ignoring attention mask)")
    parser.add_argument("--data", default="data/spikeprot0430/virus_tokenized_data_for_tde.pt", help="Path to aggregated dataset .pt")
    parser.add_argument("--esm_name", default="facebook/esm2_t6_8M_UR50D", help="ESM model id or local path")
    parser.add_argument("--sort", choices=["id", "count"], default="id", help="Sort token table by id or by count")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {data_path}")
    data = load_dataset(data_path)

    model_id = resolve_local_or_repo(args.esm_name)
    tokenizer = EsmTokenizer.from_pretrained(model_id)

    # Print token inventory
    print("\nTokenizer vocabulary (token -> id):")
    vocab = tokenizer.get_vocab()  # token -> id
    for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
        print(f"  {tok:<20} -> {tid}")

    print("\nCounting tokens and checking anomalies (ignoring attention mask) ...")
    counts_all, summary, anomalies = analyze_dataset(data, tokenizer)

    print("\nSummary:")
    print(f"  total sequences: {summary['num_sequences_total']}")
    print(f"  sequences with non-standard inner tokens: {summary['num_sequences_with_anomaly']}")

    # Build id -> token map for stable printing
    id_to_token = {tid: tok for tok, tid in vocab.items()}

    # Determine ids to print
    ids_with_counts = set(counts_all.keys())
    if args.sort == "id":
        ordered_ids = sorted(ids_with_counts)
    else:
        ordered_ids = [tid for tid, _ in counts_all.most_common()]
        for tid in sorted(ids_with_counts):
            if tid not in ordered_ids:
                ordered_ids.append(tid)

    print("\nToken frequencies (id -> token -> count):")
    for tid in ordered_ids:
        tok = id_to_token.get(tid, tokenizer.convert_ids_to_tokens(int(tid)))
        c = counts_all.get(tid, 0)
        print(f"  {tid:>5} -> {tok:<20} -> {c}")

    if anomalies:
        print("\nExamples of sequences with non-standard inner tokens (showing up to 10):")
        for month_idx, seq_idx, offending in anomalies[:10]:
            ids = data[month_idx]["samples"]["esm_input_ids"][seq_idx]
            text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
            print(f"  month_idx={month_idx} seq_idx={seq_idx} offending_ids={offending}")
            print(f"    decoded: {text}")
    else:
        print("\nNo non-standard tokens found (excluding BOS at start and EOS at end).")


if __name__ == "__main__":
    main()


