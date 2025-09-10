#!/usr/bin/env python3
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple, Optional

import torch
from transformers import EsmTokenizer, AutoTokenizer, PreTrainedTokenizerBase
import statistics

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


def standard_aa_ids(tokenizer: PreTrainedTokenizerBase) -> set:
    aa_tokens = [
        "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
    ]
    return {tokenizer.convert_tokens_to_ids(a) for a in aa_tokens}


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


def get_bos_eos_ids(tokenizer: PreTrainedTokenizerBase, ids_key: str) -> Tuple[Optional[int], Optional[int]]:
    """Get BOS and EOS token IDs appropriate for the tokenizer type."""
    if "progen" in ids_key.lower():
        # ProGen2 uses <|bos|> and typically doesn't have EOS at the end
        bos_id = progen_bos_token_id(tokenizer)
        eos_id = None  # ProGen2 sequences don't end with EOS
    else:
        # ESM2 uses cls_token_id as BOS and eos_token_id as EOS
        bos_id = tokenizer.cls_token_id
        eos_id = tokenizer.eos_token_id
    return bos_id, eos_id


def analyze_dataset(data: List[dict], tokenizer: PreTrainedTokenizerBase, ids_key: str) -> Tuple[Counter, Dict[str, int], List[Tuple[int, int, List[int]]]]:
    aa_ids = standard_aa_ids(tokenizer)
    bos_id, eos_id = get_bos_eos_ids(tokenizer, ids_key)

    counts_all: Counter = Counter()

    num_sequences_total = 0
    num_sequences_with_anomaly = 0
    anomalies: List[Tuple[int, int, List[int]]] = []

    for month_idx, el in enumerate(data):
        ids: torch.Tensor = el["samples"][ids_key]  # [N, L]
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


def get_token_ids(tokenizer: PreTrainedTokenizerBase) -> Dict[str, Optional[int]]:
    vocab = tokenizer.get_vocab()
    x_id = vocab.get("X", None)
    # Fallback if not found directly in the vocab mapping
    if x_id is None:
        xid_conv = tokenizer.convert_tokens_to_ids("X")
        x_id = int(xid_conv) if xid_conv is not None else None
    return {
        "pad": tokenizer.pad_token_id,
        "unk": tokenizer.unk_token_id,
        "X": x_id,
    }


def collect_counts_per_sequence(data: List[dict], ids_key: str, token_id: int) -> List[int]:
    counts: List[int] = []
    for el in data:
        if ids_key not in el["samples"]:
            continue
        ids: torch.Tensor = el["samples"][ids_key]
        if ids.ndim != 2:
            raise ValueError(f"Expected 2D ids, got {ids.shape}")
        # Count occurrences per-row
        row_counts = (ids == token_id).sum(dim=1).tolist()
        counts.extend(int(c) for c in row_counts)
    return counts


def describe_counts(counts: List[int]) -> Dict[str, float]:
    if not counts:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    # Use population stdev to avoid errors with single-element lists
    return {
        "mean": float(statistics.mean(counts)),
        "median": float(statistics.median(counts)),
        "std": float(statistics.pstdev(counts)),
        "min": float(min(counts)),
        "max": float(max(counts)),
    }


def find_padding_in_middle(data: List[dict], ids_key: str, pad_id: Optional[int]) -> List[Tuple[int, int]]:
    if pad_id is None:
        return []
    offenders: List[Tuple[int, int]] = []
    for month_idx, el in enumerate(data):
        if ids_key not in el["samples"]:
            continue
        ids: torch.Tensor = el["samples"][ids_key]
        if ids.ndim != 2:
            raise ValueError(f"Expected 2D ids, got {ids.shape}")
        N, L = ids.shape
        for i in range(N):
            row = ids[i]
            is_pad = (row == pad_id)
            # indices of non-pad tokens
            non_pad_indices = torch.nonzero(~is_pad, as_tuple=False).view(-1)
            if non_pad_indices.numel() == 0:
                # all pad -> no middle padding issue
                continue
            start = int(non_pad_indices.min().item())
            end = int(non_pad_indices.max().item())
            if is_pad[start:end+1].any():
                offenders.append((month_idx, i))
    return offenders


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check tokens across dataset for ESM2 and ProGen2 and detect non-standard AA tokens (ignoring attention mask)")
    parser.add_argument("--data", default="data/spikeprot0430/virus_tokenized_data_for_tde_downsampled100_filtered_test.pt", help="Path to aggregated dataset .pt")
    parser.add_argument("--esm_name", default="facebook/esm2_t6_8M_UR50D", help="ESM model id or local path")
    parser.add_argument("--progen_name", default="hugohrban/progen2-base", help="ProGen2 model id or local path")
    parser.add_argument("--sort", choices=["id", "count"], default="id", help="Sort token table by id or by count")
    args = parser.parse_args()

    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"Dataset not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading dataset: {data_path}")
    data = load_dataset(data_path)

    def run_and_print(label: str, tokenizer: PreTrainedTokenizerBase, ids_key: str) -> None:
        print(f"\n===== {label} analysis ({ids_key}) =====")
        # Print token inventory
        print("\nTokenizer vocabulary (token -> id):")
        vocab = tokenizer.get_vocab()  # token -> id
        for tok, tid in sorted(vocab.items(), key=lambda kv: kv[1]):
            print(f"  {tok:<20} -> {tid}")

        print("\nCounting tokens and checking anomalies (ignoring attention mask) ...")
        counts_all, summary, anomalies = analyze_dataset(data, tokenizer, ids_key)

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

        # Special tokens stats: <pad>, <unk>, X
        ids_of_interest = get_token_ids(tokenizer)
        print("\nSpecial-token per-sequence stats:")
        for name in ["pad", "unk", "X"]:
            tid = ids_of_interest.get(name)
            if tid is None:
                print(f"  {name}: token not present in tokenizer")
                continue
            total_count = int(counts_all.get(tid, 0))
            per_seq_counts = collect_counts_per_sequence(data, ids_key, tid)
            desc = describe_counts(per_seq_counts)
            num_sequences_with_token = int(sum(1 for c in per_seq_counts if c > 0))
            print(f"  {name}: total={total_count} sequences_with_token={num_sequences_with_token} "
                  f"mean={desc['mean']:.3f} median={desc['median']:.3f} std={desc['std']:.3f} "
                  f"min={desc['min']:.0f} max={desc['max']:.0f}")

        # Padding in the middle check
        pad_id = ids_of_interest.get("pad")
        offenders = find_padding_in_middle(data, ids_key, pad_id)
        if offenders:
            print(f"\nPadding-in-the-middle detected in {len(offenders)} sequences (showing up to 10 indices):")
            for month_idx, seq_idx in offenders[:10]:
                print(f"  month_idx={month_idx} seq_idx={seq_idx}")
        else:
            print("\nNo padding-in-the-middle detected.")

        if anomalies:
            print("\nExamples of sequences with non-standard inner tokens (showing up to 10):")
            for month_idx, seq_idx, offending in anomalies[:10]:
                ids = data[month_idx]["samples"][ids_key][seq_idx]
                text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
                print(f"  month_idx={month_idx} seq_idx={seq_idx} offending_ids={offending}")
                print(f"    decoded: {text}")
        else:
            print("\nNo non-standard tokens found (excluding BOS at start and EOS at end).")

    # ESM analysis
    esm_model_id = resolve_local_or_repo(args.esm_name)
    esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_id)
    run_and_print("ESM2", esm_tokenizer, "esm_input_ids")

    # ProGen2 analysis (if present)
    progen_present = any("progen_input_ids" in el.get("samples", {}) for el in data)
    if progen_present:
        progen_model_id = resolve_local_or_repo(args.progen_name)
        progen_tokenizer = AutoTokenizer.from_pretrained(progen_model_id, trust_remote_code=True)
        run_and_print("ProGen2", progen_tokenizer, "progen_input_ids")
    else:
        print("\nProGen2 input ids not found in dataset; skipping ProGen2 analysis.")


if __name__ == "__main__":
    main()


