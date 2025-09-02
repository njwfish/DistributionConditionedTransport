#!/usr/bin/env python3
import argparse
import os
import sys
import gc
from typing import Dict, List, Optional

import torch


def extract_year_month(time_loc: str) -> Optional[str]:
    """
    Extract YYYY-MM from a time-loc string.

    Rules per spec:
    - Keep only entries where the 7th character (index 6) is a digit.
    - Return the first 7 characters (YYYY-MM).
    """
    if not isinstance(time_loc, str):
        return None
    if len(time_loc) < 7:
        return None
    # Filter out single-digit months like "2024-8-..." where index 6 is '-'
    if not time_loc[6].isdigit():
        return None
    ym = time_loc[:7]
    # Optional sanity check for YYYY-MM
    if not (ym[0:4].isdigit() and ym[4] == '-' and ym[5:7].isdigit()):
        return None
    return ym


def aggregate_by_month(
    data: List[dict],
    sample_keys: Optional[List[str]] = None,
) -> List[dict]:
    """
    Aggregate dataset elements by YYYY-MM extracted from 'time-loc'.

    Returns a new list of dicts with keys 'samples', 'time', 'raw_texts'.
    - 'samples': dict with tensors concatenated along dim 0 for matching months
    - 'time': 'YYYY-MM'
    - 'raw_texts': list concatenated across matching months
    """
    # Map time -> aggregation buffers
    buffers: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    raw_texts_acc: Dict[str, List[str]] = {}

    for idx, el in enumerate(data):
        if not isinstance(el, dict):
            continue
        time_loc = el.get('time-loc')
        ym = extract_year_month(time_loc)
        if ym is None:
            continue

        samples = el.get('samples')
        if not isinstance(samples, dict):
            continue
        if sample_keys is None:
            sample_keys = list(samples.keys())

        if ym not in buffers:
            buffers[ym] = {k: [] for k in sample_keys}
            raw_texts_acc[ym] = []

        # Accumulate tensors for each expected sample key
        for k in sample_keys:
            t = samples.get(k)
            if isinstance(t, torch.Tensor):
                buffers[ym][k].append(t)
            else:
                raise ValueError(f"Expected tensor for key '{k}', got {type(t)} at element {idx}")

        # Accumulate raw texts
        rtexts = el.get('raw_texts', [])
        if isinstance(rtexts, list):
            raw_texts_acc[ym].extend(rtexts)
        else:
            raise ValueError(f"Expected list for 'raw_texts', got {type(rtexts)} at element {idx}")

    # Build final aggregated list sorted by time ascending
    result: List[dict] = []
    for ym in sorted(buffers.keys()):
        groups = buffers[ym]
        concatenated = {k: torch.cat(groups[k], dim=0) if len(groups[k]) > 0 else None for k in groups}
        # Ensure all tensors are present
        for k, v in concatenated.items():
            if v is None:
                raise ValueError(f"No tensors accumulated for key '{k}' and time '{ym}'")
        result.append({
            'samples': concatenated,
            'time': ym,
            'raw_texts': raw_texts_acc[ym],
        })

    return result


def build_downsampled_dataset(aggregated: List[dict], num_samples: int = 100) -> List[dict]:
    """
    For each aggregated element, sample `num_samples` sequences and subsample
    both tensors in 'samples' and the 'raw_texts' list accordingly.
    Assumes each tensor in 'samples' has shape [N, ...].
    """
    downsampled: List[dict] = []
    for el in aggregated:
        n = el['samples']['esm_input_ids'].shape[0]
        idx = torch.randperm(n)[:num_samples]
        subsampled_samples = {k: v.index_select(0, idx) for k, v in el['samples'].items()}
        subsampled_texts = [el['raw_texts'][i] for i in idx.tolist()]
        downsampled.append({
            'samples': subsampled_samples,
            'time': el['time'],
            'raw_texts': subsampled_texts,
        })
    return downsampled


def derive_downsampled_path(output_path: str, num_samples: int = 100) -> str:
    root, ext = os.path.splitext(output_path)
    if not ext:
        ext = '.pt'
    return f"{root}_downsampled{num_samples}{ext}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate dataset entries by year-month across locations.")
    parser.add_argument("--input", required=True, help="Path to input .pt file")
    parser.add_argument("--output", required=True, help="Path to output .pt file")
    parser.add_argument("--output-downsampled", required=False, help="Path to output downsampled .pt file (100 samples per time-point). If omitted, appends _downsampled100 to --output path.")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    print("Loading input dataset ...", flush=True)
    data = torch.load(input_path, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(f"Expected root object to be list, got {type(data)}")

    print(f"Loaded {len(data)} elements. Aggregating ...", flush=True)
    aggregated = aggregate_by_month(data)
    # Post-aggregation filter: drop months with fewer than 5000 sequences
    aggregated = [el for el in aggregated if el['samples']['esm_input_ids'].shape[0] >= 4000]

    # Free original data as early as possible to reduce peak memory usage
    del data
    gc.collect()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Saving aggregated dataset to: {output_path}", flush=True)
    torch.save(aggregated, output_path)

    # Build and save downsampled dataset (100 samples per time-point)
    downsampled = build_downsampled_dataset(aggregated, num_samples=100)
    output_downsampled_path = args.output_downsampled or derive_downsampled_path(output_path, num_samples=100)
    os.makedirs(os.path.dirname(output_downsampled_path), exist_ok=True)
    print(f"Saving downsampled (100) dataset to: {output_downsampled_path}", flush=True)
    torch.save(downsampled, output_downsampled_path)

    print("\nAggregation complete. Summary:", flush=True)
    for el in aggregated:
        t = el['time']
        shape = el['samples']['esm_input_ids'].shape
        print(f"time={t}\tesm_input_ids.shape={tuple(shape)}")


if __name__ == "__main__":
    main()


