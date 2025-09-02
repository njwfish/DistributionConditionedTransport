import argparse
import os
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load an aggregated dataset (from aggregate_virus_by_time.py) and randomly subsample up to\n"
            "max_samples_per_time entries per time bucket, consistently across 'samples' and 'raw_texts'."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input aggregated .pt file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the reduced aggregated .pt file",
    )
    parser.add_argument(
        "--max-samples-per-time",
        type=int,
        default=50,
        help="Maximum number of samples to keep per time bucket (default: 50)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible subsampling (default: 0)",
    )
    return parser.parse_args()


def validate_entry_schema(entry: Dict[str, Any]) -> None:
    required_keys = {"samples", "time", "raw_texts"}
    missing = required_keys.difference(entry.keys())
    if missing:
        raise KeyError(f"Entry missing required keys: {missing}. Present keys: {list(entry.keys())}")

    if not isinstance(entry["samples"], dict):
        raise TypeError("Entry['samples'] must be a dict of torch.Tensors")
    if not isinstance(entry["raw_texts"], list):
        raise TypeError("Entry['raw_texts'] must be a list of strings")


def determine_batch_size(samples: Dict[str, torch.Tensor]) -> int:
    if len(samples) == 0:
        raise ValueError("'samples' dict is empty")
    first_tensor = next(iter(samples.values()))
    if not isinstance(first_tensor, torch.Tensor):
        raise TypeError("Values in 'samples' must be torch.Tensors")
    batch_size = first_tensor.shape[0]
    for sample_key, tensor in samples.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"'samples[{sample_key}]' must be a torch.Tensor")
        if tensor.shape[0] != batch_size:
            raise ValueError(
                f"All tensors in 'samples' must share the same batch dimension."
                f" Mismatch for key '{sample_key}': {tensor.shape[0]} vs {batch_size}"
            )
    return batch_size


def subsample_indices(num_items: int, max_items: int) -> torch.Tensor:
    if num_items < max_items:
        raise ValueError(f"num_items ({num_items}) must be greater than max_items ({max_items})")
        return torch.arange(num_items)
    permutation = torch.randperm(num_items)
    return permutation[:max_items]


def subsample_entry(entry: Dict[str, Any], max_samples_per_time: int) -> Dict[str, Any]:
    validate_entry_schema(entry)
    samples: Dict[str, torch.Tensor] = entry["samples"]
    raw_texts: List[str] = entry["raw_texts"]

    batch_size = determine_batch_size(samples)
    if len(raw_texts) != batch_size:
        raise ValueError(
            f"Length of 'raw_texts' ({len(raw_texts)}) must match batch size from 'samples' ({batch_size})"
        )

    index_tensor = subsample_indices(batch_size, max_samples_per_time)

    if index_tensor.numel() == batch_size:
        # No subsampling needed; return as-is
        return entry

    # Index all sample tensors consistently along batch dimension
    reduced_samples: Dict[str, torch.Tensor] = {
        key: tensor.index_select(dim=0, index=index_tensor)
        for key, tensor in samples.items()
    }
    # Index raw_texts
    reduced_raw_texts: List[str] = [raw_texts[i] for i in index_tensor.tolist()]

    return {
        "samples": reduced_samples,
        "time": entry["time"],
        "raw_texts": reduced_raw_texts,
    }


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(f"Output file already exists: {args.output}")

    torch.manual_seed(args.seed)

    print(f"Loading aggregated data from: {args.input}")
    aggregated_data = torch.load(args.input, map_location="cpu")

    if not isinstance(aggregated_data, list):
        raise TypeError(f"Expected input to be a list of entries, got type: {type(aggregated_data)}")
    if len(aggregated_data) == 0:
        raise ValueError("Input aggregated data is empty")

    print(
        f"Subsampling up to {args.max_samples_per_time} samples per time bucket (total buckets: {len(aggregated_data)})"
    )

    reduced_entries: List[Dict[str, Any]] = []
    for entry_idx, entry in enumerate(aggregated_data):
        validate_entry_schema(entry)
        before_n = determine_batch_size(entry["samples"])  # also validates alignment
        reduced_entry = subsample_entry(entry, args.max_samples_per_time)
        after_n = determine_batch_size(reduced_entry["samples"])
        reduced_entries.append(reduced_entry)
        print(
            f"[{entry_idx+1}/{len(aggregated_data)}] time={entry['time']} before={before_n} after={after_n}"
        )

    print(f"Saving reduced aggregated data to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(reduced_entries, args.output)
    print("Done.")


if __name__ == "__main__":
    main()


