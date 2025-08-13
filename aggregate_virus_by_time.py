import argparse
import os
from typing import Any, Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate entries in a virus_tokenized_data.pt by month (first 7 chars of 'time-loc'), "
            "stacking tensors under 'samples' along the batch dimension and concatenating 'raw_texts'."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input tokenized dataset .pt file (e.g., data/spikeprot0430/virus_tokenized_data.pt)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the aggregated .pt file",
    )

    return parser.parse_args()


def normalize_time_key(time_loc_value: str):
    # Drop entries where the 7th character is not a digit, or the string is too short
    if not isinstance(time_loc_value, str) or len(time_loc_value) < 7:
        return None
    if not time_loc_value[6].isdigit():
        return None
    return time_loc_value[:7]


def validate_and_cat_tensors(tensors: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
    if len(tensors) == 0:
        raise ValueError("No tensors provided for concatenation")
    reference_shape = tensors[0].shape[1:]
    reference_dtype = tensors[0].dtype
    reference_device = tensors[0].device
    for idx, tensor in enumerate(tensors):
        if tensor.shape[1:] != reference_shape:
            raise ValueError(
                f"Tensor at index {idx} has non-matching shape {tensor.shape}; expected (*, {reference_shape})"
            )
        if tensor.dtype != reference_dtype:
            raise ValueError(
                f"Tensor at index {idx} has dtype {tensor.dtype}; expected {reference_dtype}"
            )
        if tensor.device != reference_device:
            raise ValueError(
                f"Tensor at index {idx} is on device {tensor.device}; expected {reference_device}"
            )
    return torch.cat(tensors, dim=dim)


def aggregate_by_time(tokenized_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Accumulate per month
    month_to_accumulator: Dict[str, Dict[str, Any]] = {}

    for entry_idx, entry in enumerate(tokenized_data):
        if "time-loc" not in entry:
            raise KeyError(
                f"Entry {entry_idx} missing required key 'time-loc'. Keys present: {list(entry.keys())}"
            )
        if "samples" not in entry:
            raise KeyError(
                f"Entry {entry_idx} missing required key 'samples'. Keys present: {list(entry.keys())}"
            )
        if "raw_texts" not in entry:
            raise KeyError(
                f"Entry {entry_idx} missing required key 'raw_texts'. Keys present: {list(entry.keys())}"
            )

        month_key = normalize_time_key(entry["time-loc"])  # yyyy-mm
        if month_key is None:
            # Skip entries with malformed month (e.g., non-digit at position 6)
            continue
        samples: Dict[str, torch.Tensor] = entry["samples"]
        raw_texts: List[str] = entry["raw_texts"]

        if month_key not in month_to_accumulator:
            month_to_accumulator[month_key] = {
                "samples_lists": {k: [] for k in samples.keys()},
                "raw_texts": [],
            }

        accumulator = month_to_accumulator[month_key]

        # Append tensors per sample key
        for sample_key, tensor in samples.items():
            if sample_key not in accumulator["samples_lists"]:
                accumulator["samples_lists"][sample_key] = []
            accumulator["samples_lists"][sample_key].append(tensor)

        # Extend raw_texts
        accumulator["raw_texts"].extend(list(raw_texts))

    # Build final aggregated list
    aggregated: List[Dict[str, Any]] = []

    for month_key in sorted(month_to_accumulator.keys()):
        accumulator = month_to_accumulator[month_key]
        samples_lists: Dict[str, List[torch.Tensor]] = accumulator["samples_lists"]

        # Concatenate along batch dimension
        concatenated_samples: Dict[str, torch.Tensor] = {}
        for sample_key, tensor_list in samples_lists.items():
            concatenated_samples[sample_key] = validate_and_cat_tensors(tensor_list, dim=0)

        aggregated.append(
            {
                "samples": concatenated_samples,
                "time": month_key,
                "raw_texts": accumulator["raw_texts"],
            }
        )

    return aggregated


def main() -> None:
    args = parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(
            f"Output file already exists: {args.output}."
        )

    print(f"Loading tokenized data from: {args.input}")
    tokenized_data = torch.load(args.input, map_location="cpu")

    if not isinstance(tokenized_data, list):
        raise TypeError(
            f"Expected input to be a list of entries, got type: {type(tokenized_data)}"
        )
    if len(tokenized_data) == 0:
        raise ValueError("Input tokenized data is empty")

    print("Aggregating entries by month (yyyy-mm)...")
    aggregated = aggregate_by_time(tokenized_data)

    print(f"Saving aggregated data to: {args.output}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(aggregated, args.output)
    print("Done.")


if __name__ == "__main__":
    main()


