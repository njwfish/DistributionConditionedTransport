#!/usr/bin/env python3

import argparse
import csv
import hashlib
import sys
from typing import Iterable, Optional, List, Tuple, Set, Union, Dict

import torch


def _normalize_sequence(sequence: str) -> str:
    """Normalize a raw sequence string for comparison.

    - Strips surrounding whitespace
    - Uppercases letters so 'x' and 'X' are treated the same
    """
    return sequence.strip().upper()


def _iter_timepoints(dataset: list, limit: Optional[int] = None) -> Iterable[Tuple[int, str, List[str]]]:
    """Yield (index, time, raw_texts) tuples from the dataset.

    Safely handles missing keys by substituting empty values.
    """
    max_index = len(dataset) if limit is None else min(limit, len(dataset))
    for index in range(max_index):
        item = dataset[index]
        if not isinstance(item, dict):
            # Skip non-dict entries defensively
            yield index, f"<non-dict:{type(item).__name__}>", []
            continue
        time_value = item.get("time", "")
        time_str = str(time_value) if time_value is not None else ""
        raw_texts = item.get("raw_texts", [])
        if not isinstance(raw_texts, (list, tuple)):
            raw_texts = []
        # Keep only strings, ignore non-strings defensively
        raw_texts = [s for s in raw_texts if isinstance(s, str)]
        yield index, time_str, raw_texts  # type: ignore[return-value]


def _are_compatible(rep_chars: List[str], seq_chars: List[str]) -> bool:
    """Return True if representative and sequence are identical under 'X' wildcard semantics.

    They are considered identical if for every position i, when both characters are not 'X',
    the characters are equal. Otherwise, 'X' acts as a wildcard and does not cause a conflict.
    """
    # Same length is assumed by caller
    for a, b in zip(rep_chars, seq_chars):
        if a != 'X' and b != 'X' and a != b:
            return False
    return True


def process_dataset(
    input_path: str,
    use_hash: bool = True,  # kept for CLI compatibility; unused with wildcard logic
    limit: Optional[int] = None,
    csv_path: Optional[str] = None,
) -> None:
    """Process the dataset and print per-timepoint stats.

    For each timepoint (dataset element), prints:
      time=<str>, unique_so_far=<int>, new_unique=<int>, total_this_timepoint=<int>, frac_new=<float>
    """
    dataset = torch.load(input_path, map_location="cpu")
    if not isinstance(dataset, list):
        raise TypeError(f"Expected root object to be list, got {type(dataset)}")

    # Exemplars of unique sequences under 'X' wildcard semantics, bucketed by length.
    # We keep earlier exemplars unchanged to avoid over-constraining matches.
    representatives_by_length: Dict[int, List[str]] = {}

    csv_writer: Optional[csv.writer] = None
    csv_file = None
    if csv_path:
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "index",
                "time",
                "unique_so_far",
                "new_unique_this_timepoint",
                "total_sequences_this_timepoint",
                "fraction_new_this_timepoint",
            ]
        )

    try:
        for index, time_str, raw_texts in _iter_timepoints(dataset, limit=limit):
            total_this_timepoint = len(raw_texts)
            new_unique = 0

            for raw_seq in raw_texts:
                seq = _normalize_sequence(raw_seq)
                if not seq:
                    # Skip empty strings
                    continue
                length = len(seq)
                reps = representatives_by_length.setdefault(length, [])

                seq_chars = list(seq)
                matched = False
                # Linear scan over exemplars of the same length; treat as duplicate on first compatible
                for rep in reps:
                    if _are_compatible(list(rep), seq_chars):
                        matched = True
                        break

                if not matched:
                    # New unique exemplar (keep as immutable string)
                    reps.append(seq)
                    new_unique += 1

            unique_so_far = sum(len(v) for v in representatives_by_length.values())
            frac_new = (new_unique / total_this_timepoint) if total_this_timepoint > 0 else 0.0

            # Concise, stable output
            print(
                f"time={time_str}, unique_so_far={unique_so_far}, new_unique={new_unique}, "
                f"total_this_timepoint={total_this_timepoint}, frac_new={frac_new:.6f}",
                flush=True,
            )

            if csv_writer is not None:
                csv_writer.writerow(
                    [
                        index,
                        time_str,
                        unique_so_far,
                        new_unique,
                        total_this_timepoint,
                        f"{frac_new:.8f}",
                    ]
                )
    finally:
        if csv_file is not None:
            csv_file.close()


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a torch-loaded dataset for duplicate protein sequences across timepoints. "
            "Prints cumulative unique counts and new uniques per timepoint."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to the .pt dataset file",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Use raw strings in memory instead of SHA-256 hashes (higher memory).",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Process only the first N timepoints (for quick inspection).",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional path to write results as CSV (in addition to printing).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    process_dataset(
        input_path=args.input_path,
        use_hash=not args.no_hash,
        limit=args.max_items,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    main()


