import argparse
import os
import logging
from typing import List, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute month indices for the aggregated virus dataset (used for time-aware sampling).\n"
            "Loads data/spikeprot0430/virus_tokenized_data_for_tde.pt (or a given --input),\n"
            "reads 'time' per element (yyyy-mm), creates a complete chronological month list\n"
            "from earliest to latest (including missing months), and saves an .npz with:\n"
            "- dataset_indices: np.arange(num_elements)\n"
            "- time_indices: integer month indices aligned with the chronological list"
        )
    )
    parser.add_argument(
        "--input",
        default="data/spikeprot0430/virus_tokenized_data_for_tde.pt",
        help=(
            "Path to aggregated dataset .pt file containing a list of dicts with key 'time' (yyyy-mm).\n"
            "Default: data/spikeprot0430/virus_tokenized_data_for_tde.pt"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/spikeprot0430/virus_month_index.npz",
        help=(
            "Path to write the output .npz file. It will contain 'dataset_indices' and 'time_indices'.\n"
            "Default: data/spikeprot0430/virus_month_index.npz"
        ),
    )
    parser.add_argument(
        "--log-file",
        default="/orcd/archive/abugoot/001/Projects/paolo/CoupledDistributionEmbeddings/indices_generation_log.log",
        help=(
            "Optional path to write a log file with live progress updates. "
            "Default: <output>.log"
        ),
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1000,
        help=(
            "How often to log progress (in number of items). Default: 1000"
        ),
    )
    return parser.parse_args()


def yyyymm_to_tuple(yyyy_mm: str) -> Tuple[int, int]:
    if not isinstance(yyyy_mm, str) or len(yyyy_mm) < 7:
        raise ValueError(f"Invalid yyyy-mm string: {yyyy_mm}")
    year = int(yyyy_mm[:4])
    month = int(yyyy_mm[5:7])
    if not (1 <= month <= 12):
        raise ValueError(f"Month must be in 1..12 for '{yyyy_mm}'")
    return year, month


def to_abs_month(year: int, month: int) -> int:
    # Convert to absolute month count to make continuous ranges easy
    return year * 12 + (month - 1)


def build_full_month_range(times: List[str]) -> Tuple[List[str], dict]:
    # Convert all times to absolute month values
    abs_months = []
    for t in times:
        y, m = yyyymm_to_tuple(t)
        abs_months.append(to_abs_month(y, m))

    start_abs = min(abs_months)
    end_abs = max(abs_months)

    # Create chronological list of yyyy-mm from earliest to latest inclusive
    chronological_months: List[str] = []
    abs_to_index: dict = {}
    for idx, am in enumerate(range(start_abs, end_abs + 1)):
        year = am // 12
        month = (am % 12) + 1
        label = f"{year:04d}-{month:02d}"
        chronological_months.append(label)
        abs_to_index[am] = idx

    return chronological_months, abs_to_index


def main() -> None:
    args = parse_args()

    # Configure logging to file if requested
    log_file = args.log_file
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8")],
    )
    logging.info("Starting month index build")
    logging.info(f"Input: {args.input}")
    logging.info(f"Output: {args.output}")
    logging.info(f"Log interval: {args.log_interval}")

    data = torch.load(args.input, map_location="cpu")
    if not isinstance(data, list):
        raise TypeError(
            f"Expected a list loaded from {args.input}, got {type(data)}."
        )
    if len(data) == 0:
        raise ValueError("Input dataset is empty.")
    logging.info(f"Loaded dataset with {len(data)} elements")

    # Extract 'time' (yyyy-mm) per element
    times: List[str] = []
    for i, item in enumerate(data):
        if "time" not in item:
            raise KeyError(
                f"Element {i} missing 'time' key. Keys present: {list(item.keys())}"
            )
        times.append(item["time"])  # yyyy-mm
        if (i + 1) % args.log_interval == 0 or (i + 1) == len(data):
            logging.info(
                f"Scanned times: {i + 1}/{len(data)} ({(i + 1) / len(data):.2%})"
            )

    # Build full inclusive chronological range even if some months are missing in data
    chronological_months, abs_to_index = build_full_month_range(times)
    logging.info(
        f"Chronological month range built with {len(chronological_months)} months: "
        f"{chronological_months[0]} .. {chronological_months[-1]}"
    )
    num_time_points = np.int64(len(abs_to_index))
    logging.info(f"num_time_points (unique months in range): {int(num_time_points)}")

    # Map each element to its index in the chronological month list
    time_indices = np.zeros(len(times), dtype=np.int64)
    for i, t in enumerate(times):
        y, m = yyyymm_to_tuple(t)
        am = to_abs_month(y, m)
        time_indices[i] = abs_to_index[am]
        if (i + 1) % args.log_interval == 0 or (i + 1) == len(times):
            logging.info(
                f"Mapped indices: {i + 1}/{len(times)} ({(i + 1) / len(times):.2%})"
            )

    dataset_indices = np.arange(len(times), dtype=np.int64)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        dataset_indices=dataset_indices,
        time_indices=time_indices,
        num_time_points=num_time_points,
    )
    print(
        f"Saved month indices to {args.output} (num_elements={len(times)}, months={chronological_months[0]}..{chronological_months[-1]})"
    )
    logging.info(
        f"Saved month indices to {args.output} (num_elements={len(times)}, months={chronological_months[0]}..{chronological_months[-1]})"
    )


if __name__ == "__main__":
    main()


