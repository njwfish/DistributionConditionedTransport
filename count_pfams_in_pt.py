import argparse
from typing import Any

import torch


def count_pfams(obj: Any) -> int:
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        if "pfams" in obj and isinstance(obj["pfams"], (list, tuple)):
            return len(obj["pfams"])
        if "pfam" in obj and isinstance(obj["pfam"], (list, tuple)):
            return len(obj["pfam"])
        return len(obj)
    raise TypeError(f"Unsupported .pt structure: {type(obj).__name__}")


def main() -> None:
    data = torch.load("data/pfam/pfam_tokenized_data_clan_eval.pt", map_location="cpu")
    num_pfams = count_pfams(data)
    print(num_pfams)


if __name__ == "__main__":
    main()
