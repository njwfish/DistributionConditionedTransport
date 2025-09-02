#!/usr/bin/env python3
import os
import argparse
from typing import List

from huggingface_hub import snapshot_download

from utils.hf_local import get_local_repo_dir, sanitize_repo_id, _default_assets_dir


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def download_repo(repo_id: str, local_base: str) -> str:
    local_dir = get_local_repo_dir(repo_id, base_dir=local_base)
    ensure_dir(local_dir)
    # Download into the final target directory. snapshot_download returns a path
    # to the snapshot. If it differs from local_dir, move/merge is handled by HF.
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False, resume_download=True)
    return local_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download Hugging Face models/tokenizers used by the project")
    parser.add_argument(
        "--local-dir",
        type=str,
        default=os.environ.get("HF_LOCAL_DIR", _default_assets_dir()),
        help="Local base directory to store downloaded HF assets",
    )
    parser.add_argument(
        "--extra",
        type=str,
        nargs="*",
        default=[],
        help="Additional HF repo ids (org/name) to download",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_base = args.local_dir
    ensure_dir(local_base)

    # Core repos used by virus experiment (config/experiment/virus.yaml)
    required_repos: List[str] = [
        # "facebook/esm2_t6_8M_UR50D",  # ESM2 model and tokenizer
        # "hugohrban/progen2-small",    # default in current virus config
        "hugohrban/progen2-base",      # download progen2-base only
    ]

    # Include extras from CLI
    for extra in args.extra:
        if extra and extra not in required_repos:
            required_repos.append(extra)

    print(f"Downloading {len(required_repos)} repos to {local_base} ...")
    for repo_id in required_repos:
        target = download_repo(repo_id, local_base)
        print(f"- [{repo_id}] -> {target}")

    print("Done. Set HF_LOCAL_DIR to this folder to force local usage.")


if __name__ == "__main__":
    main()


