import os
from typing import Optional


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _default_assets_dir() -> str:
    # Default local directory to store HF assets
    return os.path.join(_project_root(), "assets", "hf")


def sanitize_repo_id(repo_id: str) -> str:
    """Convert a HF repo id like 'org/name' to a safe local directory name."""
    return repo_id.replace("/", "__")


def get_local_repo_dir(repo_id: str, base_dir: Optional[str] = None) -> str:
    """
    Return the expected local directory for a given repo id. This matches the
    location used by the pre-download script.
    """
    base = base_dir or os.environ.get("HF_LOCAL_DIR") or _default_assets_dir()
    return os.path.join(base, sanitize_repo_id(repo_id))


def resolve_local_or_repo(repo_id: str, base_dir: Optional[str] = None) -> str:
    """
    If a locally downloaded directory for the HF repo exists, return that path.
    Otherwise, return the original repo id so Transformers can download it online.
    """
    local_path = get_local_repo_dir(repo_id, base_dir)
    # We consider it present if directory exists and has at least a config file
    if os.path.isdir(local_path):
        return local_path
    return repo_id


