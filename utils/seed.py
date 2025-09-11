import os
import random
from typing import Optional

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: The random seed to set.
        deterministic: If True, enable deterministic algorithms where possible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    # cuBLAS deterministic workspace (PyTorch docs recommendation)
    # Set before CUDA context initialization
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def dataloader_seed_worker(worker_id: int) -> None:
    """Seed function for DataLoader workers.

    Uses the initial torch seed to derive per-worker seeds and set NumPy and Python RNGs.
    """
    # torch.initial_seed() returns a large 64-bit number; reduce to 32 bits for numpy/random
    worker_seed = (torch.initial_seed() % 2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Create a torch.Generator seeded for DataLoader reproducibility.

    Returning None keeps default behavior.
    """
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


