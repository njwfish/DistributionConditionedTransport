import torch
from typing import Any, Optional, Tuple


def build_condition_tuple(
    batch: dict,
    device: torch.device,
    condition_type: str,
) -> Optional[Tuple[torch.Tensor, ...]]:
    """
    Build a tuple of conditioning scalars from a dataloader batch according to the
    requested condition_type.

    - condition_type == 'none': return None
    - condition_type == 'scalar_d': return (d,)
    - condition_type == 'index_pair': return (source_idx, target_idx)

    Values are returned as tensors on the given device. Handles values that may
    be ints/floats or already tensors, and supports batched tensors.
    """

    if condition_type is None or condition_type == "none":
        return None

    if condition_type == "scalar_d":
        d_val: Any = batch.get("d")
        if d_val is None:
            return None
        d_tensor = torch.as_tensor(d_val, dtype=torch.float32, device=device)
        return (d_tensor,)

    if condition_type == "index_pair":
        s_val: Any = batch.get("source_idx")
        t_val: Any = batch.get("target_idx")
        if s_val is None or t_val is None:
            return None
        s_tensor = torch.as_tensor(s_val, dtype=torch.float32, device=device)
        t_tensor = torch.as_tensor(t_val, dtype=torch.float32, device=device)
        return (s_tensor, t_tensor)

    # Unknown
    return None


def is_conditioned(condition_type: Optional[str]) -> bool:
    return bool(condition_type) and condition_type != "none"


