import torch
from typing import Optional


def normalize_latent(latent: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize a batch of latent vectors along the last dimension.

    Args:
        latent: Tensor shaped [..., latent_dim]
        eps: Small constant for numerical stability

    Returns:
        Tensor with the same shape as `latent`, where each vector along the last
        dimension has unit norm (or zero vector remains zero).
    """
    if latent is None:
        raise ValueError("normalize_latent expected a Tensor, got None")
    denom = torch.norm(latent, dim=-1, keepdim=True).clamp_min(eps)
    return latent / denom



def expand_latent_to_batch(latent: Optional[torch.Tensor], reference_batch: torch.Tensor) -> Optional[torch.Tensor]:
    """
    Expand a latent tensor along the batch dimension to match a reference batch size.

    The typical use case is when latents are provided per-set and inputs are
    flattened per-sample. This function repeats each latent vector the required
    number of times to match the first-dimension size of `reference_batch`.

    Args:
        latent: Tensor shaped [num_sets, latent_dim] or [batch_size, latent_dim], or None
        reference_batch: Tensor whose first dimension defines the desired batch size

    Returns:
        Tensor shaped [batch_size, latent_dim], or None if `latent` is None.
    """
    if latent is None:
        return None

    batch_size = reference_batch.shape[0]
    if latent.shape[0] == batch_size:
        return latent

    if batch_size % latent.shape[0] != 0:
        raise ValueError(
            f"Cannot expand latent of batch {latent.shape[0]} to {batch_size}. "
            f"Expected reference batch to be a multiple of latent batch."
        )

    repeats = batch_size // latent.shape[0]
    return latent.unsqueeze(1).repeat(1, repeats, 1).view(-1, latent.shape[-1])