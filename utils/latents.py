import torch


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


