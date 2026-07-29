from typing import Optional

import torch


def apply_mask(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero out padded positions: (B, T, D) x (B, T) bool -> (B, T, D)."""
    if mask is None:
        return x
    return x * mask.unsqueeze(-1).to(x.dtype)
