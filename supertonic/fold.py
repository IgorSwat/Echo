"""Latent folding: pack `factor` consecutive frames into the channel dim.

The Supertonic vector estimator operates on folded latents of shape
``(B, T / factor, latent_dim * factor)`` — the folding is a data-side reshape
(matching the original, where the autoencoder/vocoder handle it), so both the
training loop and the sampler fold/unfold around the model.
"""

from typing import Optional

import torch
import torch.nn.functional as F


def fold_latent(
    x: torch.Tensor,                                                # (B, T, C)
    mask: Optional[torch.Tensor] = None,                            # (B, T) bool or None
    factor: int = 6,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Fold (B, T, C) -> (B, ceil(T/factor), C*factor), zero-padding the tail.

    Channel packing matches the original vocoder's unfold order:
    ``z[b, s, c*factor + f] == x[b, s*factor + f, c]``.

    The mask is folded with ``any``: a folded frame is valid if at least one of
    its sub-frames is valid (the original quantizes durations to the folded
    grid with a ceil, so the last — possibly partial — folded frame is kept).
    """
    B, T, C = x.shape
    S = -(-T // factor)                                             # ceil(T / factor)
    pad = S * factor - T
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        if mask is not None:
            mask = F.pad(mask, (0, pad), value=False)

    z = x.view(B, S, factor, C).permute(0, 1, 3, 2).reshape(B, S, C * factor)

    if mask is None:
        return z, None
    return z, mask.view(B, S, factor).any(dim=-1)                   # (B, S) bool


def unfold_latent(
    z: torch.Tensor,                                                # (B, S, C*factor)
    factor: int = 6,
) -> torch.Tensor:
    """Inverse of fold_latent: (B, S, C*factor) -> (B, S*factor, C)."""
    B, S, CF = z.shape
    C = CF // factor
    return z.view(B, S, C, factor).permute(0, 1, 3, 2).reshape(B, S * factor, C)
