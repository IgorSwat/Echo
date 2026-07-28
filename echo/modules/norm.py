from echo import config

from typing import Optional

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization: normalizes `x` and applies a conditioning-dependent
    affine modulation (gamma scale, beta shift).
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Zero-init so the block starts as identity: gamma=0 (scale 1), beta=0.
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        cond: torch.Tensor,                                         # (B, cond_dim)
    ) -> torch.Tensor:
        gamma, beta = self.modulation(cond).chunk(2, dim=-1)          # each (B, dim)

        x = self.norm(x)                                               # (B, T, dim)
        x = x * (1 + gamma[:, None, :]) + beta[:, None, :]              # (B, T, dim)

        return x                                                       # (B, T, dim)


class ConditionalLayerNorm(nn.Module):
    """
    Layer norm that optionally applies AdaLN modulation conditioned on `cond`.
    When `use_ada_ln` is False, falls back to a plain LayerNorm and `cond` is ignored.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: Optional[int] = None,
        use_ada_ln: bool = False,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        if use_ada_ln:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when use_ada_ln=True")
            self.norm = AdaLN(dim, cond_dim)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        cond: Optional[torch.Tensor] = None,                         # (B, cond_dim) or None
    ) -> torch.Tensor:
        if self.use_ada_ln:
            return self.norm(x, cond)                                # (B, T, dim)
        return self.norm(x)                                          # (B, T, dim)
