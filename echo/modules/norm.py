from typing import Optional

import torch
import torch.nn as nn


class AdaLN(nn.Module):
    """
    Adaptive Layer Normalization with a residual gate (AdaLN-Zero style).

    Normalizes `x`, applies conditioning-dependent affine modulation (gamma, beta),
    and predicts a per-channel residual gate. Modulation is zero-initialized so
    gamma=0 (scale 1), beta=0, gate=0 at start — residual branches that multiply
    by the gate begin as identity.
    """

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        cond: torch.Tensor,                                         # (B, cond_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta, gate = self.modulation(cond).chunk(3, dim=-1)  # each (B, dim)

        x = self.norm(x)                                               # (B, T, dim)
        x = x * (1 + gamma[:, None, :]) + beta[:, None, :]              # (B, T, dim)

        return x, gate                                                 # (B, T, dim), (B, dim)


class ConditionalLayerNorm(nn.Module):
    """
    Layer norm that optionally applies AdaLN modulation conditioned on `cond`.
    When `use_ada_ln` is False, falls back to a plain LayerNorm and `cond` is ignored.
    Always returns (x, gate) so callers can apply AdaLN-Zero residual gating;
    gate is ones when AdaLN is disabled.
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_ada_ln:
            return self.norm(x, cond)                                # (B, T, dim), (B, dim)

        gate = torch.ones(x.shape[0], x.shape[-1], device=x.device, dtype=x.dtype)
        
        return self.norm(x), gate                                    # (B, T, dim), (B, dim)
