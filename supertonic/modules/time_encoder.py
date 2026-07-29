from supertonic import config

from supertonic.modules import apply_mask

from typing import Optional

import math

import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """t -> sinusoidal(t * t_scale) -> Linear -> Mish -> Linear."""

    def __init__(self, time_dim: int = 64, hdim: int = 256, t_scale: float = 1000.0) -> None:
        super().__init__()

        self.t_scale = t_scale
        half = time_dim // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half).float() / half)
        self.register_buffer("freqs", freqs)                        # (half,)

        self.mlp = nn.Sequential(
            nn.Linear(time_dim, hdim),
            nn.Mish(),
            nn.Linear(hdim, time_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:              # (B,) float
        ang = t.view(-1, 1) * self.t_scale * self.freqs              # (B, half)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)    # (B, time_dim)

        return self.mlp(emb)                                          # (B, time_dim)


class TimeConditionBlock(nn.Module):
    """Additive time conditioning: project the time embedding to `dim` and add."""

    def __init__(self, dim: int, time_dim: int) -> None:
        super().__init__()

        self.linear = nn.Linear(time_dim, dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.linear.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.linear.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        cond: torch.Tensor,                                         # (B, time_dim)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        x = x + self.linear(cond).unsqueeze(1)                      # (B, T, dim)

        return apply_mask(x, mask)                                  # (B, T, dim)
