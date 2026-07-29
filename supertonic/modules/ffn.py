from supertonic import config

from supertonic.modules import apply_mask

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """VITS-style position-wise FFN: Linear -> ReLU -> Linear, masked."""

    def __init__(self, d_model: int, ffn_dim: int) -> None:
        super().__init__()

        self.up = nn.Linear(d_model, ffn_dim)
        self.down = nn.Linear(ffn_dim, d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        x = self.up(apply_mask(x, mask))
        x = F.relu(x)
        x = self.down(apply_mask(x, mask))

        return apply_mask(x, mask)                                  # (B, T, D)
