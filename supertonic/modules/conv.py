from supertonic import config

from supertonic.modules import apply_mask

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    """
    Masked ConvNeXt-1D block: depthwise kxk conv (dilated, replicate padding) ->
    LayerNorm -> pointwise expand -> GELU -> pointwise project -> layer scale,
    with a residual path. Padded positions are zeroed after the depthwise conv
    and again at the output (Supertonic masking order).
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int,
        dilation: int = 1,
    ) -> None:
        super().__init__()

        self.pad = (kernel_size - 1) // 2 * dilation
        self.dw = nn.Conv1d(dim, dim, kernel_size, dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)
        self.pw1 = nn.Linear(dim, intermediate_dim)
        self.pw2 = nn.Linear(intermediate_dim, dim)

        # Layer scale (per-channel). Supertonic keeps gamma at 1.
        self.gamma = nn.Parameter(torch.ones(dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        residual = x

        h = F.pad(x.transpose(1, 2), (self.pad, self.pad), mode="replicate")
        h = self.dw(h).transpose(1, 2)                              # (B, T, D)
        h = apply_mask(h, mask)
        h = self.norm(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        h = residual + self.gamma * h

        return apply_mask(h, mask)                                  # (B, T, D)


class ConvNeXtStack(nn.Module):
    """A stack of ConvNeXt blocks with per-block dilations, all at one dim."""

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        kernel_size: int,
        dilations: Sequence[int] = (1,),
    ) -> None:
        super().__init__()

        self.convnext = nn.ModuleList(
            ConvNeXtBlock(dim, intermediate_dim, kernel_size, d) for d in dilations
        )

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        for block in self.convnext:
            x = apply_mask(x, mask)                                 # mask before each block
            x = block(x, mask)
        return x                                                    # (B, T, D)
