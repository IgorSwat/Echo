from echo.nn.init import init_weights_
from echo.nn.norm import ConditionalLayerNorm

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    """
    Depthwise-separable convolution with a GLU pointwise expansion, as used in the Conformer architecture. 

    NOTE: In causal mode the depthwise conv only looks backwards: output i is built from inputs i-k+1 ... i.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        use_norm: bool = True,
        dropout: float = 0.0,
        mode: str = "bidirectional",
    ) -> None:
        super().__init__()

        self.use_norm = use_norm
        self.mode = mode
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

        self.pw1 = nn.Linear(d_model, 2 * d_model)                  # pointwise expand

        # Padded by hand rather than by the Conv1d so both modes (bidirectional and causal) share one module.
        self.pad = (
            (kernel_size - 1, 0) if mode == "causal"
            else (kernel_size // 2, kernel_size // 2)
        )
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model)

        self.pw2 = nn.Linear(d_model, d_model)                      # pointwise project

        self.drop = nn.Dropout(dropout)

        init_weights_(self)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        if self.use_norm:
            x = self.norm(x)                                        # (B, T, D)

        # GLU as a form of selective channel mixing, after "Language Modeling
        # with Gated Convolutional Networks".
        x = F.glu(self.pw1(x), dim=-1)                              # (B, T, D)

        if key_padding_mask is not None:
            x = x * key_padding_mask.unsqueeze(-1).to(x.dtype)      # (B, T, D)

        x = F.pad(x.transpose(1, 2), self.pad)                      # (B, D, T + k - 1)
        x = self.dw(x).transpose(1, 2)                              # (B, T, D)
        x = F.gelu(x)                                               # (B, T, D)

        return self.drop(self.pw2(x))                               # (B, T, D)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block with a residual path.

    NOTE: In causal mode the depthwise conv only looks backwards: output i is built from inputs i-k+1 ... i.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel_size: int = 7,
        use_ada_ln: bool = False,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-6,
        cond_dim: Optional[int] = None,
        mode: str = "bidirectional",
    ) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.needs_proj = dim_in != dim_out
        self.mode = mode

        # Padded by hand rather than by the Conv1d so both modes (bidirectional and causal) share one module.
        self.pad = (
            (kernel_size - 1, 0) if mode == "causal"
            else (kernel_size // 2, kernel_size // 2)
        )
        self.dw = nn.Conv1d(dim_in, dim_in, kernel_size, groups=dim_in)

        self.pw_proj = nn.Linear(dim_in, dim_out) if self.needs_proj else None

        self.norm = ConditionalLayerNorm(dim_out, cond_dim, use_ada_ln=use_ada_ln)

        # The transformer FFN's counterpart.
        self.pw1 = nn.Linear(dim_out, 4 * dim_out)                  # pointwise expand
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim_out, dim_out)                  # pointwise project

        # Per-channel layer scale, small-constant initialized.
        self.gamma = nn.Parameter(torch.full((dim_out,), layer_scale_init))

        self.drop = nn.Dropout(dropout)
        self.resid_proj = nn.Linear(dim_in, dim_out) if self.needs_proj else None

        init_weights_(self)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim_in)
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        # Zero the padded frames so they contribute nothing to the depthwise conv.
        y = x if key_padding_mask is None else x * key_padding_mask.unsqueeze(-1).to(x.dtype)

        y = F.pad(y.transpose(1, 2), self.pad)                      # (B, dim_in, T + k - 1)
        y = self.dw(y).transpose(1, 2)                              # (B, T, dim_in)
        if self.pw_proj is not None:
            y = self.pw_proj(y)                                     # (B, T, dim_out)
        y, gate = self.norm(y, cond)                                # (B, T, dim_out), (B, dim_out)
        y = self.pw2(self.act(self.pw1(y)))                         # (B, T, dim_out)
        y = y * self.gamma                                          # layer scale
        y = self.drop(y)
        y = gate[:, None, :] * y                                    # AdaLN-Zero gate

        residual = self.resid_proj(x) if self.needs_proj else x     # (B, T, dim_out)

        return y + residual                                         # (B, T, dim_out)
