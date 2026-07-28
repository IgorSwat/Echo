from echo import config

from echo.modules.norm import ConditionalLayerNorm

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    """
    Depthwise-separable convolution module with a GLU pointwise expansion,
    as used in the Conformer architecture.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        use_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)

        self.pw1 = nn.Linear(d_model, 2 * d_model)                       # pointwise expand
        self.dw = nn.Conv1d(                                             # depthwise
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.pw2 = nn.Linear(d_model, d_model)                           # pointwise project

        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, T, D)
        # Optional norm
        if self.use_norm:
            x = self.norm(x)                                             # (B, T, D)

		# As proposed in "Language Modeling with Gated Convolutional Networks",
        # we use GLU as a form of selective channel mixing.
        x = F.glu(self.pw1(x), dim=-1)                                   # (B, T, D)
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)                   # (B, T, D)
        x = F.gelu(x)                                                    # (B, T, D)

        return self.drop(self.pw2(x))                                    # (B, T, D)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt block: depthwise kxk conv -> (optional pointwise channel proj) ->
    LayerNorm -> 1x1 expand (4C) -> GELU -> 1x1 project (C) -> layer scale ->
    Dropout, with a residual path.
    Uses ConditionalLayerNorm (AdaLN when `use_ada_ln` is set).
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
    ) -> None:
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.needs_proj = dim_in != dim_out

        # Depthwise conv (spatial mixing, stays at dim_in).
        self.dw = nn.Conv1d(
            dim_in, dim_in, kernel_size,
            padding=kernel_size // 2, groups=dim_in,
        )
        
        # Pointwise channel projection when dims differ.
        if self.needs_proj:
            self.pw_proj = nn.Linear(dim_in, dim_out)
        else:
            self.pw_proj = None

        self.norm = ConditionalLayerNorm(dim_out, cond_dim, use_ada_ln=use_ada_ln)

        # An equivalent of transformer's FFN.
        self.pw1 = nn.Linear(dim_out, 4 * dim_out)                       # pointwise expand
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4 * dim_out, dim_out)                       # pointwise project

        # Layer scale (per-channel). Zero-init-variant: small constant init.
        self.gamma = nn.Parameter(torch.full((dim_out,), layer_scale_init))

        self.drop = nn.Dropout(dropout)

        # Residual projection when channel counts differ.
        if self.needs_proj:
            self.resid_proj = nn.Linear(dim_in, dim_out)
        else:
            self.resid_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                                              # (B, T, dim_in)
        cond: Optional[torch.Tensor] = None,                          # (B, cond_dim) or None
    ) -> torch.Tensor:
        y = self.dw(x.transpose(1, 2)).transpose(1, 2)                # (B, T, dim_in)
        if self.pw_proj is not None:
            y = self.pw_proj(y)                                        # (B, T, dim_out)
        y = self.norm(y, cond)                                        # (B, T, dim_out)
        y = self.pw2(self.act(self.pw1(y)))                           # (B, T, dim_out)
        y = y * self.gamma                                            # (B, T, dim_out) layer scale
        y = self.drop(y)                                              # (B, T, dim_out)

        residual = self.resid_proj(x) if self.needs_proj else x       # (B, T, dim_out)

        return y + residual                                           # (B, T, dim_out)


class Downsample1D(nn.Module):
    """
    Stride-2 temporal downsample with channel doubling: a 1D conv that halves
    the sequence length and doubles the channel dimension.
    """

    def __init__(
        self,
        dim_in: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            dim_in, 2 * dim_in, kernel_size,
            stride=2, padding=kernel_size // 2,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.conv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, T, dim_in)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)                # (B, T//2, 2*dim_in)

        return x


class Upsample1D(nn.Module):
    """
    Temporal upsample by 2x with channel halving: interpolation
    followed by a 1D conv that projects (2*dim) -> dim.
    """

    def __init__(
        self,
        dim_in: int,
        mode: str = "nearest",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.mode = mode

        self.conv = nn.Conv1d(
            dim_in, dim_in // 2, kernel_size,
            padding=kernel_size // 2,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.conv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, T//2, 2*dim)
        B, T_half, D = x.shape

        x = F.interpolate(                                              # (B, D, T_half) -> (B, D, T)
            x.transpose(1, 2),
            size=T_half * 2, 
            mode=self.mode,
        )

        x = self.conv(x).transpose(1, 2)                                # (B, T, D//2)

        return x
