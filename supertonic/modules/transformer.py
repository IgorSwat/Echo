from supertonic import config

from supertonic.modules import apply_mask
from supertonic.modules.attention import RelPosMultiHeadAttention, RotaryCrossAttention, TanhAttention
from supertonic.modules.ffn import FeedForward

from typing import Optional

import torch
import torch.nn as nn


class AttnEncoder(nn.Module):
    """VITS-layout transformer encoder (post-norm) used for text encoding."""

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
        window_size: int = 4,
    ) -> None:
        super().__init__()

        self.attn_layers = nn.ModuleList(
            RelPosMultiHeadAttention(d_model, num_heads, window_size)
            for _ in range(num_layers)
        )
        self.norm_layers_1 = nn.ModuleList(
            nn.LayerNorm(d_model, eps=config.layer_norm_eps) for _ in range(num_layers)
        )
        self.ffn_layers = nn.ModuleList(
            FeedForward(d_model, ffn_dim) for _ in range(num_layers)
        )
        self.norm_layers_2 = nn.ModuleList(
            nn.LayerNorm(d_model, eps=config.layer_norm_eps) for _ in range(num_layers)
        )

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        x = apply_mask(x, mask)
        for attn, norm1, ffn, norm2 in zip(
            self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2
        ):
            x = norm1(x + attn(x, mask))
            x = norm2(x + ffn(x, mask))

        return apply_mask(x, mask)                                  # (B, T, D)


class TextConditionBlock(nn.Module):
    """Rotary cross-attention over text embeddings with a residual + LayerNorm."""

    def __init__(
        self,
        dim: int,
        text_dim: int,
        num_heads: int,
        rotary_dim: int = 32,
        scale: float = 16.0,
    ) -> None:
        super().__init__()

        self.attn = RotaryCrossAttention(dim, text_dim, num_heads, rotary_dim, scale)
        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        context: torch.Tensor,                                      # (B, S, text_dim)
        context_mask: Optional[torch.Tensor] = None,                # (B, S) bool or None
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        y = self.attn(apply_mask(x, mask), context, context_mask, mask)
        y = apply_mask(y, mask)
        out = self.norm(apply_mask(x, mask) + y)

        return apply_mask(out, mask)                                # (B, T, dim)


class StyleConditionBlock(nn.Module):
    """Tanh-key attention over style tokens with a residual + LayerNorm."""

    def __init__(
        self,
        dim: int,
        style_dim: int,
        n_units: int = 256,
        num_heads: int = 2,
        scale: float = 16.0,
    ) -> None:
        super().__init__()

        self.attn = TanhAttention(dim, style_dim, n_units, dim, num_heads, scale)
        self.norm = nn.LayerNorm(dim, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        style_keys: torch.Tensor,                                   # (B, n_style, style_dim)
        style_values: torch.Tensor,                                 # (B, n_style, style_dim)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        y = self.attn(apply_mask(x, mask), style_keys, style_values, mask)
        y = apply_mask(y, mask)
        out = self.norm(apply_mask(x, mask) + y)

        return apply_mask(out, mask)                                # (B, T, dim)
