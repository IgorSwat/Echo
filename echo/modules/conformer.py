from echo import config

from echo.modules.attention import BidirectionalSelfAttention
from echo.modules.conv import GatedConv
from echo.modules.ffn import FeedForward

from typing import Optional

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """
    Conformer block: macaron-style FFN, bidirectional self-attention, and a
    gated convolution module, each with its own residual connection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        conv_use_norm: bool = True,
    ) -> None:
        super().__init__()
        self.ffn1 = FeedForward(d_model, ffn_dim, dropout, use_glu=config.decoder_ffn_glu)
        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.conv = GatedConv(d_model, kernel_size, use_norm=conv_use_norm, dropout=dropout)
        self.ffn2 = FeedForward(d_model, ffn_dim, dropout, use_glu=config.decoder_ffn_glu)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        # Macaron half-step FFNs
        x = x + 0.5 * self.ffn1(x)                                   # (B, T, D)
        x = x + self.attn(self.norm_attn(x), key_padding_mask)       # (B, T, D)
        x = x + self.conv(x)                                         # (B, T, D)
        x = x + 0.5 * self.ffn2(x)                                   # (B, T, D)
        x = self.norm_out(x)                                         # (B, T, D)

        return x                                                     # (B, T, D)


class Conformer(nn.Module):
    """
    Stack of pre-norm Conformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        conv_use_norm: bool = True,
    ) -> None:
        super().__init__()

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model, num_heads, ffn_dim, kernel_size,
                dropout, use_rope, conv_use_norm,
            )
            for _ in range(num_layers)
        ])

        # One final norm
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask)                           # (B, T, D)
        x = self.norm(x)                                             # (B, T, D)

        return x                                                     # (B, T, D)
