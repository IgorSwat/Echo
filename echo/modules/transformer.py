from echo import config

from echo.modules.attention import BidirectionalSelfAttention, CrossAttention
from echo.modules.ffn import FeedForward

from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """Pre-norm transformer block with bidirectional self-attention."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = BidirectionalSelfAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.decoder_ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)          # (B, T, D)
        x = x + self.ffn(self.norm2(x))                              # (B, T, D)

        return x                                                     # (B, T, D)


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer block with cross-attention over an external context."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        super().__init__()

        # Two seperate layer norms for queries and keys/values (context)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_ctx = nn.LayerNorm(d_model)
        self.attn = CrossAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.decoder_ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        context: torch.Tensor,                                      # (B, S, D)
        key_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_q(x), self.norm_ctx(context), key_padding_mask)  # (B, T, D)
        x = x + self.ffn(self.norm2(x))                              # (B, T, D)

        return x                                                     # (B, T, D)
