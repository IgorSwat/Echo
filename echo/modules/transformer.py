from echo import config

from echo.modules.attention import BidirectionalSelfAttention, CrossAttention
from echo.modules.ffn import FeedForward
from echo.modules.norm import ConditionalLayerNorm

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
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.norm1 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = BidirectionalSelfAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.decoder_ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x, cond), key_padding_mask)    # (B, T, D)
        x = x + self.ffn(self.norm2(x, cond))                       # (B, T, D)

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
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        # Two separate layer norms for queries and keys/values (context)
        self.norm_q = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = CrossAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.decoder_ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        context: torch.Tensor,                                      # (B, S, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm_q(x, cond), self.norm_ctx(context, cond), key_padding_mask)  # (B, T, D)
        x = x + self.ffn(self.norm2(x, cond))                       # (B, T, D)

        return x                                                     # (B, T, D)
