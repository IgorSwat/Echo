from echo.modules.attention import SelfAttention, CrossAttention
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
        ffn_glu: bool = False,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.norm1 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = SelfAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
    ) -> torch.Tensor:
        h, g1 = self.norm1(x, cond)                                 # (B, T, D), (B, D)
        x = x + g1[:, None, :] * self.attn(h, key_padding_mask)    # (B, T, D)
        h, g2 = self.norm2(x, cond)                                 # (B, T, D), (B, D)
        x = x + g2[:, None, :] * self.ffn(h)                        # (B, T, D)

        return x                                                     # (B, T, D)


class CrossAttentionBlock(nn.Module):
    """Pre-norm transformer block with cross-attention over an external context.

    Query and context streams may have different hidden dims (d_query / d_kv);
    both are projected to d_model inside the CrossAttention. The residual path
    for the query stream is projected to d_model when d_query != d_model.
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
        ffn_glu: bool = False,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.needs_proj = d_query != d_model

        # Separate norms for the query and context streams (each at its own dim).
        self.norm_q = ConditionalLayerNorm(d_query, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_kv, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = CrossAttention(d_query, d_kv, d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

        if self.needs_proj:
            self.resid_proj = nn.Linear(d_query, d_model)
        else:
            self.resid_proj = None

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_query)
        context: torch.Tensor,                                      # (B, S, d_kv)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
    ) -> torch.Tensor:
        residual = self.resid_proj(x) if self.needs_proj else x        # (B, T, d_model)
        q, g1 = self.norm_q(x, cond)                                 # (B, T, d_query), (B, d_query)
        ctx, _ = self.norm_ctx(context, cond)                        # (B, S, d_kv)
        delta = self.attn(q, ctx, key_padding_mask, query_padding_mask)
        if not self.needs_proj:
            delta = g1[:, None, :] * delta
        x = residual + delta                                         # (B, T, d_model)
        h, g2 = self.norm2(x, cond)                                  # (B, T, d_model), (B, d_model)
        x = x + g2[:, None, :] * self.ffn(h)                         # (B, T, d_model)

        return x                                                     # (B, T, d_model)
