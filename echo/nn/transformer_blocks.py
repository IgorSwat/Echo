from echo.nn.attention import CrossAttention, SelfAttention
from echo.nn.ffn import FeedForward
from echo.nn.norm import ConditionalLayerNorm
from echo.nn.types import HybridLayerCache, LayerCache

from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionBlock(nn.Module):
    """
    Pre-norm transformer block with self-attention.
    """

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
        mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.norm1 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = SelfAttention(d_model, num_heads, dropout, use_rope=use_rope, mode=mode)
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        kv_cache: Optional[LayerCache] = None,                      # keys/values so far
        start_pos: int = 0,                                         # positions already cached
    ) -> tuple[torch.Tensor, LayerCache]:
        h, g1 = self.norm1(x, cond)                                 # (B, T, D), (B, D)
        attn, cache = self.attn(h, key_padding_mask, kv_cache, start_pos)
        x = x + g1[:, None, :] * attn                               # (B, T, D)

        h, g2 = self.norm2(x, cond)                                 # (B, T, D), (B, D)
        x = x + g2[:, None, :] * self.ffn(h)                        # (B, T, D)

        return x, cache                                             # (B, T, D), cache


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm transformer block with cross-attention over an external context.

    NOTE: applies projections d_query -> d_model, d_kv -> d_model.
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
        rope_norm: str = "query",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.needs_proj = d_query != d_model

        # Separate norms for the query and context streams (each at its own dim).
        self.norm_q = ConditionalLayerNorm(d_query, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_kv, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = CrossAttention(
            d_query, d_kv, d_model, num_heads, dropout,
            use_rope=use_rope, rope_norm=rope_norm,
        )
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

        self.resid_proj = nn.Linear(d_query, d_model) if self.needs_proj else None

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_query)
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
        kv_cache: Optional[LayerCache] = None,                      # context keys/values
        start_pos: int = 0,                                         # query positions consumed
    ) -> tuple[torch.Tensor, LayerCache]:
        q, g1 = self.norm_q(x, cond)                                # (B, T, d_query), (B, d_query)
        # Normalizing the context is part of building its keys/values, so it is
        # skipped along with them once the cache exists.
        ctx = None if kv_cache is not None else self.norm_ctx(context, cond)[0]
        delta, cache = self.attn(
            q, ctx, key_padding_mask, query_padding_mask, kv_cache, start_pos
        )
        if not self.needs_proj:
            # AdaLN gate shifted to start at identity rather than closed.
            gate = g1 + 1.0 if self.use_ada_ln else g1
            delta = gate[:, None, :] * delta

        residual = self.resid_proj(x) if self.needs_proj else x     # (B, T, d_model)
        x = residual + delta                                        # (B, T, d_model)

        h, g2 = self.norm2(x, cond)                                 # (B, T, d_model), (B, d_model)
        x = x + g2[:, None, :] * self.ffn(h)                        # (B, T, d_model)

        return x, cache                                             # (B, T, d_model), cache


class HybridAttentionBlock(nn.Module):
    """
    Pre-norm transformer block of self-attention + cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
        ffn_glu: bool = False,
        mode: str = "bidirectional",
        rope_norm: str = "query",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.norm1 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.self_attn = SelfAttention(
            d_model, num_heads, dropout, use_rope=use_rope, mode=mode
        )

        # Separate norms for the query and context streams (each at its own dim).
        self.norm_q = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_kv, cond_dim, use_ada_ln=use_ada_ln)
        self.cross_attn = CrossAttention(
            d_model, d_kv, d_model, num_heads, dropout,
            use_rope=use_rope, rope_norm=rope_norm,
        )

        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_model)
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        padding_mask: Optional[torch.Tensor] = None,                # (B, S_self) or None
        context_padding_mask: Optional[torch.Tensor] = None,        # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        kv_cache: Optional[HybridLayerCache] = None,                # (self cache, cross cache)
        start_pos: int = 0,                                         # positions already decoded
    ) -> tuple[torch.Tensor, HybridLayerCache]:
        self_cache, cross_cache = kv_cache if kv_cache is not None else (None, None)

        h, g1 = self.norm1(x, cond)                                 # (B, T, d_model), (B, d_model)
        attn, self_cache = self.self_attn(h, padding_mask, self_cache, start_pos)
        x = x + g1[:, None, :] * attn                               # (B, T, d_model)

        q, g2 = self.norm_q(x, cond)                                # (B, T, d_model), (B, d_model)
        ctx = None if cross_cache is not None else self.norm_ctx(context, cond)[0]
        attn, cross_cache = self.cross_attn(
            q, ctx, context_padding_mask, padding_mask, cross_cache, start_pos
        )
        # AdaLN gate shifted to start at identity rather than closed.
        cross_gate = g2 + 1.0 if self.use_ada_ln else g2
        x = x + cross_gate[:, None, :] * attn

        h, g3 = self.norm2(x, cond)                                 # (B, T, d_model), (B, d_model)
        x = x + g3[:, None, :] * self.ffn(h)                        # (B, T, d_model)

        return x, (self_cache, cross_cache)
