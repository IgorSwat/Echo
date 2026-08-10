from echo.modules.attention import SelfAttention, CrossAttention
from echo.modules.ffn import FeedForward
from echo.modules.norm import ConditionalLayerNorm
from echo.modules.types import HybridKVCache, HybridLayerCache, KVCache, LayerCache

from typing import Optional

import torch
import torch.nn as nn


# ------------------
# Transformer blocks
# ------------------


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
        mode: str = "bidirectional"
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
        kv_cache: Optional[LayerCache] = None,                        # keys/values so far
        start_pos: int = 0,                                         # positions already cached
    ) -> tuple[torch.Tensor, LayerCache]:
        h, g1 = self.norm1(x, cond)                                 # (B, T, D), (B, D)
        attn, cache = self.attn(h, key_padding_mask, kv_cache, start_pos)
        x = x + g1[:, None, :] * attn                                # (B, T, D)
        h, g2 = self.norm2(x, cond)                                 # (B, T, D), (B, D)
        x = x + g2[:, None, :] * self.ffn(h)                        # (B, T, D)

        return x, cache                                              # (B, T, D), cache


def _open_cross_gate(gate: torch.Tensor, use_ada_ln: bool) -> torch.Tensor:
    return gate + 1.0 if use_ada_ln else gate


class CrossAttentionBlock(nn.Module):
    """
    Pre-norm transformer block with cross-attention over an external context.

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
        rope_norm: str = "query",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln
        self.needs_proj = d_query != d_model

        # Separate norms for the query and context streams (each at its own dim).
        self.norm_q = ConditionalLayerNorm(d_query, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_kv, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = CrossAttention(
            d_query, d_kv, d_model, num_heads, dropout, use_rope=use_rope,
            rope_norm=rope_norm,
        )
        self.norm2 = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)

        if self.needs_proj:
            self.resid_proj = nn.Linear(d_query, d_model)
        else:
            self.resid_proj = None

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_query)
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
        kv_cache: Optional[LayerCache] = None,                        # context keys/values
        start_pos: int = 0,                                         # query positions consumed
    ) -> tuple[torch.Tensor, LayerCache]:
        q, g1 = self.norm_q(x, cond)                                 # (B, T, d_query), (B, d_query)
        # Normalizing the context is part of building its keys/values, so it is
        # skipped along with them once the cache exists.
        ctx = None if kv_cache is not None else self.norm_ctx(context, cond)[0]
        delta, cache = self.attn(
            q, ctx, key_padding_mask, query_padding_mask, kv_cache, start_pos
        )
        if not self.needs_proj:
            delta = _open_cross_gate(g1, self.use_ada_ln)[:, None, :] * delta

        residual = self.resid_proj(x) if self.needs_proj else x        # (B, T, d_model)
        x = residual + delta                                           # (B, T, d_model)

        h, g2 = self.norm2(x, cond)                                  # (B, T, d_model), (B, d_model)
        x = x + g2[:, None, :] * self.ffn(h)                         # (B, T, d_model)

        return x, cache                                              # (B, T, d_model), cache


class HybridAttentionBlock(nn.Module):
    """
    Pre-norm transformer block combining self-attention and cross-attention.

    Equivalent to a SelfAttentionBlock followed by a CrossAttentionBlock that
    share a single FFN: norm -> self-attn -> add -> norm -> cross-attn -> add
    -> norm -> ffn -> add.
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
        self.self_attn = SelfAttention(d_model, num_heads, dropout, use_rope=use_rope, mode=mode)

        # Separate norms for the query and context streams (each at its own dim).
        self.norm_q = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.norm_ctx = ConditionalLayerNorm(d_kv, cond_dim, use_ada_ln=use_ada_ln)
        self.cross_attn = CrossAttention(
            d_model, d_kv, d_model, num_heads, dropout, use_rope=use_rope,
            rope_norm=rope_norm,
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

        # Norm & self-attention
        h, g1 = self.norm1(x, cond)                                  # (B, T, d_model), (B, d_model)
        attn, self_cache = self.self_attn(h, padding_mask, self_cache, start_pos)
        x = x + g1[:, None, :] * attn                                # (B, T, d_model)

        # Norm & cross-attention
        q, g2 = self.norm_q(x, cond)                                 # (B, T, d_model), (B, d_model)
        ctx = None if cross_cache is not None else self.norm_ctx(context, cond)[0]
        attn, cross_cache = self.cross_attn(
            q, ctx, context_padding_mask, padding_mask, cross_cache, start_pos
        )
        # Identity-initialized rather than zero-initialized: this is the only
        # path the context takes in. See :func:`_open_cross_gate`.
        x = x + _open_cross_gate(g2, self.use_ada_ln)[:, None, :] * attn

        # Norm & FFN
        h, g3 = self.norm2(x, cond)                                  # (B, T, d_model), (B, d_model)
        x = x + g3[:, None, :] * self.ffn(h)                         # (B, T, d_model)

        return x, (self_cache, cross_cache)


# -------------------
# Transformer classes
# -------------------

class _SelfAttentionStack(nn.Module):
    """
    Stack of pre-norm SelfAttentionBlocks followed by one final norm.

    Subclasses fix the attention `mode`; everything else is shared.
    """

    mode: str

    def __init__(
        self,
        d_model: int,
        num_layers: int,
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

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                d_model, num_heads, ffn_dim,
                dropout, use_rope,
                use_ada_ln, cond_dim, ffn_glu,
                mode=self.mode
            )
            for _ in range(num_layers)
        ])

        # One final norm
        self.norm = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        kv_cache: Optional[KVCache] = None,                         # one entry per block
        start_pos: int = 0,                                         # positions already cached
    ) -> tuple[torch.Tensor, KVCache]:
        caches: KVCache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(
                x, key_padding_mask, cond,
                None if kv_cache is None else kv_cache[i], start_pos,
            )                                                        # (B, T, D)
            caches.append(cache)
        x, _ = self.norm(x, cond)                                    # (B, T, D)

        return x, caches                                             # (B, T, D), per-block caches


class SelfAttentionEncoder(_SelfAttentionStack):
    """
    Stack of pre-norm transformer blocks with bidirectional self-attention:
    every position sees the whole sequence.
    """

    mode = "bidirectional"


class SelfAttentionDecoder(_SelfAttentionStack):
    """
    Stack of pre-norm transformer blocks with causal self-attention:
    every position sees only itself and what came before.
    """

    mode = "causal"


class HybridAttentionDecoder(nn.Module):
    """
    Stack of causal HybridAttentionBlocks followed by one final norm.

    Each layer attends over the decoded stream so far (causal self-attention)
    and over an external context (cross-attention).
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_layers: int,
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

        self.blocks = nn.ModuleList([
            HybridAttentionBlock(
                d_model, d_kv, num_heads, ffn_dim,
                dropout, use_rope,
                use_ada_ln, cond_dim, ffn_glu,
                mode="causal",
                rope_norm=rope_norm,
            )
            for _ in range(num_layers)
        ])

        # One final norm
        self.norm = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_model)
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        padding_mask: Optional[torch.Tensor] = None,                # (B, S_self) or None
        context_padding_mask: Optional[torch.Tensor] = None,        # (B, S) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
        kv_cache: Optional[HybridKVCache] = None,                   # one entry per block
        start_pos: int = 0,                                         # positions already decoded
    ) -> tuple[torch.Tensor, HybridKVCache]:
        caches: HybridKVCache = []
        for i, block in enumerate(self.blocks):
            x, cache = block(
                x, context, padding_mask, context_padding_mask, cond,
                None if kv_cache is None else kv_cache[i], start_pos,
            )                                                        # (B, T, d_model)
            caches.append(cache)
        x, _ = self.norm(x, cond)                                    # (B, T, d_model)

        return x, caches                                             # (B, T, d_model), caches
