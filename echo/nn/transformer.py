from echo.nn.norm import ConditionalLayerNorm
from echo.nn.transformer_blocks import HybridAttentionBlock, SelfAttentionBlock
from echo.nn.types import HybridKVCache, KVCache

from typing import Optional

import torch
import torch.nn as nn


class SelfAttentionEncoder(nn.Module):
    """
    Bidirectional Transformer Encoder.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        use_rope: bool = False,
        use_glu: bool = False,
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,     # Only if AdaLN conditioning is active
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                d_model, num_heads, ffn_dim,
                dropout, use_rope,
                use_ada_ln, cond_dim, use_glu,
                mode="bidirectional",
            )
            for _ in range(num_layers)
        ])

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
            )                                                       # (B, T, D)
            caches.append(cache)

        x, _ = self.norm(x, cond)                                   # (B, T, D)

        return x, caches                                            # (B, T, D), per-block caches


class SelfAttentionDecoder(nn.Module):
    """
    Causal Transformer Decoder (for decoder-only architectures).
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        use_rope: bool = False,
        use_glu: bool = False,
        use_ada_ln: bool = False,       # Only if AdaLN conditioning is active
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                d_model, num_heads, ffn_dim,
                dropout, use_rope,
                use_ada_ln, cond_dim, use_glu,
                mode="causal",
            )
            for _ in range(num_layers)
        ])

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
            )                                                       # (B, T, D)
            caches.append(cache)

        x, _ = self.norm(x, cond)                                   # (B, T, D)

        return x, caches                                            # (B, T, D), per-block caches


class HybridAttentionDecoder(nn.Module):
    """
    Causal Transformer Decoder + cross attention (for encoder-decoder architectures)
    """

    def __init__(
        self,
        d_model: int,
        d_kv: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        use_rope: bool = False,
        rope_norm: str = "query",
        use_glu: bool = False,
        use_ada_ln: bool = False,       # Only if AdaLN conditioning is active
        cond_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.blocks = nn.ModuleList([
            HybridAttentionBlock(
                d_model, d_kv, num_heads, ffn_dim,
                dropout, use_rope,
                use_ada_ln, cond_dim, use_glu,
                mode="causal",
                rope_norm=rope_norm,
            )
            for _ in range(num_layers)
        ])

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
            )                                                       # (B, T, d_model)
            caches.append(cache)
            
        x, _ = self.norm(x, cond)                                   # (B, T, d_model)

        return x, caches                                            # (B, T, d_model), caches
