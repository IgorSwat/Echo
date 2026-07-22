from echo import config

from echo.modules.attention import CausalSelfAttention
from echo.modules.ffn import FeedForward
from echo.modules.types import KVCache

from typing import Optional

import torch
import torch.nn as nn


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block."""

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        ffn_dim: int, 
        dropout: float = 0.0, 
        use_rope: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout, use_rope=use_rope)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.decoder_ffn_glu)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.norm1(x), kv_cache, key_padding_mask, start_pos)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))

        return x, new_kv


class TransformerDecoder(nn.Module):
    """Stack of pre-norm decoder blocks with KV-cache support."""

    def __init__(
        self, 
        d_model: int, 
        num_layers: int, 
        num_heads: int, 
        ffn_dim: int, 
        dropout: float = 0.0, 
        use_rope: bool = False
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, ffn_dim, dropout, use_rope) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        new_cache: KVCache = []
        for block, cached_kv in zip(self.blocks, kv_cache):
            x, new_kv = block(x, cached_kv, key_padding_mask, start_pos)
            new_cache.append(new_kv)

        x = self.norm(x)
        
        return x, new_cache