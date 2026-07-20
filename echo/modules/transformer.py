from echo import config

from echo.modules.attention import CausalSelfAttention
from echo.modules.ffn import FeedForward
from echo.modules.types import KVCache

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------
# Transformer - Decoder
# ---------------------

class DecoderBlock(nn.Module):
    """
    Pre-norm transformer decoder block (causal self-attention + FFN).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.FFN_GLU)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # Casual self-attention.
        # Pre-norm style - that is, instead of applying the norm after final FNN,
        # we apply it at the beginning of the next block.
        attn_out, new_kv = self.attn(self.norm1(x), kv_cache)
        
		# Add 
        x = x + attn_out
        
		# Nnorm -> FFN -> Add
        x = x + self.ffn(self.norm2(x))
        
        return x, new_kv


class TransformerDecoder(nn.Module):
    """
    Stack of pre-norm decoder blocks with KV-cache support.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        
        self.norm = nn.LayerNorm(d_model)  # final norm before the output projection

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        # This might seem confusing, but basically when we don't use KV cache (that is, kv_cache=None),
        # if becomes a list of Nones instead, and stays like that till the end.
        # Autograd does not touch this.
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

		# A sequential execution block by block.
        new_cache: KVCache = []
        for block, cached_kv in zip(self.blocks, kv_cache):
            x, new_kv = block(x, cached_kv)
            new_cache.append(new_kv)

        x = self.norm(x)
        
        return x, new_cache