from echo.nn.attention_helpers import (
    absolute_freqs,
    causal_attn_mask,
    geometric_freqs,
    normalized_freqs,
    pad_attn_mask,
    rotate_halves,
    rotate_pairs,
)
from echo.nn.init import init_weights_
from echo.nn.types import LayerCache

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with fixed-frequency RoPE, bidirectional or causal.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        mode: str = "bidirectional",
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if mode not in ("bidirectional", "causal"):
            raise ValueError(f"unknown mode: {mode!r}")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.mode = mode
        self.max_seq_len = max_seq_len

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop_value = dropout
        self.resid_drop = nn.Dropout(dropout)

        # The frequencies are fixed, so with a known length the table is built once here instead of on every forward pass.
        if use_rope and max_seq_len is not None:
            cos, sin = geometric_freqs(
                self.hd, self.rope_theta, max_seq_len, device=torch.device("cpu")
            )
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        init_weights_(self)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        kv_cache: Optional[LayerCache] = None,                      # keys/values so far
        start_pos: int = 0,                                         # positions already cached
    ) -> tuple[torch.Tensor, LayerCache]:
        """
        Returns the output for `x` plus the keys/values to feed the next call.
        """
        
        B, T, D = x.shape

        qkv = self.qkv(x)                                           # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)                              # each (B, T, D)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)          # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)          # (B, nh, T, hd)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)          # (B, nh, T, hd)

        # The rotation depends on the ABSOLUTE position, so the new tokens are
        # rotated at start_pos ... start_pos + T - 1. Cached keys were rotated
        # when they were new and must not be touched again.
        if self.use_rope:
            end = start_pos + T
            if self.max_seq_len is not None:
                if end > self.max_seq_len:
                    raise ValueError(
                        f"sequence length ({end}) exceeds max_seq_len ({self.max_seq_len})"
                    )
                cos, sin = self.rope_cos, self.rope_sin
            else:
                cos, sin = geometric_freqs(self.hd, self.rope_theta, end + 1, x.device)
            q = rotate_pairs(q, cos, sin, start_pos)
            k = rotate_pairs(k, cos, sin, start_pos)

        # Prepend the cached keys/values; from here on the key axis has length S.
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            if cache_k.shape[2] != start_pos:
                raise ValueError(
                    f"kv_cache holds {cache_k.shape[2]} positions but start_pos is {start_pos}"
                )
            k = torch.cat([cache_k, k], dim=2)                      # (B, nh, S, hd)
            v = torch.cat([cache_v, v], dim=2)                      # (B, nh, S, hd)
        new_cache: LayerCache = (k, v)
        S = k.shape[2]

        attn_mask = None                                            # (B, 1, ., S) or None
        if key_padding_mask is not None:
            attn_mask = pad_attn_mask(key_padding_mask, B, S)

        # When the whole prefix is cached every key is already in the past, so
        # the causal mask degenerates to "attend to everything" and is skipped.
        if self.mode == "causal" and not (T == 1 and S == start_pos + 1):
            causal = causal_attn_mask(T, S, start_pos, x.device)    # (1, 1, T, S)
            attn_mask = causal if attn_mask is None else attn_mask & causal

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
        )                                                           # (B, nh, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)        # (B, T, D)
        out = self.resid_drop(self.proj(out))                       # (B, T, D)

        return out, new_cache                                       # (B, T, D), cache


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with learnable rotary embeddings on BOTH the query and the key stream.
    """

    ROPE_NORMS = ("query", "absolute")

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_norm: str = "query",
    ) -> None:
        super().__init__()
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if rope_norm not in self.ROPE_NORMS:
            raise ValueError(f"rope_norm must be one of {self.ROPE_NORMS}, got {rope_norm!r}")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.use_rope = use_rope
        self.rope_norm = rope_norm

        self.q = nn.Linear(d_query, d_model)
        self.kv = nn.Linear(d_kv, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop_value = dropout
        self.resid_drop = nn.Dropout(dropout)

        # Zero-init => identity rotation at the start of training. "absolute"
        # keeps one set per stream, so their ratio — which sets the alignment
        # slope — is learned.
        if use_rope:
            self.rotary_dim = self.hd // 2
            if rope_norm == "absolute":
                self.theta_q = nn.Parameter(torch.zeros(self.rotary_dim))
                self.theta_k = nn.Parameter(torch.zeros(self.rotary_dim))
            else:
                self.theta = nn.Parameter(torch.zeros(self.rotary_dim))

        init_weights_(self)

    def _rotate(
        self,
        x: torch.Tensor,                                            # (B, nh, T, hd)
        theta: torch.Tensor,                                        # (rotary_dim,)
        seq_len: int,
        padding_mask: Optional[torch.Tensor],                       # (B, T) or None
        device: torch.device,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Apply this block's rotary scheme to one stream.
        """

        if self.rope_norm == "absolute":
            cos, sin = absolute_freqs(theta, seq_len, device, start_pos)
        else:
            cos, sin = normalized_freqs(
                theta, x.shape[0], seq_len, padding_mask, device, start_pos
            )

        return rotate_halves(x, cos, sin)                           # (B, nh, T, hd)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_query)
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
        kv_cache: Optional[LayerCache] = None,                      # context keys/values
        start_pos: int = 0,                                         # query positions consumed
    ) -> tuple[torch.Tensor, LayerCache]:
        """
        Returns the output for `x` plus the context keys/values to reuse.
        """
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)

        if kv_cache is not None:
            k, v = kv_cache                                         # (B, nh, S, hd) each
            S = k.shape[2]
        else:
            S = context.shape[1]
            kv = self.kv(context)                                   # (B, S, 2*d_model)
            k, v = kv.split(self.nh * self.hd, dim=-1)              # each (B, S, d_model)
            k = k.view(B, S, self.nh, self.hd).transpose(1, 2)      # (B, nh, S, hd)
            v = v.view(B, S, self.nh, self.hd).transpose(1, 2)      # (B, nh, S, hd)

        # The key rotation is baked into the cache on the first call; only the
        # query side moves from then on. The two streams share one theta unless
        # "absolute" gave each its own.
        if self.use_rope:
            per_stream = self.rope_norm == "absolute"
            q = self._rotate(
                q, self.theta_q if per_stream else self.theta,
                T, query_padding_mask, x.device, start_pos,
            )
            if kv_cache is None:
                k = self._rotate(
                    k, self.theta_k if per_stream else self.theta,
                    S, key_padding_mask, x.device,
                )

        new_cache: LayerCache = (k, v)

        # The key_padding_mask covers the context (length S), not the query stream.
        attn_mask = None                                            # (B, 1, 1, S) or None
        if key_padding_mask is not None:
            attn_mask = pad_attn_mask(key_padding_mask, B, S)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
        )                                                           # (B, nh, T, hd)

        # The output length follows the QUERY stream: attention computes one
        # weighted average per query position.
        D = self.nh * self.hd
        out = out.transpose(1, 2).contiguous().view(B, T, D)        # (B, T, d_model)
        out = self.resid_drop(self.proj(out))                       # (B, T, d_model)

        return out, new_cache                                       # (B, T, d_model), cache
