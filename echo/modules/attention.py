from echo import config
from echo.modules.rope import rope_cos_sin, apply_rotary_emb

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalSelfAttention(nn.Module):
    """
    Multi-head bidirectional self-attention with RoPE.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop_value = dropout
        self.resid_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        B, T, D = x.shape

        qkv = self.qkv(x)                                             # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)                                # each (B, T, D)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)

        # Optional RoPE
        if self.use_rope:
            cos, sin = rope_cos_sin(self.hd, T + 1, device=x.device, theta=self.rope_theta)
            q = apply_rotary_emb(q, cos, sin, 0)                     # (B, nh, T, hd)
            k = apply_rotary_emb(k, cos, sin, 0)                     # (B, nh, T, hd)

        # Masking
        # This serves a very concrete purpose: during training, some of the input entries
        # might just be padding. We want to explicitely disable attention interactions for them.
        attn_mask = None                                             # (B, 1, 1, T) or None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, T):
                raise ValueError(f"expected key_padding_mask shape {(B, T)}, got {tuple(key_padding_mask.shape)}")
            attn_mask = key_padding_mask.view(B, 1, 1, T)            # (B, 1, 1, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
        )                                                            # (B, nh, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)         # (B, T, D)
        out = self.resid_drop(self.proj(out))                        # (B, T, D)

        return out                                                   # (B, T, D)


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with RoPE on the query stream.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta

        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop_value = dropout
        self.resid_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.q.bias)
        nn.init.normal_(self.kv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.kv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        context: torch.Tensor,                                      # (B, S, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
    ) -> torch.Tensor:
        B, T, D = x.shape
        S = context.shape[1]

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # (B, nh, T, hd)
        kv = self.kv(context)                                         # (B, S, 2D)
        k, v = kv.split(D, dim=-1)                                    # each (B, S, D)
        k = k.view(B, S, self.nh, self.hd).transpose(1, 2)            # (B, nh, S, hd)
        v = v.view(B, S, self.nh, self.hd).transpose(1, 2)            # (B, nh, S, hd)

        # Optional RoPE on the query stream
        if self.use_rope:
            cos, sin = rope_cos_sin(self.hd, T + 1, device=x.device, theta=self.rope_theta)
            q = apply_rotary_emb(q, cos, sin, 0)                     # (B, nh, T, hd)
            k = apply_rotary_emb(k, cos, sin, 0)                     # (B, nh, S, hd)

        # Masking
        # The key_padding_mask masks the context (length S), not the query stream.
        attn_mask = None                                             # (B, 1, 1, S) or None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, S):
                raise ValueError(f"expected key_padding_mask shape {(B, S)}, got {tuple(key_padding_mask.shape)}")
            attn_mask = key_padding_mask.view(B, 1, 1, S)            # (B, 1, 1, S)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
        )                                                            # (B, nh, T, hd)

        # The length of the cross-attention output follows the QUERY sequence,
        # since attention computes a single weighted average per query position.
        out = out.transpose(1, 2).contiguous().view(B, T, D)         # (B, T, D)
        out = self.resid_drop(self.proj(out))                        # (B, T, D)

        return out                                                   # (B, T, D)
