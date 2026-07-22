from echo import config
from echo.modules.rope import rope_cos_sin, apply_rotary_emb

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE and an explicit (optional) KV cache.
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
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, D = x.shape

        qkv = self.qkv(x)                                             # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        kv_total_len = T
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # Optional RoPE
        if self.use_rope:
            cos, sin = rope_cos_sin(self.hd, start_pos + T + 1, device=x.device, theta=self.rope_theta)
            q = apply_rotary_emb(q, cos, sin, start_pos)
            if kv_cache is not None:
                k_new = apply_rotary_emb(k[:, :, -T:], cos, sin, start_pos)
                k = torch.cat([k[:, :, :-T], k_new], dim=2)
            else:
                k = apply_rotary_emb(k, cos, sin, start_pos)
        new_kv = (k, v)

        # Masking
        is_causal = kv_cache is None and key_padding_mask is None
        attn_mask = None
        if key_padding_mask is not None:
            if kv_cache is not None:
                raise ValueError("key_padding_mask is only supported without a KV cache")
            if key_padding_mask.shape != (B, T):
                raise ValueError(f"expected key_padding_mask shape {(B, T)}, got {tuple(key_padding_mask.shape)}")
            causal = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            attn_mask = causal.view(1, 1, T, T) & key_padding_mask.view(B, 1, 1, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
            is_causal=is_causal,
        )                                                            # (B, nh, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_drop(self.proj(out))

        return out, new_kv