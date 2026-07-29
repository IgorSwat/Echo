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
    Multi-head cross-attention with rotary position embeddings on BOTH the
    query and key streams (Supertonic-style).

    Positions of each stream are normalized by their valid (mask) lengths, so
    both live in [0, 1]: a frame 30% through the audio shares a coordinate
    with a token 30% through the text. Since both streams are rotated, the
    attention score depends on content AND the relative normalized position
    theta * (p_q - p_k): matches at equal relative position are preserved
    exactly, off-diagonal matches are scrambled by the rotation. This gives a
    soft, learnable diagonal alignment prior by construction.

    The rotary frequencies ``theta`` are learnable and zero-initialized, so
    training starts as pure content-based attention and the positional prior
    grows only as needed. Rotation uses split-half pairing: head dim i is
    paired with dim i + hd//2.
    """

    def __init__(
        self,
        d_query: int,
        d_kv: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        use_rope: bool = False,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.use_rope = use_rope

        self.q = nn.Linear(d_query, d_model)
        self.kv = nn.Linear(d_kv, 2 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.attn_drop_value = dropout
        self.resid_drop = nn.Dropout(dropout)

        # Learnable rotary frequencies (one per head-dim pair). Zero-init =>
        # identity rotation at the start of training.
        if use_rope:
            self.rotary_dim = self.hd // 2
            self.theta = nn.Parameter(torch.zeros(self.rotary_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.q.bias)
        nn.init.normal_(self.kv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.kv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def _rotary_angles(
        self,
        batch_size: int,
        seq_len: int,
        key_padding_mask: Optional[torch.Tensor],               # (B, T) or None
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized positions ([0, 1] over the valid region) times theta."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
        if key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1).float()
        else:
            lengths = torch.full((batch_size, 1, 1), float(seq_len), device=device)
        ang = (pos / lengths) * self.theta.view(1, 1, -1)         # (B, T, rotary_dim)
        return ang.sin(), ang.cos()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        # x: (B, nh, T, hd); sin/cos: (B, T, rotary_dim)
        rd = sin.shape[-1]
        s = sin.unsqueeze(1)                                      # (B, 1, T, rd)
        c = cos.unsqueeze(1)                                      # (B, 1, T, rd)
        x1, x2 = x[..., :rd], x[..., rd : 2 * rd]                 # paired halves
        rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        tail = x[..., 2 * rd :]                                   # untouched dims (odd hd)
        if tail.shape[-1] > 0:
            rot = torch.cat([rot, tail], dim=-1)
        return rot

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, d_query)
        context: torch.Tensor,                                      # (B, S, d_kv)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        S = context.shape[1]

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # (B, nh, T, hd)
        kv = self.kv(context)                                         # (B, S, 2*d_model)
        k, v = kv.split(self.nh * self.hd, dim=-1)                    # each (B, S, d_model)
        k = k.view(B, S, self.nh, self.hd).transpose(1, 2)            # (B, nh, S, hd)
        v = v.view(B, S, self.nh, self.hd).transpose(1, 2)            # (B, nh, S, hd)

        # Dual-stream rotary at length-normalized positions.
        if self.use_rope:
            sin_q, cos_q = self._rotary_angles(B, T, query_padding_mask, x.device)
            sin_k, cos_k = self._rotary_angles(B, S, key_padding_mask, x.device)
            q = self._apply_rotary(q, sin_q, cos_q)                  # (B, nh, T, hd)
            k = self._apply_rotary(k, sin_k, cos_k)                  # (B, nh, S, hd)

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
        D = self.nh * self.hd
        out = out.transpose(1, 2).contiguous().view(B, T, D)         # (B, T, d_model)
        out = self.resid_drop(self.proj(out))                        # (B, T, d_model)

        return out                                                   # (B, T, d_model)
