from supertonic import config

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelPosMultiHeadAttention(nn.Module):
    """Self-attention with learnable relative positional embeddings (VITS style)."""

    def __init__(self, d_model: int, num_heads: int, window_size: int = 4) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = d_model // num_heads
        self.window_size = window_size

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self.emb_rel_k = nn.Parameter(torch.zeros(1, 2 * window_size + 1, self.hd))
        self.emb_rel_v = nn.Parameter(torch.zeros(1, 2 * window_size + 1, self.hd))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q, self.k, self.v, self.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(m.bias)

    def _get_relative_embeddings(self, emb: torch.Tensor, length: int) -> torch.Tensor:
        pad_len = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        if pad_len > 0:
            emb = F.pad(emb, (0, 0, pad_len, pad_len))
        return emb[:, start : start + 2 * length - 1]               # (1, 2L-1, hd)

    @staticmethod
    def _relative_position_to_absolute_position(x: torch.Tensor) -> torch.Tensor:
        # x: (B, nh, L, 2L-1) -> (B, nh, L, L)
        b, h, l, _ = x.shape
        x = F.pad(x, (0, 1))                                        # (B, nh, L, 2L)
        x = x.reshape(b, h, 2 * l * l)                              # (B, nh, 2L^2)
        x = x[:, :, : 2 * l * l - l]                                # (B, nh, L(2L-1))
        x = x.reshape(b, h, l, 2 * l - 1)
        return x[:, :, :, l - 1 :]                                  # (B, nh, L, L)

    @staticmethod
    def _absolute_position_to_relative_position(x: torch.Tensor) -> torch.Tensor:
        # x: (B, nh, L, L) -> (B, nh, L, 2L-1)
        b, h, l, _ = x.shape
        x = F.pad(x, (0, l - 1))
        x = x.reshape(b, h, 2 * l * l - l)
        x = F.pad(x, (l, 0))
        x = x.reshape(b, h, l, 2 * l)
        return x[:, :, :, 1:]                                       # (B, nh, L, 2L-1)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        mask: Optional[torch.Tensor] = None,                        # (B, T) bool or None
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        v = self.v(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)

        query = q / math.sqrt(self.hd)
        scores = torch.matmul(query, k.transpose(-2, -1))           # (B, nh, T, T)

        key_rel = self._get_relative_embeddings(self.emb_rel_k, T)  # (1, 2T-1, hd)
        rel_logits = torch.matmul(query, key_rel.unsqueeze(0).transpose(-2, -1))
        scores = scores + self._relative_position_to_absolute_position(rel_logits)

        if mask is not None:
            valid = mask.view(B, 1, T, 1) & mask.view(B, 1, 1, T)   # (B, 1, T, T)
            scores = scores.masked_fill(~valid, -1e4)
        p_attn = F.softmax(scores, dim=-1)

        output = torch.matmul(p_attn, v)
        rel_weights = self._absolute_position_to_relative_position(p_attn)
        value_rel = self._get_relative_embeddings(self.emb_rel_v, T)
        output = output + torch.matmul(rel_weights, value_rel.unsqueeze(0))

        out = output.transpose(1, 2).contiguous().view(B, T, D)     # (B, T, D)
        return self.proj(out)


class RotaryCrossAttention(nn.Module):
    """
    Cross-attention with rotary embeddings at mask-length-normalized positions
    on BOTH streams: a frame 30% through the audio shares a coordinate with a
    token 30% through the text, giving a soft learnable diagonal alignment
    prior. Frequencies ``theta`` are learnable and zero-initialized, so
    training starts as pure content-based attention. Score scale is 1/16.
    """

    def __init__(
        self,
        dim: int,
        kv_dim: int,
        num_heads: int,
        rotary_dim: int = 32,
        scale: float = 16.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = dim // num_heads
        self.rotary_dim = rotary_dim
        self.scale = scale

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(kv_dim, dim)
        self.v = nn.Linear(kv_dim, dim)
        self.proj = nn.Linear(dim, dim)

        # Learnable rotary frequencies (one per head-dim pair). Zero-init =>
        # identity rotation at the start of training.
        self.theta = nn.Parameter(torch.zeros(rotary_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q, self.k, self.v, self.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(m.bias)

    def _angles(
        self,
        batch_size: int,
        seq_len: int,
        key_padding_mask: Optional[torch.Tensor],                   # (B, T) or None
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized positions ([0, 1] over the valid region) times theta."""
        pos = torch.arange(seq_len, device=device, dtype=torch.float32).view(1, seq_len, 1)
        if key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1).float()
        else:
            lengths = torch.full((batch_size, 1, 1), float(seq_len), device=device)
        ang = (pos / lengths) * self.theta.view(1, 1, -1)           # (B, T, rotary_dim)
        return ang.sin(), ang.cos()

    @staticmethod
    def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        # x: (B, nh, T, hd); sin/cos: (B, T, rotary_dim)
        rd = sin.shape[-1]
        s = sin.unsqueeze(1)                                        # (B, 1, T, rd)
        c = cos.unsqueeze(1)                                        # (B, 1, T, rd)
        x1, x2 = x[..., :rd], x[..., rd : 2 * rd]                   # paired halves
        rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        tail = x[..., 2 * rd :]                                     # untouched dims
        if tail.shape[-1] > 0:
            rot = torch.cat([rot, tail], dim=-1)
        return rot

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim)
        context: torch.Tensor,                                      # (B, S, kv_dim)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) bool or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) bool or None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        S = context.shape[1]

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = self.k(context).view(B, S, self.nh, self.hd).transpose(1, 2)
        v = self.v(context).view(B, S, self.nh, self.hd).transpose(1, 2)

        # Dual-stream rotary at length-normalized positions.
        sin_q, cos_q = self._angles(B, T, query_padding_mask, x.device)
        sin_k, cos_k = self._angles(B, S, key_padding_mask, x.device)
        q = self._apply_rotary(q, sin_q, cos_q)
        k = self._apply_rotary(k, sin_k, cos_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, nh, T, S)
        if key_padding_mask is not None:
            scores = scores.masked_fill(~key_padding_mask.view(B, 1, 1, S), float("-inf"))
        p = F.softmax(scores, dim=-1)
        if query_padding_mask is not None:
            # Zero attention rows of padded queries.
            p = p.masked_fill(~query_padding_mask.view(B, 1, T, 1), 0.0)

        out = torch.matmul(p, v)                                    # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)

        return self.proj(out)                                       # (B, T, dim)


class TanhAttention(nn.Module):
    """
    Multi-head attention over style tokens: keys are passed through tanh and
    scores are scaled by 1/16. Style tokens are always all valid (no key mask);
    padded query rows are zeroed after the softmax.
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        n_units: int,
        out_dim: int,
        num_heads: int,
        scale: float = 16.0,
    ) -> None:
        super().__init__()
        if n_units % num_heads != 0:
            raise ValueError(f"n_units ({n_units}) must be divisible by num_heads ({num_heads})")

        self.nh = num_heads
        self.hd = n_units // num_heads
        self.scale = scale

        self.q = nn.Linear(q_dim, n_units)
        self.k = nn.Linear(kv_dim, n_units)
        self.v = nn.Linear(kv_dim, n_units)
        self.proj = nn.Linear(n_units, out_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in (self.q, self.k, self.v, self.proj):
            nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, q_dim)
        keys: torch.Tensor,                                         # (B, S, kv_dim)
        values: torch.Tensor,                                       # (B, S, kv_dim)
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) bool or None
    ) -> torch.Tensor:
        B, T, _ = x.shape
        S = keys.shape[1]

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = torch.tanh(self.k(keys))                                # (B, S, n_units)
        k = k.view(B, S, self.nh, self.hd).transpose(1, 2)          # (B, nh, S, hd)
        v = self.v(values).view(B, S, self.nh, self.hd).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (B, nh, T, S)
        p = F.softmax(scores, dim=-1)
        if query_padding_mask is not None:
            p = p.masked_fill(~query_padding_mask.view(B, 1, T, 1), 0.0)

        out = torch.matmul(p, v)                                    # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, self.nh * self.hd)

        return self.proj(out)                                       # (B, T, out_dim)
