from echo import config

from echo.modules.types import LayerCache

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with RoPE.
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

        if use_rope and max_seq_len is not None:
            cos, sin = self._rope_freqs(max_seq_len, device=torch.device("cpu"))
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def _rope_freqs(self, max_pos: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        theta = self.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=device, dtype=torch.float32) / self.hd))
        t = torch.arange(max_pos, device=device, dtype=torch.float32)
        freqs = torch.outer(t, freqs)
        return freqs.cos(), freqs.sin()

    @staticmethod
    def _apply_rotary_emb(
        x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0
    ) -> torch.Tensor:
        T = x.size(2)
        
        x_rot = x.float().reshape(*x.shape[:-1], -1, 2)
        c = cos[start_pos:start_pos + T].view(1, 1, T, -1)
        s = sin[start_pos:start_pos + T].view(1, 1, T, -1)
        out = torch.empty_like(x_rot)
        out[..., 0] = x_rot[..., 0] * c - x_rot[..., 1] * s
        out[..., 1] = x_rot[..., 0] * s + x_rot[..., 1] * c
        
        return out.flatten(-2).to(x.dtype)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        kv_cache: Optional[LayerCache] = None,                        # keys/values so far
        start_pos: int = 0,                                         # positions already cached
    ) -> tuple[torch.Tensor, LayerCache]:
        """
        Returns the output for `x` plus the keys/values to feed the next call.
        """

        B, T, D = x.shape

        qkv = self.qkv(x)                                             # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)                                # each (B, T, D)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)            # (B, nh, T, hd)

        # Optional RoPE.
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
                cos, sin = self._rope_freqs(end + 1, device=x.device)
            q = self._apply_rotary_emb(q, cos, sin, start_pos)
            k = self._apply_rotary_emb(k, cos, sin, start_pos)

        # Prepend the cached keys/values; from here on the key axis has length S.
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            if cache_k.shape[2] != start_pos:
                raise ValueError(
                    f"kv_cache holds {cache_k.shape[2]} positions but start_pos is {start_pos}"
                )
            k = torch.cat([cache_k, k], dim=2)                       # (B, nh, S, hd)
            v = torch.cat([cache_v, v], dim=2)                       # (B, nh, S, hd)
        new_cache: LayerCache = (k, v)
        S = k.shape[2]

        # Masking
        # This serves a very concrete purpose: during training, some of the input entries
        # might just be padding. We want to explicitely disable attention interactions for them.
        attn_mask = None                                             # (B, 1, 1, S) or None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, S):
                raise ValueError(f"expected key_padding_mask shape {(B, S)}, got {tuple(key_padding_mask.shape)}")
            attn_mask = key_padding_mask.view(B, 1, 1, S)            # (B, 1, 1, S)

        # Causal masking: prevent attending to future positions. Query i sits at
        # absolute position start_pos + i, so it may see keys 0 ... start_pos + i.
        # When the whole prefix is cached every key is already in the past and the
        # mask degenerates to "attend to everything", which is left as None.
        if self.mode == "causal" and not (T == 1 and S == start_pos + 1):
            q_pos = torch.arange(start_pos, start_pos + T, device=x.device).view(T, 1)
            k_pos = torch.arange(S, device=x.device).view(1, S)
            causal_mask = k_pos <= q_pos                             # (T, S)
            if attn_mask is not None:
                # Combine key_padding_mask with the causal mask.
                # (B, 1, 1, S) & (1, 1, T, S) -> broadcast to (B, 1, T, S)
                attn_mask = attn_mask & causal_mask.unsqueeze(0).unsqueeze(0)
            else:
                attn_mask = causal_mask.view(1, 1, T, S)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_value if self.training else 0.0,
        )                                                            # (B, nh, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, D)         # (B, T, D)
        out = self.resid_drop(self.proj(out))                        # (B, T, D)

        return out, new_cache                                        # (B, T, D), cache


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention with rotary position embeddings on BOTH the
    query and key streams.

    `rope_norm` picks how positions are encoded:

    - "query"    — each stream's positions are normalized by its own valid
                   length, so both land in [0, 1] and the alignment diagonal
                   adapts to each utterance's true rate. One shared learnable
                   frequency. Needs the total query length up front, which a
                   non-autoregressive model has and an autoregressive one does not.
    - "absolute" — raw indices on both streams, with an independent learnable
                   frequency each. The attention phase difference vanishes on
                   `s = (theta_q / theta_k) * i`, so the diagonal's slope is
                   learned rather than supplied, and nothing depends on a length
                   that is unknown mid-decode. Because the frequencies are
                   vectors over rotary dims, different dims can settle on
                   different slopes.
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

        # Learnable rotary frequencies (one per head-dim pair). Zero-init =>
        # identity rotation at the start of training. "absolute" keeps one set
        # per stream so their ratio, which sets the alignment slope, is learned.
        if use_rope:
            self.rotary_dim = self.hd // 2
            if rope_norm == "absolute":
                self.theta_q = nn.Parameter(torch.zeros(self.rotary_dim))
                self.theta_k = nn.Parameter(torch.zeros(self.rotary_dim))
            else:
                self.theta = nn.Parameter(torch.zeros(self.rotary_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.q.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.q.bias)
        nn.init.normal_(self.kv.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.kv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.proj.bias)

    def _rope_freqs(
        self,
        batch_size: int,
        seq_len: int,
        key_padding_mask: Optional[torch.Tensor],               # (B, T) or None
        device: torch.device,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalized positions ([0, 1] over the valid region) times learnable theta.

        Unlike the standard RoPE in SelfAttention:
        - theta is *learnable* (zero-initialized), not a fixed hyperparameter.
        - positions are *normalized* to [0, 1] based on each sequence's actual
          (non-padded) length, so variable-length contexts are handled robustly.
        - dual-stream: Q and K receive position encodings from their own sequences,
          enabling the model to reason about relative distances between two
          different timelines.
        """

        end = start_pos + seq_len
        pos = torch.arange(start_pos, end, device=device, dtype=torch.float32).view(1, seq_len, 1)
        if key_padding_mask is not None:
            lengths = key_padding_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1).float()
        else:
            # Mid-decode the total length is not known, so the sequence seen so
            # far stands in for it — the same value the uncached path would use
            # at this step.
            lengths = torch.full((batch_size, 1, 1), float(end), device=device)
        ang = (pos / lengths) * self.theta.view(1, 1, -1)         # (B, T, rotary_dim)
        return ang.sin(), ang.cos()

    @staticmethod
    def _rope_freqs_absolute(
        seq_len: int,
        theta: torch.Tensor,                                    # (rotary_dim,)
        device: torch.device,
        start_pos: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Raw positions times a per-stream learnable theta.

        No length enters, so a position\'s encoding never changes as the
        sequence grows — the property autoregressive decoding needs. Broadcast
        over the batch, since every row shares the same index grid.
        """

        pos = torch.arange(
            start_pos, start_pos + seq_len, device=device, dtype=torch.float32
        ).view(1, seq_len, 1)

        ang = pos * theta.view(1, 1, -1)                         # (1, T, rotary_dim)
        
        return ang.sin(), ang.cos()

    @staticmethod
    def _apply_rotary_emb(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        """
        Apply rotary embeddings using split-half formulation.

        Mathematically equivalent to the adjacent-pairs formulation in
        SelfAttention, but operates on the first/second halves
        of the head dimension.
        """

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
        context: Optional[torch.Tensor],                            # (B, S, d_kv), None if cached
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        query_padding_mask: Optional[torch.Tensor] = None,          # (B, T) or None
        kv_cache: Optional[LayerCache] = None,                        # context keys/values
        start_pos: int = 0,                                         # query positions consumed
    ) -> tuple[torch.Tensor, LayerCache]:
        """
        Returns the output for `x` plus the context keys/values to reuse.
        """
        B, T, _ = x.shape

        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)    # (B, nh, T, hd)

        if kv_cache is not None:
            k, v = kv_cache                                            # (B, nh, S, hd) each
            S = k.shape[2]
        else:
            S = context.shape[1]
            kv = self.kv(context)                                     # (B, S, 2*d_model)
            k, v = kv.split(self.nh * self.hd, dim=-1)                # each (B, S, d_model)
            k = k.view(B, S, self.nh, self.hd).transpose(1, 2)        # (B, nh, S, hd)
            v = v.view(B, S, self.nh, self.hd).transpose(1, 2)        # (B, nh, S, hd)

        # Dual-stream rotary at length-normalized positions. The key rotation is
        # baked into the cache on the first call; only the query side moves.
        if self.use_rope:
            if self.rope_norm == "absolute":
                sin_q, cos_q = self._rope_freqs_absolute(
                    T, self.theta_q, x.device, start_pos
                )
            else:
                sin_q, cos_q = self._rope_freqs(
                    B, T, query_padding_mask, x.device, start_pos
                )
            q = self._apply_rotary_emb(q, sin_q, cos_q)                  # (B, nh, T, hd)

            if kv_cache is None:
                if self.rope_norm == "absolute":
                    sin_k, cos_k = self._rope_freqs_absolute(S, self.theta_k, x.device)
                else:
                    sin_k, cos_k = self._rope_freqs(B, S, key_padding_mask, x.device)
                k = self._apply_rotary_emb(k, sin_k, cos_k)              # (B, nh, S, hd)

        new_cache: LayerCache = (k, v)

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

        return out, new_cache                                        # (B, T, d_model), cache