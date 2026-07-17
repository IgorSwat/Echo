"""Non-Autoregressive decoder for Mimi codec prediction.

Architecturally identical to :class:`echo.ar_decoder.ARDecoder` — same
causal decoder stack with KV-cache support — but operating on **previous**
codec layers to predict the *next* layer:

* 16 **per‑layer** codec embedding tables (one per RVQ codebook).
* Text + ``<SEP>`` conditioning (same as AR).
* No EOS prediction — output is ``CODEC_VOCAB_SIZE`` classes.
* Weight‑tied output: the logits for layer *i* reuse the input embedding
  weights of layer *i* (excl. the pad row).

Training
--------
``forward(text_tokens, prev_codec_tokens, layer_idx)`` concatenates

    [ text , <SEP> , Σ_k codec_emb_k(prev_codec_tokens[:, k, :]) ]

for *k = 0 .. layer_idx‑1* (all previous layers), runs the causal
transformer, and returns ``(B, Tt + 1 + Ta, CODEC_VOCAB_SIZE)`` logits
for layer ``layer_idx``.

Inference
---------
Same ``forward_step(token_ids, token_types, codec_layer_ids, positions,
kv_cache)`` as the AR decoder.  ``codec_layer_ids`` selects which of the
16 per‑layer embedding tables to use for each audio token.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo.config import (
    AUDIO_PAD_ID,
    CODEC_VOCAB_SIZE,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    NAR_D_FF,
    NAR_D_MODEL,
    NAR_DROPOUT,
    NAR_N_HEADS,
    NAR_N_LAYERS,
    NUM_CODEBOOKS,
    TEXT_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Token‑type constants (same as AR, plus per-layer audio)
# ---------------------------------------------------------------------------


class TokenType:
    TEXT = 0
    AUDIO = 1       # generic audio; layer selected via codec_layer_ids
    SEP = 2
    EOS = 3         # unused by NAR (kept for API symmetry)


# ---------------------------------------------------------------------------
# KV cache (identical to AR)
# ---------------------------------------------------------------------------


class KVCache:
    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        self._entries: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self._entries[layer_idx]

    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._entries[layer_idx] is None:
            self._entries[layer_idx] = (new_keys, new_values)
        else:
            old_k, old_v = self._entries[layer_idx]
            self._entries[layer_idx] = (
                torch.cat([old_k, new_keys], dim=2),
                torch.cat([old_v, new_values], dim=2),
            )
        return self._entries[layer_idx]  # type: ignore[return-value]

    @property
    def sequence_length(self) -> int:
        if self._entries[0] is None:
            return 0
        return self._entries[0][0].shape[2]

    def __len__(self) -> int:
        return self.sequence_length


# ---------------------------------------------------------------------------
# Causal self‑attention (identical to AR)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        use_causal_mask: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        dropout_p = self.attn_drop.p if self.training else 0.0

        if attn_mask is not None:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
        elif use_causal_mask:
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(out))


# ---------------------------------------------------------------------------
# Feed‑forward (identical to AR)
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x))))


# ---------------------------------------------------------------------------
# Transformer block (identical to AR)
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        use_causal_mask: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x), kv_cache=kv_cache,
                                     layer_idx=layer_idx,
                                     use_causal_mask=use_causal_mask,
                                     attn_mask=attn_mask))
        x = x + self.drop(self.ff(self.ln2(x)))
        return x


# ---------------------------------------------------------------------------
# NAR Decoder
# ---------------------------------------------------------------------------


class NARDecoder(nn.Module):
    """Non‑autoregressive codec‑layer predictor — identical decoder stack as
    :class:`echo.ar_decoder.ARDecoder`, with per‑layer codec embeddings.
    """

    def __init__(
        self,
        d_model: int = NAR_D_MODEL,
        n_heads: int = NAR_N_HEADS,
        d_ff: int = NAR_D_FF,
        n_layers: int = NAR_N_LAYERS,
        dropout: float = NAR_DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # ---- input embeddings (same as AR, but 16 codec tables) ------------
        self.text_embedding = nn.Embedding(TEXT_VOCAB_SIZE, d_model)
        # 16 per‑layer codec embedding tables (each has a pad row).
        self.code_embeddings = nn.ModuleList(
            [nn.Embedding(CODEC_VOCAB_SIZE + 1, d_model) for _ in range(NUM_CODEBOOKS)]
        )
        self.sep_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        # No eos_embedding — the NAR does not predict EOS.

        # ---- positional embeddings (same as AR) ----------------------------
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_LENGTH, d_model)
        self.audio_pos_embedding = nn.Embedding(MAX_AUDIO_LENGTH + 1, d_model)

        # ---- transformer (same as AR) --------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout)
             for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        self.emb_drop = nn.Dropout(dropout)

        # ---- weight init ---------------------------------------------------
        self.apply(self._init_weights)
        nn.init.normal_(self.sep_embedding, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Output projection (weight‑tied to target layer's embedding)
    # ------------------------------------------------------------------
    def _output_logits(self, hidden: torch.Tensor,
                       target_layer: int) -> torch.Tensor:
        return F.linear(
            hidden,
            self.code_embeddings[target_layer].weight[:CODEC_VOCAB_SIZE],
        )  # (B, T, CODEC_VOCAB_SIZE)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        text_tokens: torch.Tensor,               # (B, Tt)
        prev_codec_tokens: torch.Tensor,         # (B, L, Ta)  -- layers 0..L-1
        layer_idx: int,                          # target layer (must equal L)
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict codec layer ``layer_idx`` given all previous layers.

        Parameters
        ----------
        text_tokens
            ``(B, Tt)`` padded phoneme ids.
        prev_codec_tokens
            ``(B, L, Ta)`` stacked codec tokens from layers 0..L‑1, where
            ``L == layer_idx``.  Each layer *k* is embedded with
            ``code_embeddings[k]`` and the embeddings are summed.
        layer_idx
            The target codec layer (1..15).
        key_padding_mask
            ``(B, Tt + 1 + Ta)`` boolean, ``True`` = ignore in attention.

        Returns
        -------
        ``(B, Tt + 1 + Ta, CODEC_VOCAB_SIZE)`` logits.
        """
        B, Tt = text_tokens.shape
        L, Ta = prev_codec_tokens.shape[1], prev_codec_tokens.shape[2]
        device = text_tokens.device

        # --- text -----------------------------------------------------------
        text_pos = torch.arange(Tt, device=device)
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_pos)

        # --- <SEP> ----------------------------------------------------------
        sep_pos_emb = self.audio_pos_embedding.weight[0]
        sep_emb = (self.sep_embedding + sep_pos_emb)
        sep_emb = sep_emb.view(1, 1, -1).expand(B, 1, -1)

        # --- previous codec layers (summed) ---------------------------------
        audio_pos = torch.arange(1, Ta + 1, device=device)
        audio_emb = torch.zeros(B, Ta, self.d_model, device=device,
                                dtype=self.code_embeddings[0].weight.dtype)
        for k in range(L):
            audio_emb = audio_emb + self.code_embeddings[k](prev_codec_tokens[:, k, :])

        # Add audio positional embedding (once, after summing layers).
        audio_emb = audio_emb + self.audio_pos_embedding(audio_pos)

        # --- concatenate ----------------------------------------------------
        x = torch.cat([text_emb, sep_emb, audio_emb], dim=1)
        x = self.emb_drop(x)

        # --- attention mask (bidirectional) --------------------------------
        # The NAR predicts codebook ``layer_idx`` from the *lower* codebooks
        # only; the target is absent from the input, so there is nothing to
        # leak by attending forward.  Attention is therefore fully
        # bidirectional over ``[text, <SEP>, audio]`` -- we apply only a
        # key-padding mask (ignore padded positions) and no causal triangle.
        attn_mask = None
        if key_padding_mask is not None:
            T_total = x.shape[1]
            neg = -1e4
            # Additive key-padding mask (B, 1, 1, T): neg on padded keys,
            # broadcast over query positions and heads.
            pad = torch.zeros(B, 1, 1, T_total, device=device, dtype=x.dtype)
            pad = pad.masked_fill(key_padding_mask.view(B, 1, 1, T_total), neg)
            attn_mask = pad

        # --- transformer blocks (non-causal) --------------------------------
        for block in self.blocks:
            x = block(x, use_causal_mask=False, attn_mask=attn_mask)

        x = self.ln_f(x)
        return self._output_logits(x, target_layer=layer_idx)

    # ------------------------------------------------------------------
    # Incremental forward (with KV cache)
    # ------------------------------------------------------------------
    def forward_step(
        self,
        token_ids: torch.Tensor,         # (B, T)
        token_types: torch.Tensor,       # (B, T)  values from TokenType
        codec_layer_ids: torch.Tensor,   # (B, T)  which codec layer (0..15)
        positions: torch.Tensor,         # (B, T)  index into pos embedding
        kv_cache: Optional[KVCache] = None,
        target_layer: Optional[int] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Process one or more tokens with optional KV cache.

        Same interface as :meth:`ARDecoder.forward_step`, plus
        ``codec_layer_ids`` to select per‑layer embeddings for AUDIO tokens.

        Returns
        -------
        logits :  ``(B, T, CODEC_VOCAB_SIZE)``.
        kv_cache : updated cache.
        """
        B, T = token_ids.shape

        x = self._embed_tokens(token_ids, token_types, codec_layer_ids, positions)
        x = self.emb_drop(x)

        is_prefill = kv_cache is None or kv_cache.sequence_length == 0
        if kv_cache is None:
            kv_cache = KVCache(self.n_layers)

        if not is_prefill and T > 1:
            warnings.warn(
                "forward_step called with T > 1 during decode (cache not empty). "
                "Pass tokens one at a time after the prefill.",
                stacklevel=2,
            )

        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i,
                      use_causal_mask=is_prefill)

        x = self.ln_f(x)
        # Use the target layer for weight‑tied output.  When not provided
        # (e.g. during prefill of text + SEP portion), default to layer 1.
        tl = target_layer if target_layer is not None else 1
        logits = self._output_logits(x, target_layer=tl)
        return logits, kv_cache

    # ------------------------------------------------------------------
    # Embedding helper for forward_step
    # ------------------------------------------------------------------
    def _embed_tokens(
        self,
        token_ids: torch.Tensor,
        token_types: torch.Tensor,
        codec_layer_ids: torch.Tensor,   # which embedding table for AUDIO tokens
        positions: torch.Tensor,
    ) -> torch.Tensor:
        B, T = token_ids.shape
        device = token_ids.device
        dtype = self.code_embeddings[0].weight.dtype

        x = torch.zeros(B, T, self.d_model, device=device, dtype=dtype)

        # Text.
        is_text = token_types == TokenType.TEXT
        if is_text.any():
            ids = token_ids.clamp(0, TEXT_VOCAB_SIZE - 1)
            pos = positions.clamp(0, MAX_TEXT_LENGTH - 1)
            emb = self.text_embedding(ids) + self.text_pos_embedding(pos)
            x = x + emb * is_text.unsqueeze(-1).to(dtype)

        # Audio — per‑layer embedding lookup.
        is_audio = token_types == TokenType.AUDIO
        if is_audio.any():
            ids = token_ids.clamp(0, CODEC_VOCAB_SIZE - 1)
            pos = positions.clamp(0, MAX_AUDIO_LENGTH)
            layer_ids = codec_layer_ids.clamp(0, NUM_CODEBOOKS - 1)

            emb = torch.zeros(B, T, self.d_model, device=device, dtype=dtype)
            for k in range(NUM_CODEBOOKS):
                layer_mask = is_audio & (layer_ids == k)
                if not layer_mask.any():
                    continue
                # Gather the per‑layer embedding for these tokens.
                layer_emb = self.code_embeddings[k](ids)   # (B, T, d)
                emb = emb + layer_emb * layer_mask.unsqueeze(-1).to(dtype)
            x = x + emb + self.audio_pos_embedding(pos) * is_audio.unsqueeze(-1).to(dtype)

        # <SEP>
        is_sep = token_types == TokenType.SEP
        if is_sep.any():
            pos = positions.clamp(0, MAX_AUDIO_LENGTH)
            emb = self.sep_embedding.view(1, 1, -1) + self.audio_pos_embedding(pos)
            x = x + emb * is_sep.unsqueeze(-1).to(dtype)

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters())

        excluded = set()
        for emb in self.code_embeddings:
            excluded.add(id(emb.weight))
        excluded.add(id(self.text_embedding.weight))
        excluded.add(id(self.sep_embedding))
        excluded.add(id(self.text_pos_embedding.weight))
        excluded.add(id(self.audio_pos_embedding.weight))

        total = 0
        for p in self.parameters():
            if id(p) not in excluded:
                total += p.numel()
        return total