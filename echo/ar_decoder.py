"""Autoregressive Transformer decoder for Mimi‑codec prediction.

Implements a causal language model that predicts the **first codebook** of
the Mimi audio codec conditioned on a phoneme sequence.  The architecture
follows the VALL‑E design:

* Separate embedding tables for **text** (phonemes, 128 entries) and
  **audio** (codec tokens, 2048 entries).
* Two dedicated learned vectors for the ``<SEP>`` and ``<EOS>`` special
  tokens.
* Learned positional embeddings – separate for the text and audio halves.
* Weight‑tied output projection: the 2048 audio logits reuse the audio
  embedding weights; the single EOS logit reuses the EOS embedding vector.
* A per‑layer **KV cache** for efficient autoregressive generation.

Model input (training)
----------------------
``forward(text_tokens, audio_tokens)`` concatenates

    [ text_emb(t_0..t_{Tt-1}) , <SEP> , audio_emb(a_0..a_{Ta-1}) ]

adds positional embeddings, runs the causal transformer, and returns
logits of shape ``(B, Tt + 1 + Ta, OUTPUT_VOCAB_SIZE)``.

The logits at the ``<SEP>`` position predict ``a_0``; the logits at
``a_{Ta-1}`` predict ``<EOS>``.

Incremental inference
---------------------
``forward_step(token_ids, token_types, positions, kv_cache)`` processes one
or more tokens with an optional KV cache:

* **Prefill** – ``kv_cache`` is ``None`` or empty.  Pass the full prompt
  (text + SEP + audio prefix) in a single call.  Causal masking is applied.
* **Decode** – ``kv_cache`` already contains tokens.  Pass a single new
  token.  No causal mask is needed (the lone query attends to every cached
  key).

The caller is responsible for tracking *token types* (text / audio / SEP /
EOS) and *positions* (index into the respective positional‑embedding table).
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo.config import (
    AR_D_FF,
    AR_D_MODEL,
    AR_DROPOUT,
    AR_N_HEADS,
    AR_N_LAYERS,
    AUDIO_PAD_ID,
    CODEC_VOCAB_SIZE,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    OUTPUT_VOCAB_SIZE,
    TEXT_VOCAB_SIZE,
)

# ---------------------------------------------------------------------------
# Token‑type constants
# ---------------------------------------------------------------------------
# Used by ``forward_step`` to select the correct embedding table and
# positional‑embedding table for each token.


class TokenType:
    """Integer labels identifying which embedding a token belongs to."""

    TEXT = 0   # phoneme → text_embedding  + text_pos_embedding
    AUDIO = 1  # codec   → audio_embedding + audio_pos_embedding
    SEP = 2    # <SEP>   → sep_embedding    + audio_pos_embedding
    EOS = 3    # <EOS>   → eos_embedding    + audio_pos_embedding


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------


class KVCache:
    """Per‑layer key / value cache for autoregressive decoding.

    Keys and values are stored with shape ``(B, n_heads, T, head_dim)`` so
    they can be fed directly to ``F.scaled_dot_product_attention``.
    """

    def __init__(self, n_layers: int) -> None:
        self.n_layers = n_layers
        # Each entry is (keys, values) or None before first use.
        self._entries: list[Optional[tuple[torch.Tensor, torch.Tensor]]] = [None] * n_layers

    # -- read ---------------------------------------------------------------
    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the cached (keys, values) for *layer_idx*, or ``None``."""
        return self._entries[layer_idx]

    # -- write / append -----------------------------------------------------
    def update(
        self,
        layer_idx: int,
        new_keys: torch.Tensor,   # (B, n_heads, T_new, head_dim)
        new_values: torch.Tensor,  # (B, n_heads, T_new, head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append *new_keys* / *new_values* and return the full cached tensors."""
        if self._entries[layer_idx] is None:
            self._entries[layer_idx] = (new_keys, new_values)
        else:
            old_k, old_v = self._entries[layer_idx]
            self._entries[layer_idx] = (
                torch.cat([old_k, new_keys], dim=2),
                torch.cat([old_v, new_values], dim=2),
            )
        return self._entries[layer_idx]  # type: ignore[return-value]

    # -- queries ------------------------------------------------------------
    @property
    def sequence_length(self) -> int:
        """Number of cached time‑steps (0 before the first update)."""
        if self._entries[0] is None:
            return 0
        return self._entries[0][0].shape[2]

    def __len__(self) -> int:  # convenience
        return self.sequence_length


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    """Multi‑head causal self‑attention with optional KV‑cache support."""

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
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
        """Args
        -----
        x
            ``(B, T, d_model)`` hidden states.
        kv_cache
            Optional cache.  When provided, new K/V are appended and the
            full tensors are used for attention.
        layer_idx
            Index of this layer (for cache bookkeeping).
        use_causal_mask
            If ``True`` a lower-triangular mask is applied.  This should be
            ``True`` during training and prefill, and ``False`` during
            single-token decode (where the lone query may attend to every
            cached key).
        attn_mask
            Optional additive attention mask of shape broadcastable to
            ``(B, n_heads, T_q, T_k)`` (e.g. a combined causal + key-padding
            mask).  When provided it takes precedence over
            ``use_causal_mask`` and is passed to ``scaled_dot_product_attention``
            with ``is_causal=False``.
        """
        B, T, C = x.shape

        # Project and reshape to (B, n_heads, T, head_dim).
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Append to cache (if active) so that k, v cover all past + new tokens.
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        dropout_p = self.attn_drop.p if self.training else 0.0

        if attn_mask is not None:
            # Explicit additive mask (e.g. combined causal + key padding).
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )
        elif use_causal_mask:
            # ``is_causal=True`` builds a (T_q, T_k) lower-triangular mask
            # internally.  When T_q == 1 (decode) this is all-True and
            # effectively a no-op, but we set use_causal_mask=False in that
            # case for clarity.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=dropout_p
            )

        # Merge heads back to (B, T, d_model).
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(out))


# ---------------------------------------------------------------------------
# Feed‑forward
# ---------------------------------------------------------------------------


class FeedForward(nn.Module):
    """GELU activated two‑layer MLP (the standard transformer FFN)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x))))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """Pre‑norm transformer decoder block.

    .. code-block:: text

        x → ln1 → attn ─→ (+x) → ln2 → ffn ─→ (+x) → out
    """

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
# Full decoder
# ---------------------------------------------------------------------------


class ARDecoder(nn.Module):
    """VALL‑E style autoregressive decoder for the first Mimi codebook.

    Parameters (constructor)
    ------------------------
    Defaults are pulled from :mod:`echo.config` but can be overridden.

    d_model
        Hidden dimension of the transformer.
    n_heads
        Number of attention heads (``d_model % n_heads == 0`` required).
    d_ff
        Feed‑forward intermediate dimension.
    n_layers
        Number of :class:`TransformerBlock` layers.
    dropout
        Dropout probability.
    """

    def __init__(
        self,
        d_model: int = AR_D_MODEL,
        n_heads: int = AR_N_HEADS,
        d_ff: int = AR_D_FF,
        n_layers: int = AR_N_LAYERS,
        dropout: float = AR_DROPOUT,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers

        # ---- input embeddings ----------------------------------------------
        self.text_embedding = nn.Embedding(TEXT_VOCAB_SIZE, d_model)
        # Audio embedding has one extra row (index AUDIO_PAD_ID) reserved as a
        # learned *padding* embedding used only for batched training input.
        # The weight-tied output projection ignores this extra row (see
        # ``_output_logits``), so the output vocabulary is unaffected.
        self.audio_embedding = nn.Embedding(CODEC_VOCAB_SIZE + 1, d_model)
        # Dedicated learned vectors for the two special tokens.
        self.sep_embedding = nn.Parameter(torch.randn(d_model) * 0.02)
        self.eos_embedding = nn.Parameter(torch.randn(d_model) * 0.02)

        # ---- positional embeddings -----------------------------------------
        # Text positions: 0 .. MAX_TEXT_LENGTH‑1.
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_LENGTH, d_model)
        # Audio positions: index 0 is reserved for <SEP>, indices 1 ..
        # MAX_AUDIO_LENGTH cover the audio tokens → MAX_AUDIO_LENGTH + 1 rows.
        self.audio_pos_embedding = nn.Embedding(MAX_AUDIO_LENGTH + 1, d_model)

        # ---- transformer ---------------------------------------------------
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)

        # ---- dropout on embeddings -----------------------------------------
        self.emb_drop = nn.Dropout(dropout)

        # ---- weight init ---------------------------------------------------
        self.apply(self._init_weights)
        # Re‑scale special‑token vectors (apply uses nn.Module recursion
        # which skips bare Parameters).
        nn.init.normal_(self.sep_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.eos_embedding, mean=0.0, std=0.02)
        # GPT‑2 style scaled init for the projections that write into the
        # residual stream (attn ``o_proj`` and FFN ``w2``).  Downscaling by
        # 1/sqrt(2 * n_layers) keeps the residual‑stream variance from growing
        # with depth, which stabilises and speeds up training of the deeper
        # stack.
        residual_std = 0.02 / math.sqrt(2 * n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=residual_std)

    # ------------------------------------------------------------------
    # Initialization helper
    # ------------------------------------------------------------------
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
    # Output projection (weight‑tied)
    # ------------------------------------------------------------------
    def _output_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute output logits from hidden states.

        The 2048 audio logits share weights with ``audio_embedding``; the
        single EOS logit shares the ``eos_embedding`` vector.  This follows
        the VALL‑E weight‑tying convention.
        """
        # (B, T, CODEC_VOCAB_SIZE) -- only the real codec rows are used for the
        # output logits (the extra pad row at AUDIO_PAD_ID is excluded).
        audio_logits = F.linear(hidden, self.audio_embedding.weight[:CODEC_VOCAB_SIZE])
        # (B, T, 1) -- eos_embedding is (d_model,)
        eos_logits = F.linear(hidden, self.eos_embedding.unsqueeze(0))
        return torch.cat([audio_logits, eos_logits], dim=-1)

    # ------------------------------------------------------------------
    # Training forward
    # ------------------------------------------------------------------
    def forward(
        self,
        text_tokens: torch.Tensor,   # (B, T_text)   int64, values in [0, 128)
        audio_tokens: torch.Tensor,  # (B, T_audio)  int64, values in [0, 2048]
        key_padding_mask: Optional[torch.Tensor] = None,  # (B, T_total) bool, True = pad
    ) -> torch.Tensor:
        """Full-sequence forward pass for training.

        Builds the sequence ``[text, <SEP>, audio]``, applies causal self-
        attention, and returns logits for every position.

        Parameters
        ----------
        text_tokens, audio_tokens
            Right-padded token id tensors.  Text padding uses ``TEXT_PAD_ID``
            (0); audio padding uses ``AUDIO_PAD_ID`` (= ``CODEC_VOCAB_SIZE``),
            which indexes the extra learned pad row of ``audio_embedding``.
        key_padding_mask
            Optional boolean tensor of shape ``(B, T_text + 1 + T_audio)``
            where ``True`` marks positions to be **ignored** by attention
            (padded text / padded audio).  The SEP position must always be
            ``False``.  When ``None`` no key padding is assumed (all tokens
            are real).

        Returns
        -------
        torch.Tensor
            ``(B, T_text + 1 + T_audio, OUTPUT_VOCAB_SIZE)`` logits.
        """
        B, T_text = text_tokens.shape
        T_audio = audio_tokens.shape[1]
        device = text_tokens.device

        # --- text: embedding + text positions 0..T_text-1 -----------------
        text_pos = torch.arange(T_text, device=device)
        text_emb = self.text_embedding(text_tokens) + self.text_pos_embedding(text_pos)

        # --- <SEP>: learned vector + audio position 0 ----------------------
        sep_pos_emb = self.audio_pos_embedding.weight[0]          # (d_model,)
        sep_emb = (self.sep_embedding + sep_pos_emb)              # (d_model,)
        sep_emb = sep_emb.view(1, 1, -1).expand(B, 1, -1)         # (B, 1, d)

        # --- audio: embedding + audio positions 1..T_audio ----------------
        audio_pos = torch.arange(1, T_audio + 1, device=device)
        audio_emb = self.audio_embedding(audio_tokens) + self.audio_pos_embedding(audio_pos)

        # --- concatenate and dropout ---------------------------------------
        x = torch.cat([text_emb, sep_emb, audio_emb], dim=1)      # (B, T_total, d)
        x = self.emb_drop(x)

        # --- attention mask -------------------------------------------------
        # Combine the causal (lower-triangular) mask with an optional
        # key-padding mask into a single additive mask passed to SDPA with
        # ``is_causal=False`` (PyTorch does not allow combining
        # ``is_causal=True`` with an explicit ``attn_mask``).  A large
        # finite negative (instead of -inf) is used so that a row whose keys
        # are *all* padded never produces a NaN softmax (its loss is masked
        # anyway).
        attn_mask = None
        if key_padding_mask is not None:
            T_total = x.shape[1]
            # A large finite negative (instead of -inf) so that even a row
            # whose keys are *all* masked never produces a NaN softmax (its
            # loss is masked anyway) and so that adding two masked entries
            # never overflows to -inf across fp16/bf16/fp32.
            neg = -1e4
            # Causal additive mask (1, 1, T, T): 0 on/below diagonal, neg above.
            causal = torch.triu(
                torch.full((T_total, T_total), neg, device=device, dtype=x.dtype),
                diagonal=1,
            )
            # Key-padding additive mask (B, 1, 1, T): neg where padded.
            pad = torch.zeros(B, 1, 1, T_total, device=device, dtype=x.dtype)
            pad = pad.masked_fill(key_padding_mask.view(B, 1, 1, T_total), neg)
            attn_mask = causal.view(1, 1, T_total, T_total) + pad  # (B, 1, T, T)

        # --- transformer blocks --------------------------------------------
        for block in self.blocks:
            x = block(x, use_causal_mask=(attn_mask is None), attn_mask=attn_mask)

        x = self.ln_f(x)
        return self._output_logits(x)

    # ------------------------------------------------------------------
    # Incremental forward (with KV cache)
    # ------------------------------------------------------------------
    def forward_step(
        self,
        token_ids: torch.Tensor,     # (B, T)
        token_types: torch.Tensor,   # (B, T)   values from TokenType
        positions: torch.Tensor,     # (B, T)   index into respective pos embedding
        kv_cache: Optional[KVCache] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """Process one or more tokens with an optional KV cache.

        Two modes:

        * **Prefill** – ``kv_cache`` is ``None`` or empty.  *T* may be > 1
          and a causal mask is applied so tokens attend only to preceding
          positions.  Typically called once with the full prompt
          ``[text … <SEP> … audio_prefix]``.
        * **Decode** – ``kv_cache`` already holds tokens.  *T* should be 1.
          No causal mask is needed (the lone query may attend to every
          cached key).

        The caller manages ``token_types`` (which embedding table to use)
        and ``positions`` (index into the *respective* positional embedding
        table – text positions for text tokens, audio positions for
        audio / SEP / EOS tokens).

        Returns
        -------
        logits : torch.Tensor
            ``(B, T, OUTPUT_VOCAB_SIZE)``.
        kv_cache : KVCache
            Updated cache (same object if one was passed in, else a new one).
        """
        B, T = token_ids.shape

        # --- embeddings ----------------------------------------------------
        x = self._embed_tokens(token_ids, token_types, positions)
        x = self.emb_drop(x)

        # --- determine mode ------------------------------------------------
        is_prefill = kv_cache is None or kv_cache.sequence_length == 0
        if kv_cache is None:
            kv_cache = KVCache(self.n_layers)

        if not is_prefill and T > 1:
            warnings.warn(
                "forward_step called with T > 1 during decode (cache not "
                "empty): new tokens will attend to each other non‑causally. "
                "Pass tokens one at a time after the prefill.",
                stacklevel=2,
            )

        # --- transformer blocks --------------------------------------------
        for i, block in enumerate(self.blocks):
            x = block(
                x,
                kv_cache=kv_cache,
                layer_idx=i,
                use_causal_mask=is_prefill,
            )

        x = self.ln_f(x)
        logits = self._output_logits(x)
        return logits, kv_cache

    # ------------------------------------------------------------------
    # Embedding helper for forward_step
    # ------------------------------------------------------------------
    def _embed_tokens(
        self,
        token_ids: torch.Tensor,     # (B, T)
        token_types: torch.Tensor,   # (B, T)
        positions: torch.Tensor,     # (B, T)
    ) -> torch.Tensor:
        """Look up input + positional embeddings for a mixed batch of tokens.

        For each position exactly one embedding is selected based on
        ``token_types``:

        ==================  ===========================  ==========================
        token_type          input embedding              positional embedding
        ==================  ===========================  ==========================
        ``TokenType.TEXT``  ``text_embedding[token_id]``  ``text_pos_embedding[pos]``
        ``TokenType.AUDIO`` ``audio_embedding[token_id]`` ``audio_pos_embedding[pos]``
        ``TokenType.SEP``   ``sep_embedding``             ``audio_pos_embedding[pos]``
        ``TokenType.EOS``   ``eos_embedding``             ``audio_pos_embedding[pos]``
        ==================  ===========================  ==========================

        Indices are clamped to valid ranges as a safety measure.
        """
        B, T = token_ids.shape
        device = token_ids.device
        dtype = self.audio_embedding.weight.dtype

        x = torch.zeros(B, T, self.d_model, device=device, dtype=dtype)

        # --- text ----------------------------------------------------------
        is_text = token_types == TokenType.TEXT
        if is_text.any():
            ids = token_ids.clamp(0, TEXT_VOCAB_SIZE - 1)
            pos = positions.clamp(0, MAX_TEXT_LENGTH - 1)
            emb = self.text_embedding(ids) + self.text_pos_embedding(pos)
            x = x + emb * is_text.unsqueeze(-1).to(dtype)

        # --- audio ---------------------------------------------------------
        is_audio = token_types == TokenType.AUDIO
        if is_audio.any():
            ids = token_ids.clamp(0, CODEC_VOCAB_SIZE - 1)
            pos = positions.clamp(0, MAX_AUDIO_LENGTH)
            emb = self.audio_embedding(ids) + self.audio_pos_embedding(pos)
            x = x + emb * is_audio.unsqueeze(-1).to(dtype)

        # --- <SEP> ---------------------------------------------------------
        is_sep = token_types == TokenType.SEP
        if is_sep.any():
            pos = positions.clamp(0, MAX_AUDIO_LENGTH)
            emb = self.sep_embedding.view(1, 1, -1) + self.audio_pos_embedding(pos)
            x = x + emb * is_sep.unsqueeze(-1).to(dtype)

        # --- <EOS> ---------------------------------------------------------
        is_eos = token_types == TokenType.EOS
        if is_eos.any():
            pos = positions.clamp(0, MAX_AUDIO_LENGTH)
            emb = self.eos_embedding.view(1, 1, -1) + self.audio_pos_embedding(pos)
            x = x + emb * is_eos.unsqueeze(-1).to(dtype)

        return x

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count parameters, optionally excluding all embedding‑like tensors."""
        if not exclude_embeddings:
            return sum(p.numel() for p in self.parameters())

        excluded = {
            id(self.text_embedding.weight),
            id(self.audio_embedding.weight),
            id(self.sep_embedding),
            id(self.eos_embedding),
            id(self.text_pos_embedding.weight),
            id(self.audio_pos_embedding.weight),
        }
        total = 0
        for p in self.parameters():
            if id(p) not in excluded:
                total += p.numel()
        return total
