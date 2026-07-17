"""Shared codec embedding table for RVQ audio tokens.

A single ``nn.Embedding`` table holds ``num_codebooks`` disjoint slices (one
per codec layer), addressed via per-codebook offsets. The MASK token
(``mask_token_id = vocab_size``, i.e. 2048 when ``vocab_size = 2048``) gets
its own embedding inside every codebook slice, so the table holds
``num_codebooks * (vocab_size + 1)`` rows.

This module only performs the embedding lookup — summing across codebooks
(if desired) is the caller's responsibility (see ``AudioEncoder``).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CodecEmbedding(nn.Module):
    """
    Shared, per-codebook embedding table for RVQ codec tokens (and MASK).

    Args:
        vocab_size:     number of real token values per codebook (2048).
        num_codebooks:  number of codec layers (16).
        embedding_dim:  per-cell embedding dimension ``d``.
        mask_token_id:  id reserved for the MASK token; defaults to
                        ``vocab_size`` (i.e. 2048), extending the per-codebook
                        table to ``vocab_size + 1`` rows.

    Forward:
        tokens: integer ids of shape ``(..., C, ...)`` where the codebook
                axis is at position ``codebook_dim`` (default 1). Token values
                are in ``[0, vocab_size]`` (inclusive of the MASK id).
        Returns: embeddings of shape ``(..., C, ..., d)`` — the input shape
                 with a trailing ``d`` axis appended. No summation is done.
    """

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        embedding_dim: int,
        mask_token_id: int | None = None,
    ) -> None:
        super().__init__()
        # The MASK token extends the per-codebook vocabulary by one.
        self.mask_token_id = mask_token_id if mask_token_id is not None else vocab_size
        self.per_codebook_size = self.mask_token_id + 1   # vocab_size + 1 by default
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(num_codebooks * self.per_codebook_size, embedding_dim)

    def forward(self, tokens: torch.Tensor, codebook_dim: int = 1) -> torch.Tensor:
        # Build per-codebook offsets broadcastable along ``codebook_dim``.
        offsets = torch.arange(self.num_codebooks, device=tokens.device) * self.per_codebook_size
        shape = [1] * tokens.dim()
        shape[codebook_dim] = self.num_codebooks
        tokens = tokens + offsets.view(shape)

        emb = self.embedding(tokens)   # (..., C, ..., d)
        return emb
