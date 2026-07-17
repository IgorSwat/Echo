"""Audio encoder for the Echo text-to-speech model.

Pipeline
--------
1. ``Embedding``        : (B, 16, T) RVQ token grid -> (B, T, d_e) latents.
2. ``Positional``       : learned sinusoidal-style embeddings added per time step.
3. ``TransformerEncoder`` : bidirectional self-attention over time (PyTorch built-in).
Output: (B, T, d_f) latent, with d_f == d_e by default.

All hyperparameters live in ``echo.config.AudioEncoderConfig``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from echo.config import AudioEncoderConfig
from echo.codec_embedding import CodecEmbedding


# =============
# Audio Encoder
# =============

class AudioEncoder(nn.Module):
    """
    Bidirectional transformer encoder over discrete RVQ audio tokens.

    Args:
        config: ``AudioEncoderConfig`` instance (see echo.config).
        codec_embedding: optional shared ``CodecEmbedding``; when provided it
            is reused instead of allocating a new table (e.g. shared with the
            decoder executor).
    """

    def __init__(
        self,
        config: AudioEncoderConfig,
        codec_embedding: CodecEmbedding | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        # --- 1. Codebook embedding -----------------------------------------
        if codec_embedding is not None:
            self.embedding = codec_embedding
        else:
            self.embedding = CodecEmbedding(
                vocab_size=config.vocab_size,
                num_codebooks=config.num_codebooks,
                embedding_dim=config.embedding_dim,
                mask_token_id=config.mask_token_id,
            )
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        # --- 2. Learned positional encoding --------------------------------
        self.pos_embed = nn.Embedding(config.max_audio_length, config.d_f)

        # --- 3. Bidirectional transformer encoder -------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_f,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_f),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a discrete RVQ token grid into continuous latents.

        Args:
            tokens: (B, num_codebooks, T) integer token ids.
            key_padding_mask: (B, T) boolean, True = padded/ignored time step.
        Returns:
            (B, T, d_f) latent representation.
        """

        # ---- Step 1: Codebook embedding + dropout -------------------------
        # CodecEmbedding returns (B, C, T, d_e); sum across codebooks -> (B, T, d_e)
        x = self.embedding(tokens).sum(dim=1)
        x = self.embedding_dropout(x)

        # ---- Step 2: Learned positional encoding --------------------------
        T = x.shape[1]
        positions = torch.arange(T, device=x.device).clamp(max=self.config.max_audio_length - 1)
        x = x + self.pos_embed(positions)                # (B, T, d_f)

        # ---- Step 3: Bidirectional transformer encoder --------------------
        # nn.TransformerEncoder expects ``src_key_padding_mask`` of shape
        # (B, T) where ``True`` means **ignore** — same convention we use.
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)  # (B, T, d_f)

        return x