"""Decoder planner and executor for the Echo text-to-speech model."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from echo.config import DecoderPlannerConfig
from echo.config import DecoderExecutorConfig
from echo.codec_embedding import CodecEmbedding


# =====================
# Decoder planner layer
# =====================

class _DecoderPlannerLayer(nn.Module):
    """
    Pre-norm transformer decoder layer. Self-attention and FFN operate in
    ``d_plan``; cross-attention projects to ``d_hid`` to attend to the
    concatenated text+audio memory, then projects back to ``d_plan``.
    """

    def __init__(
        self,
        d_plan: int,
        d_hid: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()

        # --- 1. Self-attention (d_plan) -----------------------------------
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_plan,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_plan)

        # --- 2. Cross-attention (d_hid) -----------------------------------
        self.proj_down = nn.Linear(d_plan, d_hid, bias=False)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_hid,
            num_heads=num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.proj_up = nn.Linear(d_hid, d_plan, bias=False)
        self.norm2 = nn.LayerNorm(d_plan)

        # --- 3. Feed-forward (d_plan) -------------------------------------
        self.ffn = nn.Sequential(
            nn.Linear(d_plan, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_plan),
        )
        self.norm3 = nn.LayerNorm(d_plan)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # h:      (B, T_plan, d_plan)
        # memory: (B, T_text + T_audio, d_hid)

        # --- 1. Self-attention --------------------------------------------
        residual = h
        h_norm = self.norm1(h)
        sa_out, _ = self.self_attn(
            h_norm, h_norm, h_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        h = residual + self.dropout(sa_out)

        # --- 2. Cross-attention (d_plan -> d_hid -> d_plan) ---------------
        residual = h
        h_norm = self.norm2(h)
        q = self.proj_down(h_norm)          # (B, T_plan, d_hid)
        ca_out, _ = self.cross_attn(
            q, memory, memory,
            key_padding_mask=memory_key_padding_mask,
        )
        ca_out = self.proj_up(ca_out)        # (B, T_plan, d_plan)
        h = residual + self.dropout(ca_out)

        # --- 3. Feed-forward ----------------------------------------------
        residual = h
        h_norm = self.norm3(h)
        h = residual + self.dropout(self.ffn(h_norm))

        return h


# ===============
# Decoder planner
# ===============

class DecoderPlanner(nn.Module):
    """
    Transformer decoder with hidden state ``d_plan`` that cross-attends to
    concatenated text+audio encoder outputs (both at ``d_hid``).

    Input:  H (B, T_plan, d_plan), text (B, T_text, d_hid), audio (B, T_audio, d_hid)
    Output: (B, T_plan, d_plan)
    """

    def __init__(self, config: DecoderPlannerConfig) -> None:
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList([
            _DecoderPlannerLayer(
                d_plan=config.d_plan,
                d_hid=config.d_hid,
                num_heads=config.num_heads,
                ffn_dim=config.ffn_dim,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.d_plan)

    def forward(
        self,
        H: torch.Tensor,
        text_enc: torch.Tensor,
        audio_enc: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        text_key_padding_mask: Optional[torch.Tensor] = None,
        audio_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Build cross-attention memory: concatenate text and audio encodings
        # along the sequence dimension -> (B, T_text + T_audio, d_hid)
        memory = torch.cat([text_enc, audio_enc], dim=1)

        # Combine the padding masks for text and audio if provided.
        if text_key_padding_mask is not None or audio_key_padding_mask is not None:
            B = memory.shape[0]
            T_text = text_enc.shape[1]
            T_audio = audio_enc.shape[1]
            text_mask = text_key_padding_mask if text_key_padding_mask is not None \
                else torch.zeros(B, T_text, dtype=torch.bool, device=memory.device)
            audio_mask = audio_key_padding_mask if audio_key_padding_mask is not None \
                else torch.zeros(B, T_audio, dtype=torch.bool, device=memory.device)
            memory_key_padding_mask = torch.cat([text_mask, audio_mask], dim=1)
        else:
            memory_key_padding_mask = None

        # Run decoder layers.
        h = H
        for layer in self.layers:
            h = layer(
                h, memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_mask=tgt_mask,
            )

        h = self.norm(h)
        return h
    

# ===========================
# Decoder planner - stop head
# ===========================

class StopHead(nn.Module):
    """
    Binary stop classifier operating on planner hidden states.

    Input:  h (B, T_plan, d_plan) or (B, d_plan)
    Output: logits (B, T_plan, 1) or (B, 1) — one stop logit per step.
    """

    def __init__(self, d_plan: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_plan)
        self.linear = nn.Linear(d_plan, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(h)).squeeze(-1)


# ================
# Decoder executor
# ================

class DecoderExecutor(nn.Module):
    """
    Non-autoregressive, MLM-style codec grid filler.

    Given a planner hidden vector and a ``(context_steps + chunk_size) x C``
    token grid (first rows filled, remaining rows a mix of real tokens and
    MASK), predicts logits for the chunk rows over the real vocab.

    Input:  planner_state (B, d_plan),
            grid_tokens   (B, context_steps + chunk_size, C)
    Output: logits (B, chunk_size, C, vocab_size)
    """

    def __init__(
        self,
        config: DecoderExecutorConfig,
        codec_embedding: CodecEmbedding,
    ) -> None:
        super().__init__()
        self.config = config

        C = config.num_codebooks
        V = config.vocab_size
        CS = config.chunk_size
        CTX = config.context_steps
        T = CS + CTX
        d = config.d_exec

        self.num_codebooks = C
        self.vocab_size = V
        self.chunk_size = CS
        self.context_steps = CTX

        self.token_embedding = codec_embedding

        self.proj_codec = (
            nn.Linear(codec_embedding.embedding_dim, d, bias=False)
            if codec_embedding.embedding_dim != d else None
        )
        self.proj_planner = nn.Linear(config.d_plan, d, bias=False)

        self.pos_embedding = nn.Parameter(torch.zeros(T, C, d))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d,
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
            norm=nn.LayerNorm(d),
            enable_nested_tensor=False,
        )

        # Seperate head per codec layer
        self.heads = nn.ModuleList([nn.Linear(d, V) for _ in range(C)])

    def forward(
        self,
        planner_state: torch.Tensor,
        grid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        B = planner_state.shape[0]
        C = self.num_codebooks
        V = self.vocab_size
        CS = self.chunk_size
        CTX = self.context_steps
        T = CS + CTX
        d = self.config.d_exec

        cond = self.proj_planner(planner_state).unsqueeze(1)

        grid_emb = self.token_embedding(grid_tokens, codebook_dim=2)
        grid_emb = grid_emb.reshape(B, T * C, grid_emb.shape[-1])
        if self.proj_codec is not None:
            grid_emb = self.proj_codec(grid_emb)

        grid_emb = grid_emb + self.pos_embedding.reshape(1, T * C, d)

        seq = torch.cat([cond, grid_emb], dim=1)
        out = self.transformer(seq)

        out_chunk = out[:, 1 + CTX * C:, :].reshape(B, CS, C, d)

        logits = torch.empty(B, CS, C, V, device=out.device, dtype=out.dtype)
        for c, head in enumerate(self.heads):
            logits[:, :, c, :] = head(out_chunk[:, :, c, :])

        return logits
