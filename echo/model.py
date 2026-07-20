from __future__ import annotations

from echo import config
from echo.modules.codec_embedding import CodecEmbedding
from echo.modules.transformer import TransformerDecoder
from echo.modules.prediction_heads import PredictionMultihead, FusedPredictionMultihead
from echo.modules.text_embedding import TextEmbedding
from echo.modules.types import KVCache

from typing import Optional

import torch
import torch.nn as nn


class Echo(nn.Module):

    def __init__(
        self,
        text_vocab_size: int = config.TEXT_VOCAB_SIZE,
        text_emb_dim: int = config.TEXT_EMB_DIM,
        text_pos_size: int = config.TEXT_POS_SIZE,
        d_emb: int = config.D_EMB,
        d_model: int = config.D_MODEL,
        d_repr: int = config.D_REPR,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        ffn_dim: int = config.FFN_DIM,
        dropout: float = config.DROPOUT,
        num_pred_heads: int = config.PRED_NUM_HEADS,
        pred_hidden_dim: int = config.PRED_HIDDEN_DIM,
        codec_logit_dim: int = config.CODEC_LOGIT_DIM,
        pred_num_layers: int = config.PRED_NUM_LAYERS,
        pred_dropout: float = config.PRED_DROPOUT,
    ) -> None:
        super().__init__()

        # Reusable size information
        self.d_emb = d_emb
        self.d_model = d_model
        self.d_repr = d_repr
        self.num_pred_heads = num_pred_heads

        # Embeddings
        # - Text embeddings - simple Embedding table + learnable positional embeddings
        # - Audio (codec) embeddings - specialized per-codebook embeddings + learnable positional embeddings
        # - Sep embedding - a single special latent to distinguish text and audio embedding sequence
        self.text_embed = TextEmbedding(text_vocab_size, text_emb_dim, text_pos_size)
        self.sep_embed = nn.Embedding(1, d_emb)
        self.codec_embed = CodecEmbedding()

        # Dimension adapters
        # Since transformer can operate on different internal dimension than embeddings,
        # we use linear projections to match them (or nn.Identity if already matched).
        self.input_proj = nn.Linear(d_emb, d_model) if d_emb != d_model else nn.Identity()
        self.output_proj = nn.Linear(d_model, d_repr) if d_model != d_repr else nn.Identity()

        # Transformer decoder
        # The heart of the model.
        self.transformer = TransformerDecoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # Prediction heads
        self.heads = FusedPredictionMultihead(
            num_heads=num_pred_heads,
            in_dim=d_repr,
            hidden_dim=pred_hidden_dim,
            out_dim=codec_logit_dim,
            num_layers=pred_num_layers,
            dropout=pred_dropout,
        )

    def _embed(
        self,
        text: torch.Tensor | None = None,
        audio_codec: torch.Tensor | None = None,
        codec_position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Concatenate embeddings for the provided modalities.

        * text + audio → ``[E_text, E_sep, E_codec]``
        * text only     → ``[E_text]``
        * audio only    → ``[E_codec]``

        Output: ``(B, total_len, d_emb)``
        """
        parts: list[torch.Tensor] = []

        # First text embeddings
        if text is not None:
            parts.append(self.text_embed(text))

        # Separator embedding (only if text is also present)
        if text is not None and audio_codec is not None:
            B = text.size(0)
            sep = self.sep_embed.weight.view(1, 1, -1).expand(B, 1, self.d_emb)
            parts.append(sep)

        # Audio codec embeddings
        if audio_codec is not None and audio_codec.numel() > 0:
            parts.append(self.codec_embed(audio_codec, position_offset=codec_position_offset))

        if not parts:
            raise ValueError("_embed: at least one of text or audio_codec must be provided")

        return torch.cat(parts, dim=1)

    def _run_transformer(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache],
        audio_len: int,
    ) -> tuple[torch.Tensor, KVCache]:
        """
        Project → transformer → project, then slice the output.

        * audio_len > 0 — last ``audio_len`` positions (teacher-forcing)
        * audio_len == 0 — last position only

        Output: ``(B, out_len, d_repr)``
        """

        # Projections & main transformer call
        x = self.input_proj(x)
        hidden, kv_cache = self.transformer(x, kv_cache)
        hidden = self.output_proj(hidden)

        # Cut the logits to only return the ones for audio codec (since we do not want to generate text).
        if audio_len > 0:
            hidden = hidden[:, -audio_len:]
        else:
            hidden = hidden[:, -1:]

        return hidden, kv_cache

    def _predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Runs prediction heads.

        Output: ``(B, ..., NUM_CODEBOOKS, CODEC_LOGIT_DIM)``
        """
        
        return self.heads(hidden)

    
    def forward(
        self, 
        text: torch.Tensor, 
        audio_codec: torch.Tensor
    ) -> torch.Tensor:
        """
        A forward pass strictly for training.
        """

        T_audio = audio_codec.size(1)

        x = self._embed(text, audio_codec)

        # Standard forward pass 
        hidden, _ = self._run_transformer(
            x,
            kv_cache=None,
            audio_len=T_audio,
        )

        return self._predict(hidden)  # (B, T_audio, NUM_CODEBOOKS, CODEC_LOGIT_DIM)

    def prefill(
        self, text: 
        torch.Tensor, 
        audio_codec: Optional[torch.Tensor] = None
    ):
        """
        An initial forward pass for inference - prefilling entire text and reference audio tokens.
        """

        x = self._embed(text, audio_codec)
        hidden, kv_cache = self._run_transformer(x, kv_cache=None, audio_len=1)
        next_logits = self._predict(hidden)

        return next_logits, kv_cache

    def step(
        self,
        codebook: torch.Tensor,
        position: int,
        kv_cache: KVCache,
    ):
        """
        Single-step forward pass using KV cache.
        """

        frame_emb = self.codec_embed(codebook.unsqueeze(1), position_offset=position)
        hidden, new_cache = self._run_transformer(frame_emb, kv_cache, audio_len=1)
        next_logits = self._predict(hidden)

        return next_logits, new_cache

    # ------------------------------------------------------------------
    # Convenience: greedy generation loop (EOS checked on codebook 0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,
        audio_codec: Optional[torch.Tensor] = None,
        max_steps: int = config.MAX_AUDIO_LENGTH,
        temperature: float = 1.0,
        eos_id: int = config.CODEC_EOS_ID,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative decoding for a single sequence.

        Returns (frames, lengths) where frame is (1, T, NUM_CODEBOOKS) and lengths is tensor([T]).
        """
        self.eval()

        T_audio = audio_codec.size(1) if audio_codec is not None else 0
        greedy = temperature < 1e-5

        logits, kv_cache = self.prefill(text, audio_codec)
        start_pos = T_audio

        frames: list[torch.Tensor] = []
        for i in range(max_steps):
            frame_logits = logits[0, 0]                    # (NUM_CODEBOOKS, VOCAB+1)
            if greedy:
                frame = frame_logits.argmax(dim=-1)        # (NUM_CODEBOOKS,)
            else:
                probs = torch.softmax(frame_logits / temperature, dim=-1)
                frame = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if frame[0].item() == eos_id:
                break

            frames.append(frame)

            feed = frame.clone()
            feed[feed == eos_id] = pad_id
            logits, kv_cache = self.step(feed.unsqueeze(0), position=start_pos + i, kv_cache=kv_cache)

        if not frames:
            return torch.zeros(1, 0, self.num_pred_heads, dtype=torch.long), torch.tensor([0])
        
        return torch.stack(frames, dim=0).unsqueeze(0), torch.tensor([len(frames)])  # (1, T, NB), (1,)
