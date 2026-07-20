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
        max_seq_len: int = config.MAX_SEQ_LEN,
    ) -> None:
        super().__init__()

        # Reusable size information
        self.d_emb = d_emb
        self.d_model = d_model
        self.d_repr = d_repr
        self.num_pred_heads = num_pred_heads

        # Embeddings
        # - Text embeddings - simple Embedding table (no positional).
        # - Audio (codec) embeddings - specialized per-codebook embeddings (no positional).
        # - Special token embeddings - one learnable vector per special token, hardcoded here.
        # - Joint learned positional embeddings applied to the entire sequence.
        self.text_embed = TextEmbedding(text_vocab_size, text_emb_dim)
        self.codec_embed = CodecEmbedding()

        self.bos_embed = nn.Parameter(torch.empty(d_emb))
        self.ref_text_eos_embed = nn.Parameter(torch.empty(d_emb))
        self.ref_codec_eos_embed = nn.Parameter(torch.empty(d_emb))
        self.text_eos_embed = nn.Parameter(torch.empty(d_emb))

        self.pos_embed = nn.Embedding(max_seq_len, d_emb)

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
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
        audio_codec: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Concatenate embeddings for the provided modalities.

        * with audio → ``[E(<BOS>), E(ref_text), E(<REF_TEXT_EOS>), E(ref_audio_codec), E(<REF_CODEC_EOS>), E(text), E(<TEXT_EOS>), E(audio_codec)]``
        * without audio → ``[E(<BOS>), E(ref_text), E(<REF_TEXT_EOS>), E(ref_audio_codec), E(<REF_CODEC_EOS>), E(text), E(<TEXT_EOS>)]``

        Output: ``(B, total_len, d_emb)``
        """
        B = ref_text.size(0)

        parts: list[torch.Tensor] = []

        # <BOS>
        parts.append(self.bos_embed.view(1, 1, -1).expand(B, 1, self.d_emb))

        # Reference text + <REF_TEXT_EOS>
        parts.append(self.text_embed(ref_text))
        parts.append(self.ref_text_eos_embed.view(1, 1, -1).expand(B, 1, self.d_emb))

        # Reference audio codec + <REF_CODEC_EOS>
        parts.append(self.codec_embed(ref_audio_codec))
        parts.append(self.ref_codec_eos_embed.view(1, 1, -1).expand(B, 1, self.d_emb))

        # Target text + <TEXT_EOS>
        parts.append(self.text_embed(text))
        parts.append(self.text_eos_embed.view(1, 1, -1).expand(B, 1, self.d_emb))

        # Optional target audio codec (teacher forcing).
        if audio_codec is not None:
            parts.append(self.codec_embed(audio_codec))

        x = torch.cat(parts, dim=1)

        # Joint learned positional embeddings across the whole sequence.
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions)

        return x

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
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
        audio_codec: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass strictly for training.
        """

        T_audio = audio_codec.size(1)

        x = self._embed(ref_text, ref_audio_codec, text, audio_codec)
        hidden, _ = self._run_transformer(x, kv_cache=None, audio_len=T_audio + 1)  # +1 to include the <TEXT_EOS> position
        logits = self._predict(hidden)  # (B, T_audio + 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM)

        return logits

    def prefill(
        self,
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
    ):
        """
        An initial forward pass for inference - prefilling the reference prompt
        (ref_text, ref_audio_codec) and the target text. The next position will
        predict the first audio frame.
        """

        x = self._embed(ref_text, ref_audio_codec, text)  # audio_codec=None
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
        Single-step forward pass using KV cache. Embeds only the provided codebook
        frame and applies the joint positional embedding for the given position.
        """

        frame_emb = self.codec_embed(codebook.unsqueeze(1))           # (B, 1, D)
        pos = torch.tensor([position], device=frame_emb.device)
        frame_emb = frame_emb + self.pos_embed(pos)                   # (1, D) broadcasts over (B, 1, D)

        hidden, new_cache = self._run_transformer(frame_emb, kv_cache, audio_len=1)
        next_logits = self._predict(hidden)

        return next_logits, new_cache

    # ------------------------------------------------------------------
    # Convenience: greedy generation loop (EOS checked on codebook 0)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
        max_steps: int = config.MAX_AUDIO_LENGTH,
        min_steps: int = 0,
        temperature: float = 1.0,
        eos_id: int = config.EOS_ID,
        pad_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative decoding for a single sequence.

        Returns (frames, lengths) where frames is (1, T, NUM_CODEBOOKS) and lengths is tensor([T]).
        """

        self.eval()

        greedy = temperature < 1e-5

        # Prefill consumes the prompt + target text (no audio_codec).
        logits, kv_cache = self.prefill(ref_text, ref_audio_codec, text)

        # Number of tokens already in the KV cache = prompt + text + special tokens.
        # The next predicted frame will be placed at this position.
        start_pos = (
            1                              # <BOS>
            + ref_text.size(1)
            + 1                            # <REF_TEXT_EOS>
            + ref_audio_codec.size(1)
            + 1                            # <REF_CODEC_EOS>
            + text.size(1)
            + 1                            # <TEXT_EOS>
        )

        frames: list[torch.Tensor] = []
        for i in range(max_steps):
            frame_logits = logits[0, 0]                    # (NUM_CODEBOOKS, VOCAB+1)
            if greedy:
                frame = frame_logits.argmax(dim=-1)        # (NUM_CODEBOOKS,)
            else:
                probs = torch.softmax(frame_logits / temperature, dim=-1)
                frame = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if i >= min_steps and frame[0].item() == eos_id:
                break

            frames.append(frame)

            feed = frame.clone()
            feed[feed == eos_id] = pad_id
            logits, kv_cache = self.step(feed.unsqueeze(0), position=start_pos + i, kv_cache=kv_cache)

        if not frames:
            return torch.zeros(1, 0, self.num_pred_heads, dtype=torch.long), torch.tensor([0])

        return torch.stack(frames, dim=0).unsqueeze(0), torch.tensor([len(frames)])  # (1, T, NB), (1,)
