from __future__ import annotations

from echo import config
from echo.modules.codec_embedding import CodecEmbedding
from echo.modules.transformer import TransformerDecoder
from echo.modules.prediction_heads import PredictionMultihead, FusedPredictionMultihead
from echo.modules.text_embedding import TextEmbedding
from echo.modules.types import KVCache

from typing import Optional
import math

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
        self.token_embedding_dim = config.CODEC_TOKEN_EMB_DIM

        # Embeddings
        # - Text embeddings - simple Embedding table (no positional).
        # - Audio (codec) embeddings - specialized per-codebook embeddings (no positional).
        # - Special token embeddings - one learnable vector per special token, hardcoded here.
        # - Joint learned positional embeddings applied to the entire sequence.
        self.text_embed = TextEmbedding(text_vocab_size, text_emb_dim)
        self.codec_embed = CodecEmbedding(
            vocab_size=codec_logit_dim - 1,
            num_codebook_layers=num_pred_heads,
            token_embedding_dim=config.CODEC_TOKEN_EMB_DIM,
            codebook_embedding_dim=config.CODEC_EMB_DIM,
            mlp_hidden_dim=config.CODEC_MLP_HIDDEN_DIM,
            mlp_num_layers=config.CODEC_MLP_NUM_LAYERS,
            mlp_dropout=config.CODEC_MLP_DROPOUT,
        )

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
        self.codec_condition_proj = nn.Linear(self.token_embedding_dim, d_repr, bias=False)

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

        self._init_weights()

    def _init_weights(self) -> None:
        for p in (self.bos_embed, self.ref_text_eos_embed, self.ref_codec_eos_embed, self.text_eos_embed):
            nn.init.normal_(p, mean=0.0, std=config.INIT_STD)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=config.INIT_STD)
        nn.init.normal_(self.codec_condition_proj.weight, mean=0.0, std=config.INIT_STD)

    def _embed_prompt(
        self,
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
    ) -> torch.Tensor:
        """Embed an unpadded inference prompt."""
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

        x = torch.cat(parts, dim=1)

        # Joint learned positional embeddings across the whole sequence.
        T = x.size(1)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_embed(positions)

        return x

    def _embed_training_batch(
        self,
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
        audio_codec: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Assemble complete per-sample sequences and right-pad the result."""
        B = text.size(0)
        ref_text_emb = self.text_embed(ref_text)
        ref_audio_emb = self.codec_embed(ref_audio_codec)
        text_emb = self.text_embed(text)
        audio_emb = self.codec_embed(audio_codec)

        sequences: list[torch.Tensor] = []
        prediction_starts: list[int] = []
        sequence_lengths: list[int] = []
        for i in range(B):
            text_len = int(text_lengths[i].item())
            audio_len = int(audio_lengths[i].item())
            parts = [
                self.bos_embed.view(1, -1),
                ref_text_emb[i],
                self.ref_text_eos_embed.view(1, -1),
                ref_audio_emb[i],
                self.ref_codec_eos_embed.view(1, -1),
                text_emb[i, :text_len],
                self.text_eos_embed.view(1, -1),
                audio_emb[i, :audio_len],
            ]
            sequence = torch.cat(parts, dim=0)
            prediction_starts.append(sequence.size(0) - audio_len - 1)
            sequence_lengths.append(sequence.size(0))
            sequences.append(sequence)

        max_len = max(sequence_lengths)
        if max_len > self.pos_embed.num_embeddings:
            raise ValueError(f"sequence length {max_len} exceeds maximum {self.pos_embed.num_embeddings}")

        x = sequences[0].new_zeros((B, max_len, self.d_emb))
        valid_mask = torch.zeros((B, max_len), dtype=torch.bool, device=x.device)
        for i, sequence in enumerate(sequences):
            length = sequence.size(0)
            x[i, :length] = sequence
            valid_mask[i, :length] = True

        positions = torch.arange(max_len, device=x.device).view(1, -1).expand(B, -1)
        x = x + self.pos_embed(positions)
        x = x.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return x, valid_mask, torch.tensor(prediction_starts, device=x.device)

    def _run_transformer(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache],
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, KVCache]:
        x = self.input_proj(x)
        hidden, kv_cache = self.transformer(x, kv_cache, key_padding_mask)
        hidden = self.output_proj(hidden)
        return hidden, kv_cache

    def _predict(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """
        Runs prediction heads.

        Output: ``(B, ..., NUM_CODEBOOKS, CODEC_LOGIT_DIM)``
        """

        codebook_embeddings = self.codec_embed.embed_codebooks(codes)
        previous = codebook_embeddings.cumsum(dim=-2) - codebook_embeddings
        counts = torch.arange(self.num_pred_heads, device=hidden.device).clamp(min=1).sqrt()
        previous = previous / counts.view(*([1] * (previous.ndim - 2)), -1, 1)
        conditioning = self.codec_condition_proj(previous)
        return self.heads(hidden, conditioning)


    def forward(
        self,
        ref_text: torch.Tensor,
        ref_audio_codec: torch.Tensor,
        text: torch.Tensor,
        audio_codec: torch.Tensor,
        text_lengths: torch.Tensor,
        audio_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        A forward pass strictly for training.
        """

        x, valid_mask, prediction_starts = self._embed_training_batch(
            ref_text,
            ref_audio_codec,
            text,
            audio_codec,
            text_lengths,
            audio_lengths,
        )
        hidden, _ = self._run_transformer(x, kv_cache=None, key_padding_mask=valid_mask)

        max_predictions = audio_codec.size(1) + 1
        gathered = hidden.new_zeros((hidden.size(0), max_predictions, self.d_repr))
        prediction_codes = torch.full(
            (hidden.size(0), max_predictions, self.num_pred_heads),
            config.CODEC_PAD_ID,
            dtype=audio_codec.dtype,
            device=audio_codec.device,
        )
        for i in range(hidden.size(0)):
            count = int(audio_lengths[i].item()) + 1
            start = int(prediction_starts[i].item())
            gathered[i, :count] = hidden[i, start:start + count]
            prediction_codes[i, :count - 1] = audio_codec[i, :count - 1]

        return self._predict(gathered, prediction_codes)

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

        x = self._embed_prompt(ref_text, ref_audio_codec, text)
        hidden, kv_cache = self._run_transformer(x, kv_cache=None)
        return hidden[:, -1:], kv_cache

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

        hidden, new_cache = self._run_transformer(frame_emb, kv_cache)
        return hidden, new_cache

    def _sample_frame(
        self,
        hidden: torch.Tensor,
        temperature: float,
        allow_eos: bool,
        eos_id: int,
    ) -> tuple[torch.Tensor, bool]:
        """Predict one frame, conditioning each codebook on earlier books."""
        greedy = temperature < 1e-5
        frame = torch.zeros((hidden.size(0), self.num_pred_heads), dtype=torch.long, device=hidden.device)
        previous_sum = hidden.new_zeros((hidden.size(0), self.token_embedding_dim))

        for codebook in range(self.num_pred_heads):
            conditioning = None
            if codebook > 0:
                conditioning = self.codec_condition_proj(previous_sum / math.sqrt(codebook))
            logits = self.heads.forward_head(hidden, codebook, conditioning)

            if codebook == 0:
                logits[:, config.CODEC_PAD_ID] = -torch.inf
                if not allow_eos:
                    logits[:, eos_id] = -torch.inf
            else:
                logits[:, config.CODEC_PAD_ID:] = -torch.inf

            if greedy:
                token = logits.argmax(dim=-1)
            else:
                token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
            frame[:, codebook] = token

            if codebook == 0 and bool((token == eos_id).all()):
                return frame, True

            token_embedding = self.codec_embed.embed_codebooks(frame[:, None, :].clamp_max(config.CODEC_PAD_ID))
            previous_sum = token_embedding[:, 0, :codebook + 1].sum(dim=1)

        return frame, False

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterative decoding for a single sequence.

        Returns (frames, lengths) where frames is (1, T, NUM_CODEBOOKS) and lengths is tensor([T]).
        """

        self.eval()
        if ref_text.size(0) != 1:
            raise ValueError("generate currently supports batch size 1")

        # Prefill consumes the prompt + target text (no audio_codec).
        hidden, kv_cache = self.prefill(ref_text, ref_audio_codec, text)

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
            frame_batch, stopped = self._sample_frame(
                hidden[:, 0],
                temperature=temperature,
                allow_eos=i >= min_steps,
                eos_id=eos_id,
            )
            if stopped:
                break
            frame = frame_batch[0]
            frames.append(frame)
            hidden, kv_cache = self.step(frame_batch, position=start_pos + i, kv_cache=kv_cache)

        if not frames:
            empty = torch.zeros(1, 0, self.num_pred_heads, dtype=torch.long, device=ref_text.device)
            return empty, torch.tensor([0], device=ref_text.device)

        lengths = torch.tensor([len(frames)], device=ref_text.device)
        return torch.stack(frames, dim=0).unsqueeze(0), lengths  # (1, T, NB), (1,)
