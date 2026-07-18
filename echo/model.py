from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import config
from echo.codec_embedding import CodecEmbedding
from echo.prediction_head import PredictionHead


# ---------------------------------------------------------------------------
# KV cache container
# ---------------------------------------------------------------------------
# A ``KVCache`` is a list of length ``NUM_LAYERS``; element ``i`` holds the
# cached keys/values for layer ``i`` as tensors of shape
# ``(B, num_heads, past_len, head_dim)``. ``None`` denotes an empty cache.
KVCache = list[Optional[tuple[torch.Tensor, torch.Tensor]]]


@dataclass
class PrefillOutput:
    """Result of a prefill pass, used to seed iterative decoding."""

    next_logits: torch.Tensor   # (B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM) -- predicts the 1st generated frame
    kv_cache: KVCache           # per-layer key/value cache over text + sep + reference audio


@dataclass
class StepOutput:
    """Result of a single iterative decoding step."""

    next_logits: torch.Tensor   # (B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM) -- predicts the next frame
    kv_cache: KVCache           # cache updated with the frame just consumed


# ---------------------------------------------------------------------------
# Transformer decoder building blocks
# ---------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with an explicit KV cache."""

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.nh = num_heads
        self.hd = d_model // num_heads
        self.scale = self.hd ** -0.5

        # Fused QKV projection (one matmul instead of three).
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = dropout
        self.resid_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        cached_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        qkv = self.qkv(x)                       # (B, T, 3C)
        q, k, v = qkv.split(C, dim=-1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        # Append the new keys/values to the per-layer cache.
        if cached_kv is not None:
            k = torch.cat([cached_kv[0], k], dim=2)
            v = torch.cat([cached_kv[1], v], dim=2)
        new_kv = (k, v) if use_cache else None

        # In prefill (no cache) we apply a causal mask; in step mode the single
        # new query sits at the end of the cached sequence, so attending to all
        # cached keys is already causal and no mask is needed.
        is_causal = cached_kv is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=is_causal,
        )                                          # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_drop(self.proj(out))
        return out, new_kv


class FeedForward(nn.Module):
    """Position-wise feed-forward network (GELU-MLP or GLU variant)."""

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        ffn_dim: int = config.FFN_DIM,
        dropout: float = config.DROPOUT,
        use_glu: bool = config.FFN_GLU,
    ) -> None:
        super().__init__()
        self.use_glu = use_glu
        if use_glu:
            self.gate_up = nn.Linear(d_model, 2 * ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)
        else:
            self.up = nn.Linear(d_model, ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)
        self.drop = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_glu:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            x = F.gelu(gate) * up
        else:
            x = F.gelu(self.up(x))
        return self.drop(self.down(x))


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block (causal self-attention + FFN)."""

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        num_heads: int = config.NUM_HEADS,
        ffn_dim: int = config.FFN_DIM,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ffn_dim, dropout, config.FFN_GLU)

    def forward(
        self,
        x: torch.Tensor,
        cached_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, new_kv = self.attn(self.norm1(x), cached_kv, use_cache)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class TransformerDecoder(nn.Module):
    """Stack of pre-norm decoder blocks with KV-cache support."""

    def __init__(
        self,
        d_model: int = config.D_MODEL,
        num_layers: int = config.NUM_LAYERS,
        num_heads: int = config.NUM_HEADS,
        ffn_dim: int = config.FFN_DIM,
        dropout: float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, num_heads, ffn_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)  # final norm before the output projection

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVCache]:
        if kv_cache is None:
            kv_cache = [None] * len(self.blocks)

        new_cache: KVCache = []
        for block, cached_kv in zip(self.blocks, kv_cache):
            x, new_kv = block(x, cached_kv, use_cache)
            new_cache.append(new_kv)

        x = self.norm(x)
        return x, new_cache


# ---------------------------------------------------------------------------
# Prediction heads container
# ---------------------------------------------------------------------------
class PredictionHeads(nn.Module):
    """A stack of ``PRED_NUM_HEADS`` MLPs, one per codebook layer."""

    def __init__(
        self,
        num_heads: int = config.PRED_NUM_HEADS,
        in_dim: int = config.D_REPR,
        hidden_dim: int = config.PRED_HIDDEN_DIM,
        out_dim: int = config.CODEC_LOGIT_DIM,
        num_layers: int = config.PRED_NUM_LAYERS,
        dropout: float = config.PRED_DROPOUT,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [PredictionHead(in_dim, hidden_dim, out_dim, num_layers, dropout) for _ in range(num_heads)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(..., in_dim)`` -> ``(..., num_heads, out_dim)`` logits."""
        # Run every head and stack along a new codebook-axis.
        return torch.stack([head(hidden) for head in self.heads], dim=-2)


# ---------------------------------------------------------------------------
# Echo model
# ---------------------------------------------------------------------------
class Echo(nn.Module):
    """Decoder-only TTS model predicting Mimi codec tokens.

    Inputs
    ------
    * ``text``:        ``(B, T_text)`` long phoneme ids.
    * ``audio_codec``: ``(B, T_audio, NUM_CODEBOOKS)`` long codec ids
                        (reference audio during prefill, or teacher-forcing
                        targets during training).

    Outputs
    -------
    * ``forward``   returns logits for every audio position, shape
                     ``(B, T_audio, NUM_CODEBOOKS, CODEC_LOGIT_DIM)``.
                     Logit at audio position ``t`` predicts the codebook at
                     time step ``t + 1`` (standard next-frame prediction).
    * ``prefill``   returns the logits for the *next* frame plus a KV cache.
    * ``step``      consumes one generated frame and returns the logits for
                     the following frame plus the updated cache.
    """

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
        self.d_emb = d_emb
        self.d_model = d_model
        self.d_repr = d_repr
        self.num_pred_heads = num_pred_heads

        # --- Input embeddings -------------------------------------------------
        self.text_embed = nn.Embedding(text_vocab_size, text_emb_dim)
        self.text_pos = nn.Embedding(text_pos_size, text_emb_dim)
        # Single learnable <sep> token separating text and audio.
        self.sep_embed = nn.Embedding(1, d_emb)
        self.codec_embed = CodecEmbedding()

        # --- Dimension adapters (identity when dims already match) ------------
        self.input_proj = nn.Linear(d_emb, d_model) if d_emb != d_model else nn.Identity()
        self.output_proj = nn.Linear(d_model, d_repr) if d_model != d_repr else nn.Identity()

        # --- Transformer decoder ---------------------------------------------
        self.transformer = TransformerDecoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # --- Prediction heads -------------------------------------------------
        self.heads = PredictionHeads(
            num_heads=num_pred_heads,
            in_dim=d_repr,
            hidden_dim=pred_hidden_dim,
            out_dim=codec_logit_dim,
            num_layers=pred_num_layers,
            dropout=pred_dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.text_embed.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.text_embed.weight[config.TEXT_PAD_ID])
        nn.init.normal_(self.text_pos.weight, mean=0.0, std=config.INIT_STD)
        nn.init.normal_(self.sep_embed.weight, mean=0.0, std=config.INIT_STD)

    # ------------------------------------------------------------------
    # Internal: assemble the input sequence [E_text ; E_sep ; E_codec]
    # ------------------------------------------------------------------
    def _embed_inputs(
        self,
        text: torch.Tensor,
        audio_codec: Optional[torch.Tensor],
        codec_position_offset: int = 0,
    ) -> tuple[torch.Tensor, int, int]:
        """Build the concatenated embedding sequence.

        Returns ``(x, text_len, audio_len)`` where ``x`` has shape
        ``(B, text_len + 1 + audio_len, d_emb)``. ``audio_len`` is 0 when
        ``audio_codec`` is ``None`` (used by the very first prefill step when
        there is no reference audio).
        """
        B = text.size(0)
        T_text = text.size(1)

        text_emb = self.text_embed(text)
        text_pos_ids = torch.arange(T_text, device=text.device)
        text_emb = text_emb + self.text_pos(text_pos_ids).unsqueeze(0)

        sep = self.sep_embed.weight.view(1, 1, -1).expand(B, 1, self.d_emb)

        if audio_codec is not None and audio_codec.numel() > 0:
            audio_emb = self.codec_embed(audio_codec, position_offset=codec_position_offset)
            T_audio = audio_codec.size(1)
            x = torch.cat([text_emb, sep, audio_emb], dim=1)
        else:
            T_audio = 0
            x = torch.cat([text_emb, sep], dim=1)

        return x, T_text, T_audio

    def _run_transformer(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache],
        use_cache: bool,
        audio_len: int,
    ) -> tuple[torch.Tensor, KVCache]:
        """Project, run the transformer, project back to ``d_repr``.

        Returns the representation for the last ``audio_len`` positions
        ``(B, audio_len, d_repr)`` (or the last single position when
        ``audio_len`` is 0) and the updated KV cache.
        """
        x = self.input_proj(x)
        hidden, kv_cache = self.transformer(x, kv_cache, use_cache)
        hidden = self.output_proj(hidden)

        if audio_len > 0:
            hidden = hidden[:, -audio_len:]
        else:
            hidden = hidden[:, -1:]
        return hidden, kv_cache

    def _predict(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map a representation to per-codebook logits."""
        return self.heads(hidden)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, text: torch.Tensor, audio_codec: torch.Tensor) -> torch.Tensor:
        """Teacher-forcing forward pass.

        Processes the full ``[text ; sep ; audio]`` sequence and returns the
        per-codebook logits for every audio position. Logit at audio
        position ``t`` predicts the codebook at time step ``t + 1``.

        Args:
            text:        ``(B, T_text)`` long phoneme ids.
            audio_codec: ``(B, T_audio, NUM_CODEBOOKS)`` long codec ids.

        Returns:
            ``(B, T_audio, NUM_CODEBOOKS, CODEC_LOGIT_DIM)`` logits.
        """
        x, _, T_audio = self._embed_inputs(text, audio_codec)
        hidden, _ = self._run_transformer(
            x, kv_cache=None, use_cache=False, audio_len=T_audio,
        )
        return self._predict(hidden)  # (B, T_audio, NUM_CODEBOOKS, CODEC_LOGIT_DIM)

    def prefill(self, text: torch.Tensor, audio_codec: Optional[torch.Tensor] = None) -> PrefillOutput:
        """Warm-start iterative decoding.

        Processes the conditioning (text + optional reference audio) and
        returns the logits that predict the *first* generated frame, along
        with the KV cache covering all conditioning tokens.

        Args:
            text:        ``(B, T_text)`` long phoneme ids.
            audio_codec: ``(B, T_audio, NUM_CODEBOOKS)`` long reference codec
                         ids, or ``None`` to condition on text only.

        Returns:
            :class:`PrefillOutput` with ``next_logits`` of shape
            ``(B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM)``.
        """
        x, T_text, T_audio = self._embed_inputs(text, audio_codec)
        hidden, kv_cache = self._run_transformer(
            x, kv_cache=None, use_cache=True, audio_len=0,  # take the last position
        )
        next_logits = self._predict(hidden[:, -1:, :])  # (B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM)
        # Remember how many tokens are in the cache so step() can place the
        # next frame at the correct absolute audio position.
        self._prefill_seq_len = T_text + 1 + T_audio
        self._prefill_audio_len = T_audio
        return PrefillOutput(next_logits=next_logits, kv_cache=kv_cache)

    def step(
        self,
        codebook: torch.Tensor,
        position: int,
        kv_cache: KVCache,
    ) -> StepOutput:
        """One iterative decoding step.

        Consumes the codebook frame generated for the *previous* time step,
        feeds it through the transformer using the cached keys/values, and
        returns the logits that predict the *next* frame.

        Args:
            codebook: ``(B, NUM_CODEBOOKS)`` long ids of the frame just
                      generated (at absolute audio ``position``).
            position: absolute audio position of ``codebook`` (0-indexed).
                      The first generated frame sits at
                      ``prefill_audio_len``.
            kv_cache: per-layer KV cache from prefill / previous steps.

        Returns:
            :class:`StepOutput` with ``next_logits`` of shape
            ``(B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM)``.
        """
        frame_emb = self.codec_embed(
            codebook.unsqueeze(1), position_offset=position,
        )                                            # (B, 1, d_emb)
        frame_emb = self.input_proj(frame_emb)
        hidden, new_cache = self.transformer(frame_emb, kv_cache, use_cache=True)
        hidden = self.output_proj(hidden)                 # (B, 1, d_repr)
        next_logits = self._predict(hidden)              # (B, 1, NUM_CODEBOOKS, CODEC_LOGIT_DIM)
        return StepOutput(next_logits=next_logits, kv_cache=new_cache)

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
        """Iterative decoding with per-sequence EOS tracking.

        EOS is decided from the **first** codebook layer's logits (head 0):
        when that head selects ``eos_id`` for a given sequence, that sequence
        stops. EOS is a *prediction-only* token -- it is never fed back into
        the codec embedding. The returned grid **excludes the EOS frame**;
        trailing positions for sequences that finish early are filled with
        ``pad_id`` so the grid stays rectangular.

        Args:
            text:        ``(B, T_text)`` phoneme ids.
            audio_codec: optional ``(B, T_audio, NUM_CODEBOOKS)`` reference.
            max_steps:   hard cap on the number of generated frames.
            temperature: sampling temperature; values below ``1e-5`` select
                         deterministic greedy ``argmax`` decoding.
            eos_id:      token id interpreted as end-of-sequence (head 0).
            pad_id:      valid codec id (default 0) used to fill frames after
                         a sequence has finished and to replace any sampled
                         ``eos_id`` before it is fed back into the model.

        Returns:
            ``(frames, lengths)`` where ``frames`` is a
            ``(B, T_gen, NUM_CODEBOOKS)`` long tensor and ``lengths`` is a
            ``(B,)`` long tensor giving the number of real (non-EOS) frames
            produced per sequence. ``T_gen`` is ``max(lengths)``.
        """
        self.eval()
        B = text.size(0)
        greedy = temperature < 1e-5

        pre = self.prefill(text, audio_codec)
        logits = pre.next_logits                       # (B, 1, NB, V+1)
        kv_cache = pre.kv_cache
        start_pos = self._prefill_audio_len            # first generated audio position

        frames = torch.zeros(
            B, max_steps, self.num_pred_heads, dtype=torch.long, device=text.device,
        )
        lengths = torch.zeros(B, dtype=torch.long, device=text.device)
        finished = torch.zeros(B, dtype=torch.bool, device=text.device)

        for i in range(max_steps):
            active = ~finished
            if not active.any():
                break

            frame_logits = logits[:, 0]                 # (B, NB, V+1)
            if greedy:
                frame = frame_logits.argmax(dim=-1)   # (B, NB)
            else:
                probs = torch.softmax(frame_logits / temperature, dim=-1)
                flat = probs.reshape(B * self.num_pred_heads, -1)
                frame = torch.multinomial(flat, num_samples=1).view(
                    B, self.num_pred_heads,
                )

            # New EOS on head 0, only for sequences still active.
            new_eos = active & (frame[:, 0] == eos_id)
            # Sequences that produced a real (non-EOS) frame at this step.
            real = active & ~new_eos
            frames[real, i] = frame[real]
            lengths[real] = i + 1
            finished = finished | new_eos

            if finished.all():
                break

            # Feed the frame back in. EOS must NEVER reach the codec embedding:
            # replace any eos_id with pad_id and pad out finished sequences.
            feed = frame.clone()
            feed[feed == eos_id] = pad_id
            feed[finished] = pad_id
            step = self.step(feed, position=start_pos + i, kv_cache=kv_cache)
            logits = step.next_logits
            kv_cache = step.kv_cache

        t_gen = int(lengths.max().item()) if lengths.numel() and lengths.max() > 0 else 0
        return frames[:, :t_gen], lengths
