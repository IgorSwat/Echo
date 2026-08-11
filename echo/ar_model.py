from echo import config

from echo.components.multihead_predictor import MultiHeadPredictor
from echo.components.text_encoder import TextEncoder

from echo.nn.transformer import HybridAttentionDecoder
from echo.nn.types import HybridKVCache

from typing import Optional, Union

import torch
import torch.nn as nn


class EchoAR(nn.Module):
    """
    EchoAR: text-conditioned autoregressive model over two layers of prosody tokens.
    """

    # Number of stacked prosody token layers (x[..., 0] and x[..., 1]).
    NUM_TOKEN_LAYERS = 2

    # How many decoding steps to run between two EOS checks.
    EOS_CHECK_EVERY = 6

    # Size of Mimi's acoustic codebooks: ids [0, 2047] address a real entry,
    # everything above is one of the special tokens (pad/bos/eos), which the
    # codec cannot decode. ``prosody_pad`` is the lowest of them.
    CODEBOOK_SIZE = config.prosody_pad

    def __init__(self) -> None:
        super().__init__()

        cfg = config.ar_model

        self.emb_dim = cfg.emb_dim
        self.hidden_dim = cfg.hidden_dim
        self.vocab_size = config.prosody_vocab_size

        # --- Token embeddings (one table per prosody layer) ---
        # Concatenated rather than summed, so the decoder can tell them apart.
        d_tokens = self.NUM_TOKEN_LAYERS * cfg.emb_dim
        self.embed = nn.ModuleList([
            nn.Embedding(self.vocab_size, cfg.emb_dim)
            for _ in range(self.NUM_TOKEN_LAYERS)
        ])

        # --- Text conditioning ---
        self.text_encoder = TextEncoder(
            vocab_size=config.text_vocab_size,
            d_model=cfg.text_encoder_d_model,
            d_out=cfg.text_encoder_d_model,
            num_layers=cfg.text_encoder_num_layers,
            num_heads=cfg.text_encoder_num_heads,
            ffn_dim=cfg.text_encoder_ffn_dim,
            ffn_glu=cfg.text_encoder_ffn_glu,
            kernel_size=cfg.text_encoder_kernel_size,
            dropout=cfg.text_encoder_dropout,
            use_rope=cfg.text_encoder_use_rope,
            conv_use_norm=cfg.text_encoder_conv_use_norm,
            max_seq_len=config.text_len_limit,
        )

        # --- Main decoder stack ---
        # Widen the token stream to the decoder dim when the two differ.
        self.in_proj = (
            nn.Linear(d_tokens, cfg.hidden_dim) if d_tokens != cfg.hidden_dim else None
        )
        self.decoder = HybridAttentionDecoder(
            d_model=cfg.hidden_dim,
            d_kv=cfg.text_encoder_d_model,
            num_layers=cfg.decoder_num_layers,
            num_heads=cfg.decoder_num_heads,
            ffn_dim=cfg.decoder_ffn_dim,
            dropout=cfg.decoder_dropout,
            use_rope=cfg.decoder_use_rope,
            use_glu=cfg.decoder_ffn_glu,
            rope_norm=cfg.decoder_rope_norm,
        )

        # --- Output heads  ---
        self.predictor = MultiHeadPredictor(
            self.NUM_TOKEN_LAYERS,
            cfg.hidden_dim, cfg.head_hidden_dim, self.vocab_size,
            num_layers=cfg.head_num_layers,
            dropout=cfg.head_dropout,
            cond_dim=cfg.emb_dim if cfg.head_intra_frame_cond else None,
        )

        # --- Auxiliary CTC head (training only) ---
        # Teacher-forced cross-entropy cannot see a skipped word; CTC scores the
        # whole frame sequence against the whole phoneme string, so it can. The
        # transposed conv upsamples first: CTC needs one frame per phoneme, and
        # the 12.5 Hz grid sits below the phoneme rate of speech.
        self.ctc_upsample = cfg.ctc_upsample if cfg.ctc_enabled else 0
        if cfg.ctc_enabled:
            if cfg.ctc_upsample < 1:
                raise ValueError(f"ctc_upsample must be >= 1, got {cfg.ctc_upsample}")
            self.ctc_up = (
                nn.ConvTranspose1d(cfg.hidden_dim, cfg.hidden_dim,
                                   cfg.ctc_upsample, stride=cfg.ctc_upsample)
                if cfg.ctc_upsample > 1 else None
            )
            # One class per text token plus the CTC blank, which takes the index
            # just past the alphabet.
            self.ctc_blank = config.text_vocab_size
            self.ctc_head = nn.Linear(cfg.hidden_dim, config.text_vocab_size + 1)
        else:
            self.ctc_up = None
            self.ctc_head = None
            self.ctc_blank = None

        self._init_weights()

    def _init_weights(self) -> None:
        # Submodules initialized themselves; this covers what EchoAR owns directly.
        for emb in self.embed:
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)
        for module in (self.in_proj, self.ctc_head, self.ctc_up):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
                nn.init.zeros_(module.bias)

    # The heads and their FiLM modulation used to hang off EchoAR directly, before
    # they were gathered into `predictor`. Checkpoints from then are remapped on
    # load; nothing about the weights themselves changed.
    _MOVED_PREFIXES = (("heads.", "predictor.heads."), ("film.", "predictor.film."))

    def load_weights(self, state: dict) -> None:
        """Load a checkpoint, tolerating a missing CTC head but nothing else.

        The CTC branch is a training-time auxiliary that inference never touches,
        so a checkpoint trained before it existed — or with it disabled — is
        still a perfectly good model.
        """
        state = {
            next((new + k[len(old):] for old, new in self._MOVED_PREFIXES
                  if k.startswith(old)), k): v
            for k, v in state.items()
        }
        missing, unexpected = self.load_state_dict(state, strict=False)
        missing = [k for k in missing if not k.startswith("ctc_")]
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint does not match the model: "
                f"missing {missing}, unexpected {unexpected}"
            )

    # ---------------
    # Forward passes
    # ---------------

    def encode_text(
        self,
        text: torch.Tensor,                                          # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
    ) -> torch.Tensor:
        """Run the text branch on its own."""

        return self.text_encoder(text, text_padding_mask)            # (B, S, d_text)

    def ctc_log_probs(self, h: torch.Tensor) -> torch.Tensor:
        """
        Per-frame phoneme log-probabilities for the CTC loss.
        """
        
        if self.ctc_head is None:
            raise RuntimeError(
                "the CTC head is disabled; set ar_model.ctc.enabled in config.json"
            )

        if self.ctc_up is not None:
            h = self.ctc_up(h.transpose(1, 2)).transpose(1, 2)       # (B, T*u, hidden_dim)

        return self.ctc_head(h).log_softmax(dim=-1)                  # (B, T*u, text_vocab + 1)

    def _trunk(
        self,
        x: torch.Tensor,                                             # (B, T, 2) long
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                    # decoder caches, per block
        start_pos: int = 0,                                          # frames already decoded
    ) -> tuple[torch.Tensor, HybridKVCache]:
        """Everything up to the per-layer heads: the frame states and the caches."""
        if x.shape[-1] != self.NUM_TOKEN_LAYERS:
            raise ValueError(
                f"expected {self.NUM_TOKEN_LAYERS} token layers, got {x.shape[-1]}"
            )
        if context is None and text is None:
            raise ValueError("provide either `text` or a precomputed `context`")

        if start_pos > 0:
            x = x[:, start_pos:]                                     # (B, T_new, 2)

        h = torch.cat(
            [emb(x[..., i]) for i, emb in enumerate(self.embed)], dim=-1
        )                                                            # (B, T, 2*emb_dim)

        # Text conditioning (bidirectional; the text is fully known up front).
        ctx = context if context is not None else self.encode_text(text, text_padding_mask)

        if self.in_proj is not None:
            h = self.in_proj(h)                                      # (B, T, hidden_dim)

        return self.decoder(
            h, ctx, padding_mask, text_padding_mask,
            kv_cache=kv_cache, start_pos=start_pos,
        )                                                            # (B, T_new, hidden), caches

    def forward(
        self,
        x: torch.Tensor,                                             # (B, T, 2) long
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                    # decoder caches, per block
        start_pos: int = 0,                                          # frames already decoded
        return_cache: bool = False,
        cond_tokens: Optional[torch.Tensor] = None,                  # (B, T, 1) long or None
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        A forward pass through the model.
        """

        h, caches = self._trunk(
            x, text, padding_mask, text_padding_mask, context, kv_cache, start_pos,
        )                                                            # (B, T_new, hidden_dim)

        if self.predictor.uses_cond and cond_tokens is None:
            raise ValueError("intra_frame_cond is enabled; forward() needs `cond_tokens`")
        if cond_tokens is not None and start_pos > 0:
            cond_tokens = cond_tokens[:, start_pos:]                 # follow the `x` slice

        # Layer j is embedded with `self.embed[j]`, the input table for that very
        # layer, so the predictor's FiLM reads the same alphabet the decoder does.
        cond_emb = None
        if self.predictor.uses_cond and cond_tokens is not None:
            cond_emb = torch.stack(
                [self.embed[j](cond_tokens[..., j]) for j in range(cond_tokens.shape[-1])],
                dim=2,
            )                                                        # (B, T, layers-1, emb_dim)

        logits = self.predictor(h, cond_emb)                         # (B, T, 2, vocab)

        if return_cache and return_hidden:
            return logits, caches, h
        if return_cache:
            return logits, caches
        if return_hidden:
            return logits, h

        return logits

    # -----------
    # Generation
    # -----------

    def _sample_frame(
        self,
        h: torch.Tensor,                                             # (B, 1, hidden_dim)
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Decode one frame, layer by layer.
        """

        frame: list[torch.Tensor] = []
        cond: Optional[torch.Tensor] = None                          # (B, 1) or None

        for k in range(self.NUM_TOKEN_LAYERS):
            cond_emb = None if cond is None else self.embed[k - 1](cond)
            logits = self.predictor.head(k, h, cond_emb)[:, 0]       # (B, vocab)
            logits[..., config.prosody_bos] = float("-inf")
            logits[..., config.prosody_pad] = float("-inf")
            # The mask token is a training-time input only; never decodable.
            logits[..., config.prosody_mask] = float("-inf")

            if temperature == 0.0:
                tok = logits.argmax(dim=-1)                          # (B,)
            else:
                logits = logits / temperature
                if top_k > 0:
                    # Everything below the k-th largest logit drops out of the
                    # draw. Suppressed ids are already -inf and stay there.
                    kth = logits.topk(
                        min(top_k, logits.shape[-1]), dim=-1
                    ).values[:, -1:]                                 # (B, 1)
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                tok = torch.multinomial(
                    logits.softmax(dim=-1), num_samples=1
                ).squeeze(1)                                         # (B,)

            frame.append(tok)
            cond = tok[:, None]                                      # feeds the next layer

        return torch.stack(frame, dim=1)                             # (B, layers)

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,                                          # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        max_frames: int = 1000,
        use_cache: bool = True,
        eos_check_every: Optional[int] = None,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive decoding from text alone.

        Returns ``(B, T, NUM_TOKEN_LAYERS)`` of codec token ids.
        """

        eos_check_every = self.EOS_CHECK_EVERY if eos_check_every is None else eos_check_every
        if eos_check_every < 1:
            raise ValueError(f"eos_check_every must be >= 1, got {eos_check_every}")
        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")

        was_training = self.training
        self.eval()

        B = text.shape[0]
        ctx = self.encode_text(text, text_padding_mask)              # (B, S, d_text)

        x = torch.full(
            (B, 1, self.NUM_TOKEN_LAYERS), config.prosody_bos,
            dtype=torch.long, device=text.device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=text.device)
        # Real (pre-EOS) frames per row, tracked on-device so that counting it
        # costs no synchronization; only the final trim reads it back.
        lengths = torch.zeros(B, dtype=torch.long, device=text.device)

        caches: Optional[HybridKVCache] = None
        for step in range(max_frames):
            # Everything before the newest frame is already in the caches, so the
            # decoder only has to run on what was appended since the last step.
            start_pos = x.shape[1] - 1 if caches is not None else 0
            h, new_caches = self._trunk(
                x, context=ctx, text_padding_mask=text_padding_mask,
                kv_cache=caches, start_pos=start_pos,
            )
            caches = new_caches if use_cache else None

            nxt = self._sample_frame(h[:, -1:], temperature, top_k)  # (B, layers)
            finished = finished | (nxt[:, 0] == config.prosody_eos)
            lengths += (~finished).long()                            # (B,)

            # Already-terminated rows contribute padding from here on, so every
            # frame decoded past EOS is pad and drops out in the trim.
            nxt = torch.where(
                finished[:, None], torch.full_like(nxt, config.prosody_pad), nxt
            )
            x = torch.cat([x, nxt[:, None, :]], dim=1)               # (B, t + 1, layers)

            if (step + 1) % eos_check_every == 0 and bool(finished.all()):
                break

        if was_training:
            self.train()

        # One synchronization for the whole decode: the longest row decides how much of the padded tail survives.
        keep = int(lengths.max())
        out_tokens = x[:, 1:1 + keep]                                # (B, T, layers), no BOS

        valid = (out_tokens >= 0) & (out_tokens < self.CODEBOOK_SIZE)

        return torch.where(valid, out_tokens, torch.zeros_like(out_tokens))
