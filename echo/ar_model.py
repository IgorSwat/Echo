from echo import config

from echo.components.prediction_head import PredictionHead
from echo.components.text_encoder import TextEncoder

from echo.nn.transformer import HybridAttentionDecoder
from echo.nn.types import HybridKVCache

from typing import Optional, Union

import torch
import torch.nn as nn


class EchoAR(nn.Module):
    """
    EchoAR: text-conditioned autoregressive model over Mimi's semantic tokens (1 layer).
    """

    # How many codec layers the model reads and writes. Kept as a named constant
    # because the dataset has to be told how many to load.
    NUM_TOKEN_LAYERS = 1

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

        # --- Token embeddings ---
        self.embed = nn.Embedding(self.vocab_size, cfg.emb_dim)
        d_tokens = cfg.emb_dim

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

        # --- Output head ---
        self.head = PredictionHead(
            cfg.hidden_dim, cfg.head_hidden_dim, self.vocab_size,
            num_layers=cfg.head_num_layers,
            dropout=cfg.head_dropout,
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
        nn.init.normal_(self.embed.weight, mean=0.0, std=config.init_std)
        for module in (self.in_proj, self.ctc_head, self.ctc_up):
            if module is not None:
                nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
                nn.init.zeros_(module.bias)

    def load_weights(self, state: dict) -> None:
        """Load a checkpoint, tolerating a missing CTC head but nothing else.

        The CTC branch is a training-time auxiliary that inference never touches,
        so a checkpoint trained before it existed — or with it disabled — is
        still a perfectly good model.
        """
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
        x: torch.Tensor,                                             # (B, T) long
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                    # decoder caches, per block
        start_pos: int = 0,                                          # frames already decoded
    ) -> tuple[torch.Tensor, HybridKVCache]:
        """Everything up to the head: the frame states and the caches."""
        if x.dim() != 2:
            raise ValueError(f"expected a (B, T) token sequence, got {tuple(x.shape)}")
        if context is None and text is None:
            raise ValueError("provide either `text` or a precomputed `context`")

        if start_pos > 0:
            x = x[:, start_pos:]                                     # (B, T_new)

        h = self.embed(x)                                            # (B, T, emb_dim)

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
        x: torch.Tensor,                                             # (B, T) long
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                    # decoder caches, per block
        start_pos: int = 0,                                          # frames already decoded
        return_cache: bool = False,
        return_hidden: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        """
        A forward pass through the model.
        """

        h, caches = self._trunk(
            x, text, padding_mask, text_padding_mask, context, kv_cache, start_pos,
        )                                                            # (B, T_new, hidden_dim)

        logits = self.head(h)                                        # (B, T, vocab)

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

    def _sample_token(
        self,
        h: torch.Tensor,                                             # (B, 1, hidden_dim)
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Draw the next token from one frame state.
        """

        logits = self.head(h)[:, 0]                                  # (B, vocab)
        logits[..., config.prosody_bos] = float("-inf")
        logits[..., config.prosody_pad] = float("-inf")
        # The mask token is a training-time input only; never decodable.
        logits[..., config.prosody_mask] = float("-inf")

        if temperature == 0.0:
            return logits.argmax(dim=-1)                             # (B,)

        logits = logits / temperature
        if top_k > 0:
            # Everything below the k-th largest logit drops out of the draw.
            # Suppressed ids are already -inf and stay there.
            kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(1)

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

        Returns ``(B, T)`` of Mimi semantic-layer token ids.
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

        x = torch.full((B, 1), config.prosody_bos, dtype=torch.long, device=text.device)
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

            nxt = self._sample_token(h[:, -1:], temperature, top_k)  # (B,)
            finished = finished | (nxt == config.prosody_eos)
            lengths += (~finished).long()                            # (B,)

            # Already-terminated rows contribute padding from here on, so every
            # frame decoded past EOS is pad and drops out in the trim.
            nxt = torch.where(finished, torch.full_like(nxt, config.prosody_pad), nxt)
            x = torch.cat([x, nxt[:, None]], dim=1)                  # (B, t + 1)

            if (step + 1) % eos_check_every == 0 and bool(finished.all()):
                break

        if was_training:
            self.train()

        # One synchronization for the whole decode: the longest row decides how much of the padded tail survives.
        keep = int(lengths.max())
        out_tokens = x[:, 1:1 + keep]                                # (B, T), no BOS

        valid = (out_tokens >= 0) & (out_tokens < self.CODEBOOK_SIZE)

        return torch.where(valid, out_tokens, torch.zeros_like(out_tokens))
