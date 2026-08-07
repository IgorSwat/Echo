from echo import config

from echo.modules.text_encoder import TextEncoder
from echo.modules.transformer import HybridAttentionDecoder
from echo.modules.types import HybridKVCache

from typing import Optional, Union

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    MLP token head, predicting given layer of the output codebook.
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        layers: list[nn.Module] = []
        dim = d_in
        for _ in range(num_layers - 1):
            layers += [nn.Linear(dim, d_hidden), nn.GELU(), nn.Dropout(dropout)]
            dim = d_hidden
        layers.append(nn.Linear(dim, d_out))

        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, T, d_in)
        return self.net(x)                                               # (B, T, d_out)


class IntraFrameFiLM(nn.Module):
    """
    Modulate the frame state by a token from the codebook layer below it.
    """

    def __init__(self, d_model: int, d_cond: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(d_cond, 2 * d_model)

        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self,
        h: torch.Tensor,                                             # (B, T, d_model)
        cond_emb: torch.Tensor,                                      # (B, T, d_cond)
    ) -> torch.Tensor:
        scale, shift = self.to_scale_shift(cond_emb).chunk(2, dim=-1)
        return h + (scale * self.norm(h) + shift)                    # (B, T, d_model)


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

        # Token layers are embedded separately and concatenated.
        d_tokens = self.NUM_TOKEN_LAYERS * cfg.emb_dim

        # --- Token embeddings (one table per prosody layer) ---
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
        self.in_proj = nn.Linear(d_tokens, cfg.hidden_dim) if d_tokens != cfg.hidden_dim else None

        self.decoder = HybridAttentionDecoder(
            d_model=cfg.hidden_dim,
            d_kv=cfg.text_encoder_d_model,
            num_layers=cfg.decoder_num_layers,
            num_heads=cfg.decoder_num_heads,
            ffn_dim=cfg.decoder_ffn_dim,
            dropout=cfg.decoder_dropout,
            use_rope=cfg.decoder_use_rope,
            ffn_glu=cfg.decoder_ffn_glu,
            rope_norm=cfg.decoder_rope_norm,
        )

        # --- Output heads (one per token layer) ---
        self.heads = nn.ModuleList([
            PredictionHead(
                cfg.hidden_dim, cfg.head_hidden_dim, self.vocab_size,
                num_layers=cfg.head_num_layers,
                dropout=cfg.head_dropout,
            )
            for _ in range(self.NUM_TOKEN_LAYERS)
        ])

        # --- Intra-frame conditioning (one FiLM per head above the first) ---
        # Head k is modulated by layer k - 1 of the same frame; head 0 has no
        # lower layer to read and stays unconditioned.
        self.film = nn.ModuleList([
            IntraFrameFiLM(cfg.hidden_dim, cfg.emb_dim)
            for _ in range(self.NUM_TOKEN_LAYERS - 1)
        ]) if cfg.head_intra_frame_cond else None

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embed:
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)
        if self.in_proj is not None:
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.in_proj.bias)

    def encode_text(
        self,
        text: torch.Tensor,                                         # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
    ) -> torch.Tensor:
        """
        Run the text branch on its own.
        """

        return self.text_encoder(text, text_padding_mask)            # (B, S, d_text)

    def _trunk(
        self,
        x: torch.Tensor,                                            # (B, T, 2) long
        text: Optional[torch.Tensor] = None,                        # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
        context: Optional[torch.Tensor] = None,                     # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                   # decoder caches, per block
        start_pos: int = 0,                                         # frames already decoded
    ) -> tuple[torch.Tensor, HybridKVCache]:
        """
        Everything up to the per-layer heads: the frame states and the caches.
        """
        if x.shape[-1] != self.NUM_TOKEN_LAYERS:
            raise ValueError(
                f"expected {self.NUM_TOKEN_LAYERS} token layers, got {x.shape[-1]}"
            )
        if context is None and text is None:
            raise ValueError("provide either `text` or a precomputed `context`")

        # Nothing between the embeddings and the decoder mixes across time, so
        # mid-decode the frames already covered by the caches can be dropped up
        # front: only the new ones need embedding at all.
        if start_pos > 0:
            x = x[:, start_pos:]                                     # (B, T_new, 2)

        # Each token layer gets its own table; the two embeddings are concatenated
        # along the feature dim rather than summed, so the decoder can tell them apart.
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
        )                                                            # (B, T_new, hidden_dim), caches

    def _head(
        self,
        layer: int,
        h: torch.Tensor,                                            # (B, T, hidden_dim)
        cond: Optional[torch.Tensor] = None,                        # (B, T) long or None
    ) -> torch.Tensor:
        """
        Logits for one token layer, optionally modulated by the layer below it.
        """
        if self.film is not None and layer > 0:
            if cond is None:
                raise ValueError(
                    f"head {layer} needs layer {layer - 1}'s tokens when "
                    "intra_frame_cond is enabled"
                )
            # The conditioning token is embedded with the *input* table for its
            # own layer: same alphabet, same meaning, one less table to train.
            h = self.film[layer - 1](h, self.embed[layer - 1](cond))
        return self.heads[layer](h)                                  # (B, T, vocab)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, 2) long
        text: Optional[torch.Tensor] = None,                        # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
        context: Optional[torch.Tensor] = None,                     # (B, S, d_text) or None
        kv_cache: Optional[HybridKVCache] = None,                   # decoder caches, per block
        start_pos: int = 0,                                         # frames already decoded
        return_cache: bool = False,
        cond_tokens: Optional[torch.Tensor] = None,                 # (B, T, 1) long or None
    ) -> Union[torch.Tensor, tuple[torch.Tensor, HybridKVCache]]:
        """
        Logits for every frame in `x`; with `return_cache`, the decoder caches too.
        """
        h, caches = self._trunk(
            x, text, padding_mask, text_padding_mask, context, kv_cache, start_pos,
        )                                                            # (B, T_new, hidden_dim)

        if self.film is not None and cond_tokens is None:
            raise ValueError("intra_frame_cond is enabled; forward() needs `cond_tokens`")
        if cond_tokens is not None and start_pos > 0:
            cond_tokens = cond_tokens[:, start_pos:]                 # follow the `x` slice

        # One head per token layer, stacked to mirror the input layout.
        logits = torch.stack(
            [
                self._head(k, h, None if cond_tokens is None or k == 0 else cond_tokens[..., k - 1])
                for k in range(self.NUM_TOKEN_LAYERS)
            ],
            dim=2,
        )                                                            # (B, T, 2, vocab)

        return (logits, caches) if return_cache else logits

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,                                         # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
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
        greedy = temperature == 0.0

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
            h = h[:, -1:]                                            # (B, 1, hidden_dim)

            # Layers of one frame are decoded in sequence, not in parallel: with
            # intra-frame conditioning on, layer k's head reads the token layer
            # k - 1 just produced. The decoder is not re-run — only the heads are.
            frame: list[torch.Tensor] = []
            cond: Optional[torch.Tensor] = None                      # (B, 1) or None
            for k in range(self.NUM_TOKEN_LAYERS):
                logits = self._head(k, h, cond)[:, 0]                # (B, vocab)
                logits[..., config.prosody_bos] = float("-inf")
                logits[..., config.prosody_pad] = float("-inf")
                # The mask token is a training-time input only; never decodable.
                logits[..., config.prosody_mask] = float("-inf")

                if greedy:
                    tok = logits.argmax(dim=-1)                      # (B,)
                else:
                    logits = logits / temperature
                    if top_k > 0:
                        # Everything below the k-th largest logit drops out of the
                        # draw. Suppressed ids are already -inf and stay there.
                        kth = logits.topk(
                            min(top_k, logits.shape[-1]), dim=-1
                        ).values[:, -1:]                             # (B, 1)
                        logits = logits.masked_fill(logits < kth, float("-inf"))
                    tok = torch.multinomial(
                        logits.softmax(dim=-1), num_samples=1
                    ).squeeze(1)                                     # (B,)
                frame.append(tok)
                cond = tok[:, None]                                  # feeds the next layer

            nxt = torch.stack(frame, dim=1)                          # (B, layers)
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

        # One synchronization for the whole decode: the longest row decides how
        # much of the padded tail survives.
        keep = int(lengths.max())

        out_tokens = x[:, 1:1 + keep]                                # (B, T, layers), BOS dropped

        # Termination is read off layer 0 alone, so any special id the other
        # layers emit — or any id at all outside the acoustic codebook — can
        # survive the trim. Mimi's decoder indexes its codebooks with these
        # directly, and an out-of-range id faults the lookup kernel, so
        # everything outside [0, CODEBOOK_SIZE - 1] is folded back to 0.
        valid = (out_tokens >= 0) & (out_tokens < self.CODEBOOK_SIZE)
        return torch.where(valid, out_tokens, torch.zeros_like(out_tokens))
