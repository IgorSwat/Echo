from echo import config

from echo.components.prediction_head import PredictionHead
from echo.components.text_encoder import TextEncoder

from echo.nn.transformer import HybridAttentionEncoder

from typing import Optional

import torch
import torch.nn as nn


class EchoNAR(nn.Module):
    """
    EchoNAR: non-autoregressive model over Mimi's acoustic codebooks (layers 1+).

    EchoAR writes layer 0 — Mimi's semantic quantizer, distilled from WavLM —
    one frame at a time. Everything above it is a residual RVQ: layer k holds
    the quantization error the layers below it left behind, so it is largely
    *determined* by them within a frame and close to unpredictable across
    frames. That is the opposite of what a causal decoder is good for, so this
    stage drops causality and time-stepping alike and writes one whole layer per
    forward pass:

        layer 0        -> layer 1
        layers 0..1    -> layer 2
        ...
        layers 0..K-2  -> layer K-1

    which is VALL-E's NAR stage. Every layer is written by the same network: the
    codes below it are summed in embedding space, the layer being written enters
    as AdaLN conditioning, and only the output head is per-layer. A layer's
    prediction still sees the whole utterance, in both directions, plus the text.

    NOTE: there is no acoustic prompt. VALL-E needs one because it has to carry
    an unseen speaker's timbre into the acoustic layers; here the corpus is one
    speaker and the timbre is the model's to memorize. A prompt would be the
    first thing to add for a multi-speaker corpus.
    """

    # Codec layers this stage spans, layer 0 (the AR stage's) included; it writes
    # the ``NUM_TOKEN_LAYERS - 1`` above it. The dataset has to load the same
    # number, so it lives here as a named constant, as it does on EchoAR.
    NUM_TOKEN_LAYERS = config.nar_model.num_layers

    # Size of Mimi's codebooks: ids [0, 2047] address a real entry, everything
    # above is a special token the codec cannot decode. Unlike the AR stage this
    # model has no use for BOS/EOS — layer 0 fixes the frame count before it
    # starts — so its heads only ever score real ids.
    CODEBOOK_SIZE = config.prosody_pad

    def __init__(self) -> None:
        super().__init__()

        cfg = config.nar_model

        if cfg.num_layers < 2:
            raise ValueError(
                f"nar_model.num_layers must be >= 2 (layer 0 plus at least one "
                f"layer to write), got {cfg.num_layers}"
            )

        self.num_layers = cfg.num_layers
        self.num_written = cfg.num_layers - 1
        self.emb_dim = cfg.emb_dim
        self.hidden_dim = cfg.hidden_dim
        self.cond_dim = cfg.cond_dim

        # --- Token embeddings ---
        # One table per *input* layer, so layer K-1 gets none: it is only ever
        # predicted, never read. Sized to the full prosody vocabulary so the pad
        # id of a padded frame is embeddable, as it is on the AR side.
        self.embed = nn.ModuleList([
            nn.Embedding(config.prosody_vocab_size, cfg.emb_dim)
            for _ in range(self.num_written)
        ])

        # --- Layer conditioning ---
        # The only thing separating "write layer 1" from "write layer 7" inside
        # the stack. Indexed by ``layer - 1``, since layer 0 is never written.
        self.layer_embed = nn.Embedding(self.num_written, cfg.cond_dim)

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

        # --- Main stack ---
        self.in_proj = (
            nn.Linear(cfg.emb_dim, cfg.hidden_dim)
            if cfg.emb_dim != cfg.hidden_dim else None
        )
        self.encoder = HybridAttentionEncoder(
            d_model=cfg.hidden_dim,
            d_kv=cfg.text_encoder_d_model,
            num_layers=cfg.encoder_num_layers,
            num_heads=cfg.encoder_num_heads,
            ffn_dim=cfg.encoder_ffn_dim,
            dropout=cfg.encoder_dropout,
            use_rope=cfg.encoder_use_rope,
            use_glu=cfg.encoder_ffn_glu,
            rope_norm=cfg.encoder_rope_norm,
            use_ada_ln=True,
            cond_dim=cfg.cond_dim,
        )

        # --- Output heads, one per written layer ---
        # Shared trunk, split head: the layers agree about what the frame is and
        # disagree only about which residual comes next, and a codebook's ids
        # mean nothing to its neighbour, so the last projection cannot be shared.
        self.heads = nn.ModuleList([
            PredictionHead(
                cfg.hidden_dim, cfg.head_hidden_dim, self.CODEBOOK_SIZE,
                num_layers=cfg.head_num_layers,
                dropout=cfg.head_dropout,
            )
            for _ in range(self.num_written)
        ])

        self._init_weights()

    def _init_weights(self) -> None:
        # Submodules initialized themselves; this covers what EchoNAR owns directly.
        for table in self.embed:
            nn.init.normal_(table.weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.layer_embed.weight, mean=0.0, std=config.init_std)
        if self.in_proj is not None:
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.in_proj.bias)

    def load_weights(self, state: dict) -> None:
        """Load a checkpoint, refusing anything that does not match exactly.

        Kept for symmetry with EchoAR, whose CTC head makes its version
        interesting; here there is no optional branch, so this is a plain load.
        """
        self.load_state_dict(state)

    # ---------------
    # Forward passes
    # ---------------

    def encode_text(
        self,
        text: torch.Tensor,                                          # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
    ) -> torch.Tensor:
        """Run the text branch on its own, once for every layer that follows."""

        return self.text_encoder(text, text_padding_mask)            # (B, S, d_text)

    def embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Sum the embeddings of every layer already written.

        Summation, not concatenation: the stack is a residual quantizer, so the
        layers add up to one vector by construction, and their embeddings are
        asked to do the same. It also keeps the input width independent of how
        many layers are in hand, which is what lets a single network write all
        of them.
        """

        h = self.embed[0](codes[..., 0])                             # (B, T, emb_dim)
        for j in range(1, codes.shape[-1]):
            h = h + self.embed[j](codes[..., j])

        return h                                                     # (B, T, emb_dim)

    def layer_cond(self, layer: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """The AdaLN conditioning vector for "write layer ``layer``"."""

        index = torch.full((batch_size,), layer - 1, dtype=torch.long, device=device)

        return self.layer_embed(index)                               # (B, cond_dim)

    def _check_layer(self, layer: int, given: int) -> None:
        if not 1 <= layer < self.num_layers:
            raise ValueError(
                f"layer must be in [1, {self.num_layers - 1}], got {layer}"
            )
        if given != layer:
            raise ValueError(
                f"writing layer {layer} needs exactly the {layer} layers below it, "
                f"got {given}"
            )

    def forward(
        self,
        codes: torch.Tensor,                                         # (B, T, layer) long
        layer: int,                                                  # which layer to write
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
    ) -> torch.Tensor:
        """
        Score every frame of one codec layer, given every layer below it.
        """

        if codes.dim() != 3:
            raise ValueError(
                f"expected a (B, T, layers) code stack, got {tuple(codes.shape)}"
            )
        if context is None and text is None:
            raise ValueError("provide either `text` or a precomputed `context`")
        self._check_layer(layer, codes.shape[-1])

        h = self.embed_codes(codes)                                  # (B, T, emb_dim)
        if self.in_proj is not None:
            h = self.in_proj(h)                                      # (B, T, hidden_dim)

        # Text conditioning (bidirectional on both sides; nothing here is decoded
        # step by step, so neither stream has anything to hide from the other).
        ctx = context if context is not None else self.encode_text(text, text_padding_mask)
        cond = self.layer_cond(layer, codes.shape[0], codes.device)  # (B, cond_dim)

        h = self.encoder(h, ctx, padding_mask, text_padding_mask, cond)

        return self.heads[layer - 1](h)                              # (B, T, CODEBOOK_SIZE)

    # -----------
    # Generation
    # -----------

    @staticmethod
    def _sample_layer(
        logits: torch.Tensor,                                        # (B, T, vocab)
        temperature: float,
        top_k: int,
    ) -> torch.Tensor:
        """
        Draw one whole layer at once.

        Greedy by default, and that is the sensible default here rather than a
        conservative one: the acoustic residual is nearly determined by the
        layers below it, so a sampled draw mostly adds noise the codec then
        renders as noise. Temperature is exposed because "mostly" is not "only".
        """

        if temperature == 0.0:
            return logits.argmax(dim=-1)                             # (B, T)

        logits = logits / temperature
        if top_k > 0:
            kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        B, T, V = logits.shape
        draw = torch.multinomial(logits.reshape(-1, V).softmax(dim=-1), num_samples=1)

        return draw.view(B, T)                                       # (B, T)

    @torch.no_grad()
    def generate(
        self,
        codes: torch.Tensor,                                         # (B, T) or (B, T, L) long
        text: Optional[torch.Tensor] = None,                         # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                 # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        context: Optional[torch.Tensor] = None,                      # (B, S, d_text) or None
        temperature: float = 0.0,
        top_k: int = 0,
        max_layer: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Fill the acoustic layers above what was handed in.

        Takes the AR stage's layer-0 tokens — or any prefix of the stack — and
        returns ``(B, T, num_layers)``, ready to hand to ``mimi.decode`` once
        transposed. One forward per layer, no time-stepping and no caches, so
        the whole thing costs ``num_layers - 1`` passes whatever the length.
        """

        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {top_k}")

        if codes.dim() == 2:
            codes = codes[..., None]                                 # (B, T, 1)
        if codes.dim() != 3:
            raise ValueError(
                f"expected (B, T) or (B, T, layers) codes, got {tuple(codes.shape)}"
            )

        target = self.num_layers if max_layer is None else max_layer
        if not codes.shape[-1] < target <= self.num_layers:
            raise ValueError(
                f"max_layer must be in ({codes.shape[-1]}, {self.num_layers}], got {target}"
            )

        was_training = self.training
        self.eval()

        ctx = context if context is not None else self.encode_text(text, text_padding_mask)

        for layer in range(codes.shape[-1], target):
            logits = self.forward(
                codes, layer, padding_mask=padding_mask,
                text_padding_mask=text_padding_mask, context=ctx,
            )                                                        # (B, T, vocab)
            nxt = self._sample_layer(logits, temperature, top_k)     # (B, T)

            # A padded frame carries the pad id on every layer, which is what the
            # model read below and what the collate wrote — so the stack it reads
            # back on the next layer is the one it was trained on.
            if padding_mask is not None:
                nxt = torch.where(
                    padding_mask, nxt, torch.full_like(nxt, config.prosody_pad)
                )
            codes = torch.cat([codes, nxt[..., None]], dim=-1)       # (B, T, layer + 1)

        if was_training:
            self.train()

        return codes                                                 # (B, T, target)
