from echo import config

from echo.modules.conv import GatedConv
from echo.modules.text_encoder import TextEncoder
from echo.modules.transformer import HybridAttentionDecoder

from typing import Optional

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


class EchoAR(nn.Module):
    """
    EchoAR: text-conditioned autoregressive model over two layers of prosody tokens.

    Pipeline:
      1. Embed both token layers with separate tables and concatenate
         -> (B, T, 2*emb_dim).
      2. Encode text -> (B, S, d_text) cross-attention context.
      3. Causal GatedConv stack over the token embeddings (residual per layer).
      4. Project to hidden_dim (when needed) and run the causal
         HybridAttentionDecoder against the text context -> H (B, T, hidden_dim).
      5. One MLP head per token layer maps H to logits -> (B, T, 2, vocab).

    Everything on the token stream is causal, so position i is built from
    positions <= i only and the whole model can be trained with teacher forcing
    in a single pass.
    """

    # Number of stacked prosody token layers (x[..., 0] and x[..., 1]).
    NUM_TOKEN_LAYERS = 2

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

        # --- Convolutional front-end over the token embeddings ---
        self.convs = nn.ModuleList([
            GatedConv(
                d_tokens, cfg.conv_kernel_size,
                use_norm=cfg.conv_use_norm,
                dropout=cfg.conv_dropout,
                mode="causal",
            )
            for _ in range(cfg.conv_num_layers)
        ])

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

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, 2) long
        text: Optional[torch.Tensor] = None,                        # (B, S) long or None
        padding_mask: Optional[torch.Tensor] = None,                # (B, T) or None
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
        context: Optional[torch.Tensor] = None,                     # (B, S, d_text) or None
    ) -> torch.Tensor:
        if x.shape[-1] != self.NUM_TOKEN_LAYERS:
            raise ValueError(
                f"expected {self.NUM_TOKEN_LAYERS} token layers, got {x.shape[-1]}"
            )
        if context is None and text is None:
            raise ValueError("provide either `text` or a precomputed `context`")

        # Each token layer gets its own table; the two embeddings are concatenated
        # along the feature dim rather than summed, so the decoder can tell them apart.
        h = torch.cat(
            [emb(x[..., i]) for i, emb in enumerate(self.embed)], dim=-1
        )                                                            # (B, T, 2*emb_dim)

        # Text conditioning (bidirectional; the text is fully known up front).
        ctx = context if context is not None else self.encode_text(text, text_padding_mask)

        # Convolutional front-end, residual per layer as in ConformerBlock.
        for conv in self.convs:
            h = h + conv(h)                                          # (B, T, 2*emb_dim)

        if self.in_proj is not None:
            h = self.in_proj(h)                                      # (B, T, hidden_dim)

        h = self.decoder(h, ctx, padding_mask, text_padding_mask)    # (B, T, hidden_dim)

        # One head per token layer, stacked to mirror the input layout.
        return torch.stack([head(h) for head in self.heads], dim=2)  # (B, T, 2, vocab)

    @torch.no_grad()
    def generate(
        self,
        text: torch.Tensor,                                         # (B, S) long
        text_padding_mask: Optional[torch.Tensor] = None,           # (B, S) or None
        max_frames: int = 1000,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding from text alone.
        
        Returns ``(B, T, NUM_TOKEN_LAYERS)`` of codec token ids.
        """
        was_training = self.training
        self.eval()

        B = text.shape[0]
        ctx = self.encode_text(text, text_padding_mask)              # (B, S, d_text)

        x = torch.full(
            (B, 1, self.NUM_TOKEN_LAYERS), config.prosody_bos,
            dtype=torch.long, device=text.device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=text.device)

        for _ in range(max_frames):
            logits = self(x, context=ctx, text_padding_mask=text_padding_mask)
            logits = logits[:, -1]                                   # (B, layers, vocab)
            logits[..., config.prosody_bos] = float("-inf")
            logits[..., config.prosody_pad] = float("-inf")

            nxt = logits.argmax(dim=-1)                              # (B, layers)
            finished = finished | (nxt[:, 0] == config.prosody_eos)
            if bool(finished.all()):
                break

            # Already-terminated rows contribute padding from here on.
            nxt = torch.where(
                finished[:, None], torch.full_like(nxt, config.prosody_pad), nxt
            )
            x = torch.cat([x, nxt[:, None, :]], dim=1)               # (B, t + 1, layers)

        if was_training:
            self.train()

        return x[:, 1:]                                              # (B, T, layers), BOS dropped
