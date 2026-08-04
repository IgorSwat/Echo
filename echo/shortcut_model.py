from echo import config

from echo.modules.conv import ConvNeXtBlock

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoShortcut(nn.Module):
    """
    A shortcut model to quickly transform Mimi codec representation into
    continous Blue codec latent.

    It's called 'shortcut', because it replaces the more expensive pipeline of 
    Mimi decode() -> audio -> Blue encode().
    """

    # Number of stacked prosody token layers (x[..., 0] and x[..., 1]).
    NUM_TOKEN_LAYERS = 2

    def __init__(self) -> None:
        super().__init__()

        cfg = config.shortcut_model

        emb_dim = cfg.emb_dim
        d_out = cfg.d_out

        # Kernel size + upsampling factor applied after each convolution part.
        # The total upsample should be 86 Hz (Blue) / 12.5 Hz (Mimi) = 6.88.
        self.stage_specs = tuple(cfg.stages)

        # Index of the final interpolating stage; it is the one that rounds up.
        self._last_upsample = max(
            (i for i, (_, f) in enumerate(self.stage_specs) if f != 1.0), default=-1
        )

        self.emb_dim = emb_dim
        self.d_out = d_out
        self.vocab_size = config.prosody_vocab_size

        # Token layers are embedded separately and concatenated; the hidden
        # width stays at 2*emb_dim through the whole trunk.
        self.hidden_dim = self.NUM_TOKEN_LAYERS * emb_dim

        # Token embeddings (one table per prosody layer)
        self.embed = nn.ModuleList([
            nn.Embedding(self.vocab_size, emb_dim)
            for _ in range(self.NUM_TOKEN_LAYERS)
        ])

        # Upsampling trunk
        self.stages = nn.ModuleList([
            nn.ModuleList([
                ConvNeXtBlock(
                    self.hidden_dim, self.hidden_dim,
                    kernel_size=kernel_size,
                    dropout=cfg.dropout,
                    mode="bidirectional",
                )
                for _ in range(cfg.blocks_per_stage)
            ])
            for kernel_size, _ in self.stage_specs
        ])

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, d_out)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in self.embed:
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _upsample(x: torch.Tensor, factor: float, round_up: bool = False) -> torch.Tensor:
        """
        Interpolate along time. F.interpolate's bilinear mode is 2D-only, so the
        stream is viewed as a height-1 image and only the width (time) is scaled.

        With `round_up`, the target length is computed here as ceil(T*factor)
        and passed explicitly, instead of letting F.interpolate floor it (so
        e.g. 372 * 1.72 = 639.84 gives 640 rather than 639).
        """

        y = x.transpose(1, 2).unsqueeze(2)                                # (B, D, 1, T)
        if round_up:
            length = math.ceil(x.shape[1] * factor)
            y = F.interpolate(
                y, size=(1, length), mode="bilinear", align_corners=False,
            )                                                             # (B, D, 1, L)
        else:
            y = F.interpolate(
                y, scale_factor=(1.0, factor), mode="bilinear", align_corners=False,
            )                                                             # (B, D, 1, L)

        return y.squeeze(2).transpose(1, 2)                                # (B, L, D)

    def forward(
        self,
        x: torch.Tensor,                                                  # (B, T, 2) long
    ) -> torch.Tensor:
        if x.shape[-1] != self.NUM_TOKEN_LAYERS:
            raise ValueError(
                f"expected {self.NUM_TOKEN_LAYERS} token layers, got {x.shape[-1]}"
            )

        # Each token layer gets its own table; the two embeddings are concatenated
        # along the feature dim rather than summed, so the trunk can tell them apart.
        h = torch.cat(
            [emb(x[..., i]) for i, emb in enumerate(self.embed)], dim=-1
        )                                                                 # (B, T, hidden_dim)

        for i, (blocks, (_, factor)) in enumerate(zip(self.stages, self.stage_specs)):
            for block in blocks:
                h = block(h)                                              # (B, T_s, hidden_dim)
            if factor != 1.0:
                # The last interpolation rounds the length up.
                h = self._upsample(h, factor, round_up=i == self._last_upsample)

        return self.out_proj(h)                                           # (B, L, d_out)
