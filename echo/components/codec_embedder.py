from echo import config

from echo.nn.sequence import merge_padded, valid_mask

from typing import Optional

import torch
import torch.nn as nn


class CodecEmbedder(nn.Module):
    """
    Embeds a frame sequence of stacked prosody-token layers.
    """

    SEGMENT_REFERENCE = 0
    SEGMENT_TARGET = 1
    NUM_SEGMENTS = 2

    def __init__(self, num_token_layers: int, vocab_size: int, emb_dim: int) -> None:
        super().__init__()

        self.num_token_layers = num_token_layers
        self.emb_dim = emb_dim
        self.d_out = num_token_layers * emb_dim

        self.embed = nn.ModuleList([
            nn.Embedding(vocab_size, emb_dim) for _ in range(num_token_layers)
        ])
        self.segment = nn.Embedding(self.NUM_SEGMENTS, self.d_out)

        self._init_weights()

    def _init_weights(self) -> None:
        for emb in (*self.embed, self.segment):
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)

    def embed_layer(self, layer: int, tokens: torch.Tensor) -> torch.Tensor:
        """
        One token layer through its own table -- the alphabet the decoder reads.
        """

        return self.embed[layer](tokens)                             # (..., emb_dim)

    def embed_frames(self, x: torch.Tensor) -> torch.Tensor:
        """
        Stacked token layers, concatenated into one vector per frame.
        """

        if x.shape[-1] != self.num_token_layers:
            raise ValueError(
                f"expected {self.num_token_layers} token layers, got {x.shape[-1]}"
            )

        return torch.cat(
            [emb(x[..., i]) for i, emb in enumerate(self.embed)], dim=-1
        )                                                            # (B, T, d_out)

    def forward(
        self,
        x: torch.Tensor,                                             # (B, T, L) long
        key_padding_mask: Optional[torch.Tensor] = None,             # (B, T) or None
        ref: Optional[torch.Tensor] = None,                          # (B, T_ref, L) or None
        ref_key_padding_mask: Optional[torch.Tensor] = None,         # (B, T_ref) or None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Embed a frame sequence, optionally behind a reference prefix.
        """

        segment: Optional[torch.Tensor] = None

        if ref is not None:
            B, T = x.shape[:2]
            device = x.device

            x, key_padding_mask, offset = merge_padded(
                ref, ref_key_padding_mask,
                x, valid_mask(key_padding_mask, B, T, device),
                fill=config.prosody_pad,
            )                                                        # (B, F, L), (B, F), (B,)

            pos = torch.arange(x.shape[1], device=device)[None, :]    # (1, F)
            segment = torch.where(
                pos < offset[:, None], self.SEGMENT_REFERENCE, self.SEGMENT_TARGET
            )

            # The separating BOS frame is the first frame of the target half, and
            # it belongs to neither; nor does the padded tail.
            plain = (pos == offset[:, None]) | ~key_padding_mask
            segment = torch.where(plain, -1, segment)                # (B, F)

        h = self.embed_frames(x)                                     # (B, F, d_out)

        if segment is not None:
            h = h + torch.where(
                (segment >= 0)[..., None],
                self.segment(segment.clamp_min(0)),
                torch.zeros((), dtype=h.dtype, device=h.device),
            )

        return h, key_padding_mask                                   # (B, F, d_out), (B, F)
