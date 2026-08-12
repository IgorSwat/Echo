from echo import config

from echo.nn.conformer import Conformer
from echo.nn.sequence import merge_padded, valid_mask

from typing import Optional

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Text encoder: token embeddings -> Conformer -> optional projection to d_out.
    """

    # Segment ids for the additive embedding; the separator has neither.
    SEGMENT_REFERENCE = 0
    SEGMENT_TARGET = 1
    NUM_SEGMENTS = 2

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_out: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        conv_use_norm: bool = True,
        ffn_glu: bool = False,
        max_seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.segment = nn.Embedding(self.NUM_SEGMENTS, d_model)

        self.conformer = Conformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_rope=use_rope,
            conv_use_norm=conv_use_norm,
            ffn_glu=ffn_glu,
            max_seq_len=max_seq_len,
        )

        self.proj = nn.Linear(d_model, d_out) if d_out != d_model else None

        self._init_weights()

    def _init_weights(self) -> None:
        # Only the embeddings and projection: the Conformer initialized itself.
        for emb in (self.embed, self.segment):
            nn.init.normal_(emb.weight, mean=0.0, std=config.init_std)
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.proj.bias)

    def _pair(
        self,
        x: torch.Tensor,                                            # (B, S) long
        key_padding_mask: Optional[torch.Tensor],                   # (B, S) or None
        ref: torch.Tensor,                                          # (B, S_ref) long
        ref_key_padding_mask: Optional[torch.Tensor],               # (B, S_ref) or None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build ``[ref] <sep> [x]`` with the per-token segment ids to match.
        """

        B, S = x.shape
        device = x.device

        # The separator rides along on the target half, which puts it at index
        # `offset` in every row once the two are compacted together.
        sep = torch.full((B, 1), config.text_sep, dtype=x.dtype, device=device)
        suffix = torch.cat([sep, x], dim=1)                         # (B, S + 1)
        suffix_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=device),
            valid_mask(key_padding_mask, B, S, device),
        ], dim=1)                                                   # (B, S + 1)

        merged, mask, offset = merge_padded(
            ref, ref_key_padding_mask, suffix, suffix_mask, fill=config.text_pad,
        )                                                           # (B, L), (B, L), (B,)

        pos = torch.arange(merged.shape[1], device=device)[None, :]  # (1, L)
        segment = torch.where(
            pos < offset[:, None], self.SEGMENT_REFERENCE, self.SEGMENT_TARGET
        )                                                           # (B, L)
        # The separator sits exactly at the boundary and belongs to neither half;
        # the padded tail belongs to neither either. Both are marked -1.
        plain = (pos == offset[:, None]) | ~mask                    # (B, L)

        return merged, mask, torch.where(plain, -1, segment)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, S) long
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) or None
        ref: Optional[torch.Tensor] = None,                         # (B, S_ref) long or None
        ref_key_padding_mask: Optional[torch.Tensor] = None,        # (B, S_ref) or None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode one transcript, or a reference/target pair as a single sequence.

        Returns the encoding and the key-padding mask that goes with it -- in
        paired mode the sequence is longer than either input, so the caller
        cannot reuse the mask it passed in. The reference half occupies
        ``ref_key_padding_mask.sum(1)`` positions, with the separator just after.
        """

        segment: Optional[torch.Tensor] = None
        if ref is not None:
            x, key_padding_mask, segment = self._pair(
                x, key_padding_mask, ref, ref_key_padding_mask,
            )

        h = self.embed(x)                                           # (B, L, d_model)

        if segment is not None:
            # -1 marks the separator and the padded tail: token embedding only.
            h = h + torch.where(
                (segment >= 0)[..., None],
                self.segment(segment.clamp_min(0)),
                torch.zeros((), dtype=h.dtype, device=h.device),
            )

        h = self.conformer(h, key_padding_mask)                     # (B, L, d_model)

        if self.proj is not None:
            h = self.proj(h)                                        # (B, L, d_out)

        return h, key_padding_mask                                  # (B, L, d_out), (B, L)
