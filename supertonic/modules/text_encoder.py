from supertonic import config

from supertonic.modules import apply_mask
from supertonic.modules.conv import ConvNeXtStack
from supertonic.modules.transformer import AttnEncoder

from typing import Optional, Sequence

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Supertonic TTL text encoder: token embeddings -> dilated ConvNeXt stack ->
    VITS rel-pos attention encoder, with the conv output added back residually
    (a strong local signal alongside the global context).

    The style-prompted stage of the original text_encoder.onnx is omitted: the
    vector estimator carries its own style conditioning.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        intermediate_dim: int,
        kernel_size: int,
        dilations: Sequence[int],
        num_heads: int,
        num_layers: int,
        window_size: int = 4,
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)
        self.convnext = ConvNeXtStack(d_model, intermediate_dim, kernel_size, dilations)
        self.attn_encoder = AttnEncoder(d_model, intermediate_dim, num_heads, num_layers, window_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=config.init_std)

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, S) bool or None
    ) -> torch.Tensor:
        x = apply_mask(self.embed(text), key_padding_mask)          # (B, S, D)
        conv = self.convnext(x, key_padding_mask)                   # (B, S, D)
        attn = self.attn_encoder(conv, key_padding_mask)            # (B, S, D)

        return apply_mask(attn + conv, key_padding_mask)            # (B, S, D)
