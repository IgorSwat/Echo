from echo import config

from echo.modules.conformer import Conformer

from typing import Optional

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Text encoder: token embeddings -> Conformer -> optional projection to D_out.
    """

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
    ) -> None:
        super().__init__()

        self.embed = nn.Embedding(vocab_size, d_model)

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
        )

		# Additional linear projection in case the output dimension should be different.
        self.proj = nn.Linear(d_model, d_out) if d_out != d_model else None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=config.init_std)
        if self.proj is not None:
            nn.init.normal_(self.proj.weight, mean=0.0, std=config.init_std)
            nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T) long
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
    ) -> torch.Tensor:
        x = self.embed(x)                                              # (B, T, D)
        x = self.conformer(x, key_padding_mask)                        # (B, T, D)
        
        if self.proj is not None:
            x = self.proj(x)                                           # (B, T, d_out)

        return x                                                     # (B, T, d_out)
