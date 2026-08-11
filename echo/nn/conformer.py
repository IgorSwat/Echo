from echo.nn.attention import SelfAttention
from echo.nn.conv import GatedConv
from echo.nn.ffn import FeedForward
from echo.nn.norm import ConditionalLayerNorm

from typing import Optional

import torch
import torch.nn as nn


class ConformerBlock(nn.Module):
    """
    Conformer block: macaron-style half-step FFNs around self-attention and a
    gated convolution, each with its own residual connection.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        conv_use_norm: bool = True,
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
        ffn_glu: bool = False,
        max_seq_len: Optional[int] = None,
        mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.ffn1 = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)
        self.norm_attn = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)
        self.attn = SelfAttention(
            d_model, num_heads, dropout,
            use_rope=use_rope, max_seq_len=max_seq_len, mode=mode,
        )
        self.conv = GatedConv(
            d_model, kernel_size,
            use_norm=conv_use_norm, dropout=dropout, mode=mode,
        )
        self.ffn2 = FeedForward(d_model, ffn_dim, dropout, use_glu=ffn_glu)
        self.norm_out = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)                                  # (B, T, D)
        h, g = self.norm_attn(x, cond)                              # (B, T, D), (B, D)
        attn, _ = self.attn(h, key_padding_mask)                    # cache unused: bidirectional
        x = x + g[:, None, :] * attn                                # (B, T, D)
        x = x + self.conv(x, key_padding_mask)                      # (B, T, D)
        x = x + 0.5 * self.ffn2(x)                                  # (B, T, D)
        x, _ = self.norm_out(x, cond)                               # (B, T, D)

        return x                                                    # (B, T, D)


class Conformer(nn.Module):
    """
    Stack of Conformer blocks followed by one final norm.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        kernel_size: int,
        dropout: float = 0.0,
        use_rope: bool = False,
        conv_use_norm: bool = True,
        use_ada_ln: bool = False,
        cond_dim: Optional[int] = None,
        ffn_glu: bool = False,
        max_seq_len: Optional[int] = None,
        mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        self.use_ada_ln = use_ada_ln

        self.blocks = nn.ModuleList([
            ConformerBlock(
                d_model, num_heads, ffn_dim, kernel_size,
                dropout, use_rope, conv_use_norm,
                use_ada_ln, cond_dim, ffn_glu, max_seq_len,
                mode=mode,
            )
            for _ in range(num_layers)
        ])

        self.norm = ConditionalLayerNorm(d_model, cond_dim, use_ada_ln=use_ada_ln)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, D)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        cond: Optional[torch.Tensor] = None,                        # (B, cond_dim) or None
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, key_padding_mask, cond)                    # (B, T, D)
        x, _ = self.norm(x, cond)                                   # (B, T, D)

        return x                                                    # (B, T, D)
