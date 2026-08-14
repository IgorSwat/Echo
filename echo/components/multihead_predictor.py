from echo.nn.init import init_weights_

from typing import Optional

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    MLP token head, predicting one layer of the output codebook.

    NOTE: it includes (num_layers - 1) hidden layers and 1 output layer
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

        init_weights_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:              # (B, T, d_in)
        return self.net(x)                                           # (B, T, d_out)


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


class MultiHeadPredictor(nn.Module):
    """
    One MLP head per token layer, over a shared frame state.

    With `cond_dim` set, head k > 0 is FiLM-modulated by layer k - 1 of the same
    frame: P(c0 | h) * P(c1 | c0, h) rather than layers independent given h.

    NOTE: conditioning arrives already embedded; the caller owns the token tables.
    """

    def __init__(
        self,
        num_heads: int,
        d_in: int,
        d_hidden: int,
        d_out: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        cond_dim: Optional[int] = None,                              # None disables FiLM
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.num_heads = num_heads

        self.heads = nn.ModuleList([
            PredictionHead(d_in, d_hidden, d_out, num_layers=num_layers, dropout=dropout)
            for _ in range(num_heads)
        ])

        # One FiLM per head above the first, so a single head has none: there
        # is no layer below it to condition on, and `uses_cond` says so.
        self.film = nn.ModuleList([
            IntraFrameFiLM(d_in, cond_dim) for _ in range(num_heads - 1)
        ]) if cond_dim is not None and num_heads > 1 else None

    @property
    def uses_cond(self) -> bool:
        """Whether the heads above the first need intra-frame conditioning."""

        return self.film is not None

    def head(
        self,
        layer: int,
        h: torch.Tensor,                                             # (B, T, d_in)
        cond_emb: Optional[torch.Tensor] = None,                     # (B, T, cond_dim) or None
    ) -> torch.Tensor:
        """
        Logits for one token layer, optionally modulated by the layer below it.
        """

        if self.film is not None and layer > 0:
            if cond_emb is None:
                raise ValueError(
                    f"head {layer} needs layer {layer - 1}'s embedded tokens when "
                    "intra-frame conditioning is enabled"
                )
            h = self.film[layer - 1](h, cond_emb)

        return self.heads[layer](h)                                  # (B, T, d_out)

    def forward(
        self,
        h: torch.Tensor,                                             # (B, T, d_in)
        cond_emb: Optional[torch.Tensor] = None,                     # (B, T, heads-1, cond_dim)
    ) -> torch.Tensor:
        """
        Every head at once, stacked to mirror the token layout.
        """

        return torch.stack(
            [
                self.head(k, h, None if cond_emb is None or k == 0 else cond_emb[:, :, k - 1])
                for k in range(self.num_heads)
            ],
            dim=2,
        )                                                            # (B, T, heads, d_out)
