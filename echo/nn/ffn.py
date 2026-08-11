from echo.nn.init import init_weights_

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network for transformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_glu: bool = False,
    ) -> None:
        super().__init__()

        self.use_glu = use_glu
        if use_glu:
            self.gate_up = nn.Linear(d_model, 2 * ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)
        else:
            self.up = nn.Linear(d_model, ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)

        self.drop = nn.Dropout(dropout)

        init_weights_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # (B, T, d_model)
        if self.use_glu:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            x = F.gelu(gate) * up
        else:
            x = F.gelu(self.up(x))

        return self.drop(self.down(x))                              # (B, T, d_model)
