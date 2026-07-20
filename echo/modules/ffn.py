from echo import config

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (GELU-MLP or GLU variant) for Transformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        dropout: float = 0.0,
        use_glu: bool = False,
    ) -> None:
        super().__init__()

		# Using GLU increases number of parameters by ~50%.
        self.use_glu = use_glu
        if use_glu:
            self.gate_up = nn.Linear(d_model, 2 * ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)
        else:
            self.up = nn.Linear(d_model, ffn_dim)
            self.down = nn.Linear(ffn_dim, d_model)

        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_glu:
            gate, up = self.gate_up(x).chunk(2, dim=-1)
            x = F.gelu(gate) * up
        else:
            x = F.gelu(self.up(x))
            
        return self.drop(self.down(x))