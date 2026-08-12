from echo.nn.init import init_weights_

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    MLP token head, predicting the next prosody token from a frame state.

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
