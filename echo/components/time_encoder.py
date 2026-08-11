from echo.nn.init import init_weights_

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """
    Sinusoidal timestep embedding followed by a Linear-SiLU-Linear MLP.
    """

    def __init__(self, dim: int, max_period: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

        init_weights_(self.net)

    def sinusoidal_embedding(
        self, t: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device, dtype=dtype))
            * torch.arange(half, device=device, dtype=dtype) / half
        )                                                           # (half,)
        args = t[:, None].float() * freqs[None, :]                  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, 2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))                                # (B, dim)

        return emb.to(dtype=dtype)                                  # (B, dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:             # (B,) long/float
        emb = self.sinusoidal_embedding(t, t.device, t.dtype)       # (B, dim)

        return self.net(emb)                                        # (B, dim)
