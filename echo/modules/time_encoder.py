from echo import config

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """
    Sinusoidal timestep embedding followed by an MLP (Linear-SiLU-Linear).
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

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def sinusoidal_embedding(
        self, t: torch.Tensor, device: torch.device, dtype: torch.dtype,
    ) -> torch.Tensor:                                                  # (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, device=device, dtype=dtype))
            * torch.arange(half, device=device, dtype=dtype) / half
        )                                                                # (half,)
        args = t[:, None].float() * freqs[None, :]                       # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)      # (B, 2*half)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))                                      # (B, dim)
        return emb.to(dtype=dtype)

    def forward(self, t: torch.Tensor) -> torch.Tensor:                  # (B,) long/float
        emb = self.sinusoidal_embedding(t, t.device, t.dtype)            # (B, dim)
        
        return self.net(emb)                                              # (B, dim)
