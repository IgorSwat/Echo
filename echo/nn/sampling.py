from echo.nn.init import init_weights_

import torch
import torch.nn as nn
import torch.nn.functional as F


class Downsample2x(nn.Module):
    """
    Halves the time axis and double the channels, with a stride-2 convolution.
    """

    def __init__(self, dim_in: int, kernel_size: int = 3) -> None:
        super().__init__()

        self.conv = nn.Conv1d(
            dim_in, 2 * dim_in, kernel_size,
            stride=2, padding=kernel_size // 2,
        )

        init_weights_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # (B, T, dim_in)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)         # (B, T//2, 2*dim_in)


class Upsample2x(nn.Module):
    """
    Double the time axis and halve the channels, with interpolation + convolution
    """

    def __init__(self, dim_in: int, mode: str = "nearest", kernel_size: int = 3) -> None:
        super().__init__()
        self.mode = mode

        self.conv = nn.Conv1d(
            dim_in, dim_in // 2, kernel_size,
            padding=kernel_size // 2,
        )

        init_weights_(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:             # (B, T//2, 2*dim)
        x = F.interpolate(
            x.transpose(1, 2),
            size=x.shape[1] * 2,
            mode=self.mode,
        )                                                           # (B, D, T)

        return self.conv(x).transpose(1, 2)                         # (B, T, D//2)
