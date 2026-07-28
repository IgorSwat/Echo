from echo import config

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedConv(nn.Module):
    """
    Depthwise-separable convolution module with a GLU pointwise expansion,
    as used in the Conformer architecture.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        use_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.LayerNorm(d_model)
            
        self.pw1 = nn.Linear(d_model, 2 * d_model)                       # pointwise expand
        self.dw = nn.Conv1d(                                             # depthwise
            d_model, d_model, kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.pw2 = nn.Linear(d_model, d_model)                           # pointwise project
        
        self.drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:                  # (B, T, D)
        # Optional norm
        if self.use_norm:
            x = self.norm(x)                                             # (B, T, D)

		# As proposed in "Language Modeling with Gated Convolutional Networks",
        # we use GLU as a form of selective channel mixing.
        x = F.glu(self.pw1(x), dim=-1)                                   # (B, T, D)
        x = self.dw(x.transpose(1, 2)).transpose(1, 2)                   # (B, T, D)
        x = F.gelu(x)                                                    # (B, T, D)
        
        return self.drop(self.pw2(x))                                    # (B, T, D)
