import torch
import torch.nn as nn

from echo import config


class PredictionHead(nn.Module):
    """
    A small MLP producing codec token logits from a hidden representation.
    """

    def __init__(
        self,
        in_dim: int = config.D_REPR,
        hidden_dim: int = config.PRED_HIDDEN_DIM,
        out_dim: int = config.CODEC_LOGIT_DIM,
        num_layers: int = config.PRED_NUM_LAYERS,
        dropout: float = config.PRED_DROPOUT,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            # First layer expands the representation to the hidden width.
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            # Final layer maps the hidden width down to the logit width.
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=config.INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``(..., in_dim)`` to ``(..., out_dim)`` logits."""
        
        return self.net(hidden)
