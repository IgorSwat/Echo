from echo import config

import torch.nn as nn


def init_weights_(module: nn.Module) -> None:
    """
    Initialize every Linear/Conv1d within given module in place.
    NOTE: Recurses into submodules.
    """

    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
