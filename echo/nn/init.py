from echo.nn.norm import AdaLN

import torch.nn as nn


def init_weights_(module: nn.Module) -> None:
    """
    Initialize every Linear/Conv1d within given module in place.

    Two things this does *not* do, both of which look like details and are not:

    It does not use a fixed std. A constant std is a per-layer gain of
    ``std * sqrt(fan_in)``, so it attenuates narrow layers and amplifies wide
    ones, and the error compounds along the depth. Scaling by fan-in makes every
    layer unit-gain by construction.

    It does not touch :class:`AdaLN`. Its output projection is deliberately
    zeroed so a residual branch starts closed and opens only as the gate learns
    to; re-drawing it from the same normal as everything else silently turns
    AdaLN-Zero back into ordinary AdaLN. A block that builds its norms before
    calling this would otherwise undo them, which is easy to do and invisible
    afterwards.

    NOTE: Recurses into submodules.
    """

    inside_adaln = {
        id(sub) for m in module.modules() if isinstance(m, AdaLN) for sub in m.modules()
    }

    for m in module.modules():
        if id(m) in inside_adaln:
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            nn.init.normal_(m.weight, mean=0.0, std=max(fan_in, 1) ** -0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
