import torch
import torch.nn as nn
import torch.nn.functional as F

from echo import config


# Single prediction head component.
# Sort of deprecated, since multi-head version is now vectorized with it's own implementation.
class PredictionHead(nn.Module):
    """
    A small MLP producing codec token logits for given codec layer from a hidden representation.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int ,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_layers = num_layers

        # (num_layers - 1) hidden layers + 1 output layer
        layers: list[nn.Module] = []
        if num_layers == 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Maps ``(..., in_dim)`` to ``(..., out_dim)`` logits."""
        
        return self.net(hidden)


# Multiple (stacked) prediction heads component - with no vectorization across heads.
class PredictionMultihead(nn.Module):
    """
    A stack of ``num_heads`` prediction heads (MLPs), one per codebook layer.
    """

    def __init__(
        self,
        num_heads: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList(
            [PredictionHead(in_dim, hidden_dim, out_dim, num_layers, dropout) for _ in range(num_heads)]
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``(..., in_dim)`` -> ``(..., num_heads, out_dim)`` logits."""

        # Run every head and stack along a new codebook-axis.
        return torch.stack([head(hidden) for head in self.heads], dim=-2)
    

# Multiple prediction heads component - vectorized version.
# Benchmarks suggest this version leads to ~10% improvement of decoding speed on Apple Silicon.
class FusedPredictionMultihead(nn.Module):
    """Vectorized prediction heads. One batched matmul per layer instead of N separate forward passes."""

    def __init__(
        self,
        num_heads: int,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        if num_layers == 1:
            # (num_heads, out_dim, in_dim)
            self.weights.append(nn.Parameter(torch.empty(num_heads, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.empty(num_heads, out_dim)))
        else:
            # Layer 1: in_dim → hidden_dim
            self.weights.append(nn.Parameter(torch.empty(num_heads, hidden_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.empty(num_heads, hidden_dim)))
            # Hidden layers: hidden_dim → hidden_dim
            for _ in range(num_layers - 2):
                self.weights.append(nn.Parameter(torch.empty(num_heads, hidden_dim, hidden_dim)))
                self.biases.append(nn.Parameter(torch.empty(num_heads, hidden_dim)))
            # Output layer: hidden_dim → out_dim
            self.weights.append(nn.Parameter(torch.empty(num_heads, out_dim, hidden_dim)))
            self.biases.append(nn.Parameter(torch.empty(num_heads, out_dim)))

        self._init_weights()

    def _init_weights(self) -> None:
        for w, b in zip(self.weights, self.biases):
            nn.init.normal_(w, mean=0.0, std=config.init_std)
            nn.init.zeros_(b)

    def forward(
        self,
        hidden: torch.Tensor,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(..., in_dim) → (..., num_heads, out_dim)."""
        N = self.num_heads
        # Broadcast hidden across the head dimension
        x = hidden.unsqueeze(-2).expand(*hidden.shape[:-1], N, hidden.shape[-1])   # (..., N, in_dim)
        if conditioning is not None:
            if conditioning.shape != x.shape:
                raise ValueError(f"expected conditioning shape {tuple(x.shape)}, got {tuple(conditioning.shape)}")
            x = x + conditioning

        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            # w: (N, out_f, in_f),  x: (..., N, in_f)  →  (..., N, out_f)
            x = torch.einsum("...ni,noi->...no", x, w) + b
            if i < self.num_layers - 1:     # no activation after the output layer
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)

        return x

    def forward_head(
        self,
        hidden: torch.Tensor,
        head: int,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run one codebook head for sequential within-frame decoding."""
        x = hidden if conditioning is None else hidden + conditioning
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = F.linear(x, w[head], b[head])
            if i < self.num_layers - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x
