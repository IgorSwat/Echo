from echo.nn.conv import ConvNeXtBlock
from echo.nn.init import init_weights_
from echo.nn.transformer_blocks import HybridAttentionBlock

from typing import Optional

import torch
import torch.nn as nn


class DualStreamBlock(nn.Module):
    """
    Two persistent streams at different rates, coupled once per block.

    Stream A carries the latent's own 86.13 Hz grid at width ``dim_a`` and is
    processed by ConvNeXt blocks. Stream B carries a quarter-rate 21.53 Hz grid
    at width ``dim_b`` and is processed by one hybrid attention block. Both
    persist across the whole stack; neither is re-derived from the other.

    That persistence is the point. An earlier design rebuilt the attention
    stream from the convolution stream in every block -- pack, attend once,
    squeeze the result back through a stride-4 upsample -- which made the
    effective attention depth one, repeated. Measured against a 41M baseline it
    lost 7% uniformly across t, with no bottleneck anywhere to explain it: the
    upsample was full rank, the packing left no artifact at the 21.5 Hz period,
    and the stream was demonstrably carrying load. Shallow computation was what
    was left. Here stream B is a genuine transformer trunk that happens to run
    at a quarter of the rate.

    The two exchange through one projection each way, and both are
    zero-initialized so the streams start independent and the coupling grows
    only as it earns its place:

        a <- Conv(a)  + W_up(b)     transposed convolution, T/4 -> T
        b <- Attn(b)  + W_down(a)   reshape T -> T/4, then a linear map

    The downward direction can use a plain reshape because packing four frames
    into channels discards nothing; the upward direction cannot, so it is a
    learned transposed convolution with overlap.
    """

    FACTOR = 4

    def __init__(
        self,
        dim_a: int,
        dim_b: int,
        cond_dim: int,
        d_kv: int,
        num_heads: int,
        ffn_dim: int,
        num_conv: int = 2,
        kernel_size: int = 7,
        up_kernel: int = 8,
        dropout: float = 0.0,
        use_rope: bool = True,
        rope_norm: str = "query",
        use_ada_ln: bool = True,
        ffn_glu: bool = False,
    ) -> None:
        super().__init__()

        if (up_kernel - self.FACTOR) % 2 != 0:
            raise ValueError(f"up_kernel ({up_kernel}) - {self.FACTOR} must be even")

        self.dim_a = dim_a
        self.dim_b = dim_b

        self.convs = nn.ModuleList([
            ConvNeXtBlock(dim_a, dim_a, kernel_size=kernel_size, use_ada_ln=use_ada_ln,
                          dropout=dropout, cond_dim=cond_dim)
            for _ in range(num_conv)
        ])

        self.attn = HybridAttentionBlock(
            d_model=dim_b, d_kv=d_kv, num_heads=num_heads, ffn_dim=ffn_dim,
            dropout=dropout, use_rope=use_rope, use_ada_ln=use_ada_ln,
            cond_dim=cond_dim, ffn_glu=ffn_glu, rope_norm=rope_norm,
        )

        self.up = nn.ConvTranspose1d(
            dim_b, dim_a, up_kernel, stride=self.FACTOR,
            padding=(up_kernel - self.FACTOR) // 2,
        )
        self.down = nn.Linear(self.FACTOR * dim_a, dim_b)

        # Both couplings start closed. Opening a stream into another at init was
        # measured to cost more than it buys: with the exchange live from step 0
        # an untrained stream writes noise into a stream that would otherwise be
        # learning cleanly. The weights themselves still take gradient on the
        # first step, so nothing is frozen.
        nn.init.zeros_(self.up.weight); nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.down.weight); nn.init.zeros_(self.down.bias)

    def forward(
        self,
        a: torch.Tensor,                                            # (B, T, dim_a)
        b: torch.Tensor,                                            # (B, T/4, dim_b)
        context: torch.Tensor,                                      # (B, S, d_kv)
        cond: torch.Tensor,                                         # (B, cond_dim)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        packed_padding_mask: Optional[torch.Tensor] = None,         # (B, T/4) or None
        context_padding_mask: Optional[torch.Tensor] = None,        # (B, S) or None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for conv in self.convs:
            a = conv(a, cond, key_padding_mask)                     # (B, T, dim_a)
        a = a + self.up(b.transpose(1, 2)).transpose(1, 2)          # (B, T, dim_a)

        # Stream B reads what stream A just produced, so the exchange within a
        # block runs one way then the other rather than both from stale state.
        h = a if key_padding_mask is None else a * key_padding_mask.unsqueeze(-1).to(a.dtype)
        B, T, _ = h.shape
        h = h.reshape(B, T // self.FACTOR, self.FACTOR * self.dim_a)

        b, _ = self.attn(b, context, packed_padding_mask, context_padding_mask, cond)
        b = b + self.down(h)                                        # (B, T/4, dim_b)

        return a, b
