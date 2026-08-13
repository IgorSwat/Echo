from echo.nn.conv import ConvNeXtBlock
from echo.nn.init import init_weights_
from echo.nn.transformer_blocks import HybridAttentionBlock

from typing import Optional

import torch
import torch.nn as nn


class TwoStreamBlock(nn.Module):
    """
    Full-rate convolution beside quarter-rate attention, summed.

    Stream A stays on the latent's own 86.13 Hz grid and runs two ConvNeXt
    blocks over it. Stream B packs four consecutive frames into the channel
    axis, concatenates the prosody embedding on that coarser grid, and runs one
    hybrid attention block there.

    The packing is a reshape, not a pooling, and that is the whole point. A
    U-Net's strided downsample throws the top half of the spectrum away, so
    anything the attention stream wants to say about fine detail has to be
    re-invented on the way back up. Packing keeps every component -- it moves
    them into channels -- so attention sees T/4 positions and pays 1/16 of the
    quadratic term while nothing is discarded, and the transposed convolution
    can put the detail back.

    Two rates also put the prosody where it belongs. Mimi runs at 12.5 Hz
    against the latents' 86.13 Hz, so on the full grid the nearest-neighbour
    stretch repeats each token ~6.9 times; on the quarter grid it repeats ~1.7,
    and the staircase the ConvNeXt blocks used to have to smooth is mostly gone.
    Re-injecting per block is what keeps the conditioning from decaying with
    depth, which concatenating once at the input could not do.
    """

    # Frames packed into the channel axis for the attention stream.
    FACTOR = 4

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        prosody_dim: int,
        cond_dim: int,
        d_kv: int,
        num_heads: int,
        ffn_dim: int,
        pack_dim: Optional[int] = None,
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
            raise ValueError(
                f"up_kernel ({up_kernel}) - {self.FACTOR} must be even so the "
                f"transposed convolution lands back on exactly T frames"
            )

        self.dim_in = dim_in
        self.dim_out = dim_out

        # Packing the full conv stream would tie the attention width to it at
        # 4 * dim_in + prosody_dim. A narrower `pack_dim` decouples the two, so
        # the convolution can carry detail at a width the attention does not
        # have to pay for -- at the cost of stream B seeing a projection rather
        # than the stream itself.
        pack_dim = dim_in if pack_dim is None else min(pack_dim, dim_in)
        self.pack_dim = pack_dim
        self.d_b = pack_dim * self.FACTOR + prosody_dim

        # --- Stream A: full rate ---
        self.conv1 = ConvNeXtBlock(
            dim_in, dim_in, kernel_size=kernel_size, use_ada_ln=use_ada_ln,
            dropout=dropout, cond_dim=cond_dim,
        )
        self.conv2 = ConvNeXtBlock(
            dim_in, dim_out, kernel_size=kernel_size, use_ada_ln=use_ada_ln,
            dropout=dropout, cond_dim=cond_dim,
        )

        # --- Stream B: quarter rate ---
        self.pre_pack = nn.Linear(dim_in, pack_dim) if pack_dim != dim_in else None
        self.attn = HybridAttentionBlock(
            d_model=self.d_b, d_kv=d_kv, num_heads=num_heads, ffn_dim=ffn_dim,
            dropout=dropout, use_rope=use_rope, use_ada_ln=use_ada_ln,
            cond_dim=cond_dim, ffn_glu=ffn_glu, rope_norm=rope_norm,
        )

        # The stream leaves through one zero-initialized gate, so at init it
        # contributes nothing and the block reduces to stream A. It has to be
        # exactly one such mechanism: a layer scale stacked on top of a gate is
        # what left the ConvNeXt branches with a 1e-8 gradient, and the gate
        # here is on its own for that reason.
        #
        # Note the attention block cannot supply this itself -- its cross
        # attention gate starts at identity, so its output is close to its
        # input rather than to zero.
        self.gate = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, self.d_b))

        self.up = nn.ConvTranspose1d(
            self.d_b, dim_out, up_kernel,
            stride=self.FACTOR, padding=(up_kernel - self.FACTOR) // 2,
        )

        # Only what this block owns directly: the two ConvNeXt blocks and the
        # attention block initialized themselves, and re-drawing them here would
        # undo it.
        init_weights_(self.up)
        if self.pre_pack is not None:
            init_weights_(self.pre_pack)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(
        self,
        x: torch.Tensor,                                            # (B, T, dim_in), T % 4 == 0
        prosody: torch.Tensor,                                      # (B, T/4, prosody_dim)
        context: torch.Tensor,                                      # (B, S, d_kv)
        cond: torch.Tensor,                                         # (B, cond_dim)
        key_padding_mask: Optional[torch.Tensor] = None,            # (B, T) or None
        packed_padding_mask: Optional[torch.Tensor] = None,         # (B, T/4) or None
        context_padding_mask: Optional[torch.Tensor] = None,        # (B, S) or None
    ) -> torch.Tensor:
        # --- Stream A ---
        a = self.conv1(x, cond, key_padding_mask)                   # (B, T, dim_in)
        a = self.conv2(a, cond, key_padding_mask)                   # (B, T, dim_out)

        # --- Stream B ---
        b = x if self.pre_pack is None else self.pre_pack(x)        # (B, T, pack_dim)
        # Padded frames are zeroed before they are packed: a group of four spans
        # the end of a short row, and whatever those positions carry would
        # otherwise ride into the attention stream on the live channels of an
        # otherwise valid group, where no key-padding mask can reach it.
        if key_padding_mask is not None:
            b = b * key_padding_mask.unsqueeze(-1).to(b.dtype)
        B, T, D = b.shape
        # Consecutive frames land side by side in the channel axis: frame t goes
        # to group t // 4 at offset (t % 4) * D, which is what a contiguous view
        # does and why nothing has to be gathered.
        b = b.reshape(B, T // self.FACTOR, self.FACTOR * D)
        b = torch.cat([b, prosody], dim=-1)                         # (B, T/4, d_b)

        b, _ = self.attn(b, context, packed_padding_mask, context_padding_mask, cond)
        b = self.gate(cond)[:, None, :] * b                         # (B, T/4, d_b)
        b = self.up(b.transpose(1, 2)).transpose(1, 2)              # (B, T, dim_out)

        return a + b                                                # (B, T, dim_out)
