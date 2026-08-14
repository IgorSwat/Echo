from echo import config

from echo.fm_model import EchoFM
from echo.nn.conv import ConvNeXtBlock
from echo.nn.dual_stream import DualStreamBlock

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoFMDual(EchoFM):
    """
    EchoFM with two persistent streams instead of one trunk.

    The latent is carried twice: once on its own 86.13 Hz grid at ``dim_a``,
    where ConvNeXt blocks see every frame, and once on a quarter-rate grid at
    ``dim_b``, where attention runs over a quarter of the positions. Both
    streams take the prosody embedding at their own rate, both run the length of
    the stack, and each block exchanges once in each direction.

    The shape comes from ``fm_model.dual`` in the config, the same way the
    baseline's comes from ``fm_model.blocks``; the shipped layout puts 16
    full-rate convolutions against 4 quarter-rate attention blocks, exchanging
    after every fourth convolution. That asymmetry is deliberate. A parameter on
    the full grid is applied at 600 positions and one on the quarter grid at
    150, so where the parameters sit decides how much arithmetic they do and how
    many gradient samples they see. An earlier layout
    (6 blocks of 2 convolutions, dim_a 256, dim_b 768) left 87.3% of the trunk
    on the quarter grid, which bought 1.85x the baseline's parameters while
    doing 0.79x its compute -- each parameter doing 0.43x the work of a baseline
    one. More weights, less learning per step. The shipped layout puts 59.4% of
    the trunk on the full grid instead, and being wider and shallower it also
    maps better onto the hardware: 2.06x the compute of the old layout at 26%
    less wall clock.

    The head reads both. Concatenating stream A with a reshape of stream B is
    lossless -- four quarter-rate frames of width ``dim_b`` are exactly
    ``dim_b // 4`` channels at full rate -- so the head sees everything either
    stream holds, and each stream reaches the loss directly rather than only
    through the other.
    """

    def __init__(self) -> None:
        # The trunk shape is stated in the config, the way the baseline's is.
        # Nothing here takes an override: two runs of this class are then
        # comparable by construction, and a checkpoint is readable from the
        # config alone rather than from whichever flags produced it.
        cfg = config.fm_model.dual

        if cfg.dim_b % DualStreamBlock.FACTOR != 0:
            raise ValueError(f"dim_b ({cfg.dim_b}) must be divisible by "
                             f"{DualStreamBlock.FACTOR} for the head's reshape")
        if cfg.dim_b % cfg.num_heads != 0:
            raise ValueError(f"dim_b ({cfg.dim_b}) must be divisible by "
                             f"num_heads ({cfg.num_heads})")

        super().__init__(audio_in_dim=config.latent_dim)

        dim_a, dim_b = cfg.dim_a, cfg.dim_b
        dropout, head_hidden = cfg.dropout, cfg.head_hidden
        F_ = DualStreamBlock.FACTOR

        # Stems: each stream starts from the latent on its own grid, with the
        # prosody embedding concatenated at that grid's rate.
        self.stem_a = ConvNeXtBlock(
            config.latent_dim + self.prosody_dim, dim_a, kernel_size=7,
            use_ada_ln=True, dropout=dropout, cond_dim=self.cond_dim,
        )
        self.stem_b = nn.Linear(F_ * config.latent_dim + self.prosody_dim, dim_b)
        nn.init.normal_(self.stem_b.weight, std=(F_ * config.latent_dim
                                                 + self.prosody_dim) ** -0.5)
        nn.init.zeros_(self.stem_b.bias)

        # The head reads both streams at full rate, but each is normalized on
        # its OWN channels first. One LayerNorm over the concatenation would
        # scale both by the joint standard deviation, and the two streams do not
        # arrive at comparable scales: measured at init, stream A carries rms
        # 0.277 over 256 channels against stream B's 2.424 over 192, so B holds
        # 98.3% of the variance and the output moves 6.6x more for a change in B
        # than in A. The convolution stream -- the only one that sees every
        # frame -- would enter the head as a 1.7% signal, which is why the model
        # could race ahead early on coarse structure and then stall exactly when
        # fine detail started to matter. Normalizing per stream costs two small
        # LayerNorms and puts them on equal footing.
        # How stream B reaches full rate. The reshape is lossless but imposes a
        # phase-shared weight constraint: frame t reads a FIXED slice of the
        # channels chosen by t % 4, and out_proj applies one matrix to all four
        # slices, so stream B has to make them interchangeable. On the 256/768
        # layout it only partly did, costing a 2.29% loss spread across phases
        # against the baseline's 0.78%, and a stride-4 transposed convolution
        # giving each phase its own taps was tried as the fix. It was measured
        # slightly WORSE (0.7808 against 0.7771 at 1200 steps) for 1.18M more
        # parameters, and on this layout the artifact it addressed is gone
        # anyway -- per-phase spread is 0.88% against the baseline's 0.81%, i.e.
        # the baseline's own noise floor. So there is nothing left to fix and
        # the alternative is gone with it.
        cat_dim = dim_a + dim_b // F_
        del self.final_norm                     # the base class's single norm
        self.norm_a = nn.LayerNorm(dim_a)
        self.norm_b = nn.LayerNorm(dim_b // F_)
        self.out_proj = nn.Sequential(
            nn.Linear(cat_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, config.latent_dim),
        )
        for lin in (self.out_proj[0], self.out_proj[2]):
            nn.init.normal_(lin.weight, std=lin.in_features ** -0.5)
            nn.init.zeros_(lin.bias)

    # ---------------
    # Stack assembly
    # ---------------

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        # Called from the base __init__, so it reads the config rather than
        # anything this class has had a chance to store on itself yet.
        c = config.fm_model.dual
        blocks = [
            DualStreamBlock(
                dim_a=c.dim_a, dim_b=c.dim_b, cond_dim=self.cond_dim,
                d_kv=self.hidden_dim, num_heads=c.num_heads,
                ffn_dim=int(c.ffn_mult * c.dim_b), num_conv=c.num_conv,
                dropout=c.dropout,
            )
            for _ in range(c.num_blocks)
        ]
        return nn.ModuleList(blocks), c.dim_a + c.dim_b // DualStreamBlock.FACTOR

    def _init_weights(self) -> None:
        nn.init.normal_(self.prosody_embed.weight, mean=0.0, std=config.init_std)
        # out_proj is replaced by the two-layer head after the base __init__.

    # --------------
    # Forward pass
    # --------------

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        latent: torch.Tensor,                                       # (B, T, latent_dim)
        prosody: torch.Tensor,                                      # (B, K) long
        time: torch.Tensor,                                         # (B,)
        text_key_padding_mask: Optional[torch.Tensor] = None,       # (B, S) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
        prosody_key_padding_mask: Optional[torch.Tensor] = None,    # (B, K) or None
        prosody_drop_mask: Optional[torch.Tensor] = None,           # (B,) bool or None
    ) -> torch.Tensor:
        F_ = DualStreamBlock.FACTOR

        cond = self.time_encoder(time)                              # (B, cond_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)   # (B, S, hidden)

        B, T, _ = latent.shape
        mask = latent_key_padding_mask

        pad = (-T) % F_
        if pad:
            latent = F.pad(latent, (0, 0, 0, pad))
            mask = (F.pad(mask, (0, pad), value=False) if mask is not None
                    else F.pad(latent.new_ones(B, T, dtype=torch.bool), (0, pad), value=False))
        frames = T + pad
        frames_q = frames // F_
        mask_q = mask.view(B, frames_q, F_).any(-1) if mask is not None else None

        def embed(n_frames: int, m: Optional[torch.Tensor]) -> torch.Tensor:
            enc = self.embed_prosody(prosody, n_frames, prosody_key_padding_mask, m)
            if prosody_drop_mask is not None and prosody_drop_mask.any():
                enc = torch.where(prosody_drop_mask.view(-1, 1, 1),
                                  self.null_prosody.expand_as(enc), enc)
            return enc

        # --- stems: one per stream, each at its own rate ---
        a = self.stem_a(
            torch.cat([latent, embed(frames, mask)], dim=-1), cond, mask,
        )                                                           # (B, T', dim_a)

        lat_q = latent
        if mask is not None:
            lat_q = lat_q * mask.unsqueeze(-1).to(lat_q.dtype)
        lat_q = lat_q.reshape(B, frames_q, F_ * config.latent_dim)
        b = self.stem_b(
            torch.cat([lat_q, embed(frames_q, mask_q)], dim=-1),
        )                                                           # (B, T'/4, dim_b)

        for block in self.blocks:
            a, b = block(a, b, text_enc, cond, mask, mask_q, text_key_padding_mask)

        # Strictly position-local: frame t reads group t // 4 and nothing else,
        # so padding cannot reach a valid frame here and needs no masking. The
        # block's B->A upsample is the one that does, and it masks.
        b_full = b.reshape(B, frames, b.shape[-1] // F_)             # (B, T', dim_b/4)
        x = torch.cat([self.norm_a(a), self.norm_b(b_full)], dim=-1)
        out = self.out_proj(x)                                      # (B, T', latent_dim)

        return out[:, :T] if pad else out
