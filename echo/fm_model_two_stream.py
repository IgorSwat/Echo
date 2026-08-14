from echo import config

from echo.fm_model import EchoFM
from echo.nn.conv import ConvNeXtBlock
from echo.nn.transformer_blocks import (
    CrossAttentionBlock,
    HybridAttentionBlock,
    SelfAttentionBlock,
)
from echo.nn.two_stream import TwoStreamBlock

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Design E: a full-rate convolution stream that reaches 256, beside an attention
# stream capped at 896 by packing a 192-wide projection. Letting the attention
# width follow the convolution's (4 * dim_in + 128) would put the last blocks at
# 1152 and cost 18.7M parameters more than the budget allows.
DEFAULT_BLOCKS: list[dict[str, Any]] = [
    {"type": "convnext",   "dim_in": 24,  "dim_out": 64,  "kernel_size": 7, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 64,  "dim_out": 96,  "pack_dim": 64,
     "num_heads": 8, "ffn_dim": 768,  "dropout": 0.1},
    {"type": "two_stream", "dim_in": 96,  "dim_out": 128, "pack_dim": 96,
     "num_heads": 8, "ffn_dim": 1024, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 128, "dim_out": 160, "pack_dim": 128,
     "num_heads": 8, "ffn_dim": 1280, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 160, "dim_out": 192, "pack_dim": 160,
     "num_heads": 8, "ffn_dim": 1536, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 192, "dim_out": 224, "pack_dim": 192,
     "num_heads": 8, "ffn_dim": 1792, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 224, "dim_out": 256, "pack_dim": 192,
     "num_heads": 8, "ffn_dim": 1792, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 256, "dim_out": 256, "pack_dim": 192,
     "num_heads": 8, "ffn_dim": 1792, "dropout": 0.1},
    {"type": "two_stream", "dim_in": 256, "dim_out": 256, "pack_dim": 192,
     "num_heads": 8, "ffn_dim": 1792, "dropout": 0.1},
]


class EchoFM2S(EchoFM):
    """
    EchoFM with a two-rate trunk.

    Everything outside the stack is unchanged -- the same text encoder, time
    encoder, prosody embedding and output head -- so a run against
    :class:`~echo.fm_model.EchoFM` isolates the trunk.

    What changes is where the prosody enters. The baseline concatenates it to
    the latent once, at the input, and lets it travel the whole stack on the
    channel axis. Here the latent enters alone and each block's attention stream
    concatenates the prosody itself, on the quarter-rate grid where the token
    rate nearly matches.
    """

    # Maps the "type" in a block spec to its module class; every one of them
    # accepts AdaLN conditioning.
    BLOCK_REGISTRY = {
        "convnext": ConvNeXtBlock,
        "hybrid_attention": HybridAttentionBlock,
        "self_attention": SelfAttentionBlock,
        "cross_attention": CrossAttentionBlock,
        "two_stream": TwoStreamBlock,
    }
    COND_TYPES = set(BLOCK_REGISTRY)

    def __init__(
        self,
        blocks: Optional[list[dict[str, Any]]] = None,
        full_rate_prosody: bool = False,
        gate_init: float = 0.0,
    ) -> None:
        specs = [dict(b) for b in (blocks if blocks is not None else DEFAULT_BLOCKS)]

        # With `full_rate_prosody` the stream is ALSO concatenated to the latent
        # at the input, as the baseline does, so it reaches every block at full
        # rate instead of only through the quarter-rate attention and its
        # stride-4 upsample.
        in_dim = config.latent_dim
        if full_rate_prosody:
            in_dim += config.fm_model.prosody_embedding_dim
            specs[0]["dim_in"] = in_dim

        object.__setattr__(self, "_specs", specs)
        object.__setattr__(self, "_full_rate_prosody", full_rate_prosody)
        object.__setattr__(self, "_gate_init", gate_init)
        super().__init__(audio_in_dim=in_dim)

        # A zero gate hands stream B no gradient at all on the first step, and
        # 85% of this model's parameters sit behind it. `gate_init` opens it by a
        # constant instead; 1.0 matches what HybridAttentionBlock already does
        # for its cross attention, which starts at identity rather than closed.
        if gate_init != 0.0:
            for block in self.blocks:
                if isinstance(block, TwoStreamBlock):
                    nn.init.constant_(block.gate[-1].bias, gate_init)

    # ---------------
    # Stack assembly
    # ---------------

    def _build_stem(self) -> tuple[nn.Module, int]:
        # The first block reads the latent directly; there is no stem to lift it.
        return nn.Identity(), self.audio_in_dim

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        blocks: list[nn.Module] = []
        cur_dim = self.audio_in_dim

        for spec in self._specs:
            spec = dict(spec)
            block_type = spec.pop("type", None)
            if block_type not in self.BLOCK_REGISTRY:
                raise ValueError(
                    f"Unknown block type '{block_type}'. "
                    f"Expected one of {list(self.BLOCK_REGISTRY)}"
                )

            if block_type in self.COND_TYPES:
                spec.setdefault("use_ada_ln", True)
                spec.setdefault("cond_dim", self.cond_dim)
            if block_type == "two_stream":
                spec.setdefault("prosody_dim", self.prosody_dim)
                spec.setdefault("d_kv", self.hidden_dim)

            width = spec.get("dim_in", spec.get("d_model"))
            if width is not None and width != cur_dim:
                key = "dim_in" if "dim_in" in spec else "d_model"
                raise ValueError(
                    f"block {len(blocks)} ({block_type}) declares {key} {width}, "
                    f"but the stack is {cur_dim} wide at that point."
                )

            blocks.append(self.BLOCK_REGISTRY[block_type](**spec))
            cur_dim = spec.get("dim_out", spec.get("d_model"))

        return nn.ModuleList(blocks), cur_dim

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
        cond = self.time_encoder(time)                              # (B, cond_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)   # (B, S, hidden)

        B, T, _ = latent.shape
        mask = latent_key_padding_mask

        # The packing needs a length divisible by four. Padding here rather than
        # in the collate keeps the extra frames invisible to everything upstream:
        # they are masked out on the way in and sliced off on the way out.
        pad = (-T) % TwoStreamBlock.FACTOR
        if pad:
            latent = F.pad(latent, (0, 0, 0, pad))
            mask = (F.pad(mask, (0, pad), value=False) if mask is not None
                    else F.pad(latent.new_ones(B, T, dtype=torch.bool), (0, pad), value=False))

        frames_q = (T + pad) // TwoStreamBlock.FACTOR
        # A packed group counts as real if any of its four frames does.
        mask_q = (mask.view(B, frames_q, TwoStreamBlock.FACTOR).any(-1)
                  if mask is not None else None)

        # Stretched onto the quarter-rate grid, where 12.5 Hz against 21.53 Hz
        # repeats each token ~1.7 times instead of ~6.9.
        prosody_enc = self.embed_prosody(
            prosody, frames_q, prosody_key_padding_mask, mask_q,
        )                                                           # (B, T/4, prosody_dim)

        if prosody_drop_mask is not None and prosody_drop_mask.any():
            null = self.null_prosody.expand_as(prosody_enc)
            prosody_enc = torch.where(
                prosody_drop_mask.view(-1, 1, 1), null, prosody_enc
            )

        x = latent                                                  # (B, T+pad, latent_dim)
        if self._full_rate_prosody:
            # The same tokens, stretched onto the full grid as well, so the
            # convolution stream reads them directly rather than only through
            # whatever survives the upsample.
            full = self.embed_prosody(prosody, latent.shape[1],
                                      prosody_key_padding_mask, mask)
            if prosody_drop_mask is not None and prosody_drop_mask.any():
                full = torch.where(prosody_drop_mask.view(-1, 1, 1),
                                   self.null_prosody.expand_as(full), full)
            x = torch.cat([x, full], dim=-1)                        # (B, T+pad, 152)

        for block in self.blocks:
            if isinstance(block, TwoStreamBlock):
                x = block(x, prosody_enc, text_enc, cond, mask, mask_q,
                          text_key_padding_mask)
            else:                                                   # ConvNeXt stem
                x = block(x, cond, mask)

        x = self.final_norm(x)
        out = self.out_proj(x)                                      # (B, T+pad, latent_dim)

        return out[:, :T] if pad else out                           # (B, T, latent_dim)
