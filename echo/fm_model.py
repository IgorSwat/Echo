from echo import config

from echo.components.text_encoder import TextEncoder
from echo.components.time_encoder import TimeEncoder

from echo.nn.conv import ConvNeXtBlock
from echo.nn.transformer_blocks import (
    CrossAttentionBlock,
    HybridAttentionBlock,
    SelfAttentionBlock,
)

from typing import Optional

import torch
import torch.nn as nn


class EchoFM(nn.Module):
    """
    EchoFM: text-, distil- and time-conditioned audio latent flow-matching backbone.
    """

    # Maps the "type" in a block spec to its module class.
    BLOCK_REGISTRY = {
        "convnext": ConvNeXtBlock,
        "hybrid_attention": HybridAttentionBlock,
        "self_attention": SelfAttentionBlock,
        "cross_attention": CrossAttentionBlock,
    }

    # Block types that accept AdaLN conditioning (use_ada_ln / cond_dim).
    COND_TYPES = {"convnext", "hybrid_attention", "self_attention", "cross_attention"}

    # ODE integrators available to :meth:`sample`.
    SOLVERS = ("euler", "midpoint")

    def __init__(self, audio_in_dim: Optional[int] = None) -> None:
        super().__init__()

        cfg = config.fm_model

        self.hidden_dim = cfg.text_embedding_dim
        self.cond_dim = cfg.time_embedding_dim
        self.audio_in_dim = audio_in_dim if audio_in_dim is not None else config.latent_dim

        # --- Conditioning streams ---
        self.time_encoder = TimeEncoder(cfg.time_embedding_dim)
        self.text_encoder = TextEncoder(
            vocab_size=config.text_vocab_size,
            d_model=cfg.text_embedding_dim,
            d_out=cfg.text_embedding_dim,
            num_layers=cfg.text_encoder_num_layers,
            num_heads=cfg.text_encoder_num_heads,
            ffn_dim=cfg.text_encoder_ffn_dim,
            ffn_glu=cfg.text_encoder_ffn_glu,
            kernel_size=cfg.text_encoder_kernel_size,
            dropout=cfg.text_encoder_dropout,
            use_rope=cfg.text_encoder_use_rope,
            conv_use_norm=cfg.text_encoder_conv_use_norm,
            max_seq_len=config.text_len_limit,
        )

        # --- Main processing stack ---
        self.blocks, out_dim = self._build_blocks()

        # Null-text condition for classifier-free guidance.
        self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.text_embedding_dim))

        self.final_norm = nn.LayerNorm(out_dim)
        self.out_proj = nn.Linear(out_dim, config.latent_dim)

        self._init_weights()

    # ---------------
    # Stack assembly
    # ---------------

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        blocks: list[nn.Module] = []
        cur_dim = self.audio_in_dim

        for spec in config.fm_model.blocks:
            spec = dict(spec)                                       # copy: don't mutate config
            block_type = spec.pop("type", None)
            if block_type is None:
                raise ValueError(f"Block spec missing 'type' key: {spec}")
            if block_type not in self.BLOCK_REGISTRY:
                raise ValueError(
                    f"Unknown block type '{block_type}'. "
                    f"Expected one of {list(self.BLOCK_REGISTRY)}"
                )

            # AdaLN defaults, overridable per block.
            if block_type in self.COND_TYPES:
                spec.setdefault("use_ada_ln", True)
                spec.setdefault("cond_dim", self.cond_dim)
            # The running dimension wins, so a config only specifies dim_out.
            if "dim_in" in spec:
                spec["dim_in"] = cur_dim

            blocks.append(self.BLOCK_REGISTRY[block_type](**spec))
            cur_dim = self._infer_out_dim(block_type, spec)

        return nn.ModuleList(blocks), cur_dim

    @staticmethod
    def _infer_out_dim(block_type: str, spec: dict) -> int:
        if block_type == "convnext":
            return spec["dim_out"]
        if block_type in ("hybrid_attention", "self_attention", "cross_attention"):
            return spec["d_model"]
        raise ValueError(f"cannot infer output dim for block type '{block_type}'")

    def _init_weights(self) -> None:
        # Submodules initialized themselves; this covers what EchoFM owns directly.
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.out_proj.bias)

    # --------------
    # Forward pass
    # --------------

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        latent: torch.Tensor,                                       # (B, T, latent_dim)
        time: torch.Tensor,                                         # (B,)
        text_key_padding_mask: Optional[torch.Tensor] = None,       # (B, S) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
        text_drop_mask: Optional[torch.Tensor] = None,              # (B,) bool or None
    ) -> torch.Tensor:
        cond = self.time_encoder(time)                              # (B, cond_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)   # (B, S, hidden)

        # Classifier-free guidance: swap in the learned null-text condition.
        if text_drop_mask is not None and text_drop_mask.any():
            null = self.null_text.expand_as(text_enc)               # (B, S, hidden)
            text_enc = torch.where(text_drop_mask.view(-1, 1, 1), null, text_enc)

        x = latent                                                  # (B, T, latent_dim)
        mask = latent_key_padding_mask

        for block in self.blocks:
            if isinstance(block, HybridAttentionBlock):
                # `mask` covers the latent it attends over, `text_key_padding_mask`
                # the context it reads.
                x, _ = block(x, text_enc, mask, text_key_padding_mask, cond)
            elif isinstance(block, CrossAttentionBlock):
                x, _ = block(x, text_enc, text_key_padding_mask, cond, mask)
            elif isinstance(block, SelfAttentionBlock):
                x, _ = block(x, mask, cond)
            else:                                                   # ConvNeXtBlock
                x = block(x, cond, mask)                            # (B, T', d')

        x = self.final_norm(x)                                      # (B, T, hidden)

        return self.out_proj(x)                                     # (B, T, latent_dim)

    # -----------
    # Generation
    # -----------

    def _velocity(
        self,
        text: torch.Tensor,                                         # (1, S) long
        x: torch.Tensor,                                            # (1, T, latent_dim)
        t: torch.Tensor,                                            # (1,)
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        One velocity evaluation at (x, t), with classifier-free guidance.

        NOTE: with cfg_scale != 1 the model runs batch-doubled (conditioned +
        null-text) and extrapolates v_uncond + cfg * (v_cond - v_uncond).
        """

        if cfg_scale == 1.0:
            return self(text, x, t)

        drop = torch.tensor([False, True], device=x.device)
        v2 = self(text.repeat(2, 1), x.repeat(2, 1, 1), t.repeat(2), None, None, drop)

        return v2[1:2] + cfg_scale * (v2[0:1] - v2[1:2])            # (1, T, latent_dim)

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,                                         # (1, S) long
        x0: torch.Tensor,                                           # (1, T, latent_dim) distil
        steps: int,
        cfg_scale: float = 1.0,
        solver: str = "euler",
        generator: Optional[torch.Generator] = None,
        dither: bool = True,
    ) -> torch.Tensor:
        """
        Integrate the velocity field from t=0 (the source) to t=1 (data).

        NOTE: euler costs one model evaluation per step, midpoint (RK2) two.
        """
        if solver not in self.SOLVERS:
            raise ValueError(f"solver must be one of {self.SOLVERS}, got {solver!r}")

        dt = 1.0 / steps
        sigma = config.fm_model.source_noise if dither else 0.0
        x = x0 if sigma <= 0.0 else x0 + sigma * torch.randn(
            x0.shape, device=x0.device, dtype=x0.dtype, generator=generator
        )

        for i in range(steps):
            t0 = torch.full((1,), i / steps, device=text.device)
            if solver == "euler":
                x = x + dt * self._velocity(text, x, t0, cfg_scale)
            else:                                                   # midpoint (RK2)
                x_mid = x + 0.5 * dt * self._velocity(text, x, t0, cfg_scale)
                t_mid = torch.full((1,), (i + 0.5) / steps, device=text.device)
                x = x + dt * self._velocity(text, x_mid, t_mid, cfg_scale)

        return x                                                    # (1, T, latent_dim)
