from echo import config

from echo.modules.conv import ConvNeXtBlock, Downsample1D, SkipConnection1D, Upsample1D
from echo.modules.text_encoder import TextEncoder
from echo.modules.time_encoder import TimeEncoder
from echo.modules.transformer import CrossAttentionBlock, SelfAttentionBlock

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoFM(nn.Module):
    """
    EchoFM: text-, distil- and time-conditioned audio latent flow-matching backbone.

    Pipeline:
      1. Encode time  -> (B, time_embedding_dim) conditioning vector (AdaLN).
      2. Encode text  -> (B, S, text_embedding_dim) conditioning context.
      3. Run the main processing stack (blocks defined in config["fm_model"]["blocks"]).
      4. Project hidden -> latent_dim output.

    The flow runs from **noise** to the data latent; the distil latent is
    *conditioning*, concatenated channel-wise onto the integration state, which
    is why the stack starts at ``2 * latent_dim``.

    Using the distil as the source of the transport instead makes it a
    deterministic function of the target, and a deterministic pairing leaves the
    model nothing to sample: the L2-optimal answer is the mean over every
    high-quality rendering consistent with that distil, which is audible as
    blur. Noise restores the seed, so the detail the 2-codebook compression
    destroyed is drawn rather than averaged, and the distil still supplies the
    rhythm and content it was always meant to carry.

    U-Net skips: each `downsample` stashes the current feature; each `skip`
    pops the latest stash and fuses it with the current stream (after the
    matching `upsample`).
    """

    # Registry mapping the "type" string in a block spec to its module class.
    BLOCK_REGISTRY = {
        "convnext": ConvNeXtBlock,
        "self_attention": SelfAttentionBlock,
        "cross_attention": CrossAttentionBlock,
        "downsample": Downsample1D,
        "upsample": Upsample1D,
        "skip": SkipConnection1D,
    }

    # Block types that accept AdaLN conditioning (use_ada_ln / cond_dim).
    COND_TYPES = {"convnext", "self_attention", "cross_attention"}

    def __init__(self, audio_in_dim: Optional[int] = None) -> None:
        super().__init__()

        cfg = config.fm_model

        self.hidden_dim = cfg.text_embedding_dim
        self.cond_dim = cfg.time_embedding_dim
        # The state being integrated and the distil conditioning enter together,
        # stacked on the channel axis: latent_dim for each.
        self.audio_in_dim = (
            audio_in_dim if audio_in_dim is not None else 2 * config.latent_dim
        )

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

        # --- Main processing stack (built from config["fm_model"]["blocks"]) ---
        # The block flow defines its own dim progression starting from latent_dim.
        self.blocks, out_dim = self._build_blocks()

        # Learned null-text condition for classifier-free guidance: replaces the
        # text encoder output for samples whose conditioning is dropped.
        self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.text_embedding_dim))

        self.final_norm = nn.LayerNorm(out_dim)
        self.out_proj = nn.Linear(out_dim, config.latent_dim)

        self._init_weights()

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        blocks: list[nn.Module] = []
        cur_dim = self.audio_in_dim
        # Parallel to runtime skip stack: dims of features stashed before each downsample.
        skip_dims: list[int] = []
        for spec in config.fm_model.blocks:
            spec = dict(spec)                                          # copy (don't mutate config)
            block_type = spec.pop("type", None)
            if block_type is None:
                raise ValueError(f"Block spec missing 'type' key: {spec}")
            if block_type not in self.BLOCK_REGISTRY:
                raise ValueError(
                    f"Unknown block type '{block_type}'. "
                    f"Expected one of {list(self.BLOCK_REGISTRY)}"
                )
            cls = self.BLOCK_REGISTRY[block_type]
            # Inject AdaLN conditioning defaults; overridable per-block.
            if block_type in self.COND_TYPES:
                spec.setdefault("use_ada_ln", True)
                spec.setdefault("cond_dim", self.cond_dim)
            # Override dim_in with the current dimension so the config
            # only needs to specify dim_out for convnext blocks.
            if "dim_in" in spec:
                spec["dim_in"] = cur_dim
            if block_type == "downsample":
                skip_dims.append(cur_dim)
            elif block_type == "skip":
                if not skip_dims:
                    raise ValueError(
                        "skip block has no matching downsample "
                        "(skip stack is empty during build)"
                    )
                spec["dim_x"] = cur_dim
                spec["dim_skip"] = skip_dims.pop()
                spec.setdefault("mode", "add")
            blocks.append(cls(**spec))
            cur_dim = self._infer_out_dim(block_type, spec, cur_dim)
        if skip_dims:
            raise ValueError(
                f"{len(skip_dims)} downsample(s) without matching skip block(s)"
            )
        return nn.ModuleList(blocks), cur_dim

    @staticmethod
    def _infer_out_dim(block_type: str, spec: dict, in_dim: int) -> int:
        if block_type == "convnext":
            return spec["dim_out"]
        if block_type in ("self_attention", "cross_attention"):
            return spec["d_model"]
        if block_type == "downsample":
            return 2 * spec["dim_in"]
        if block_type == "upsample":
            return spec["dim_in"] // 2
        if block_type == "skip":
            return spec["dim_x"]
        raise ValueError(f"cannot infer output dim for block type '{block_type}'")

    def _init_weights(self) -> None:
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        latent: torch.Tensor,                                       # (B, T, latent_dim), the state
        time: torch.Tensor,                                         # (B,)
        distil: Optional[torch.Tensor] = None,                      # (B, T, latent_dim) conditioning
        text_key_padding_mask: Optional[torch.Tensor] = None,       # (B, S) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
        text_drop_mask: Optional[torch.Tensor] = None,              # (B,) bool or None
    ) -> torch.Tensor:
        if distil is None:
            raise ValueError("EchoFM conditions on a distil latent; pass `distil`")
        if distil.shape != latent.shape:
            raise ValueError(
                f"distil must match the state it conditions: got {tuple(distil.shape)} "
                f"against {tuple(latent.shape)}"
            )

        # First encode both text & time
        cond = self.time_encoder(time)                                # (B, cond_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)    # (B, S, hidden)

        # Classifier-free guidance: swap in the learned null-text condition
        # for samples whose text conditioning is dropped.
        if text_drop_mask is not None and text_drop_mask.any():
            null = self.null_text.expand_as(text_enc)                # (B, S, hidden)
            text_enc = torch.where(text_drop_mask.view(-1, 1, 1), null, text_enc)

        # The conditioning rides alongside the state through the whole stack:
        # both live on the same frame grid, so a channel-wise stack keeps them
        # aligned frame-for-frame with no resampling.
        x = torch.cat([latent, distil], dim=-1)                        # (B, T, 2*latent_dim)
        mask = latent_key_padding_mask
        skips: list[torch.Tensor] = []

        for block in self.blocks:
            if isinstance(block, CrossAttentionBlock):
                x, _ = block(x, text_enc, text_key_padding_mask, cond, mask)  # (B, T', d')
            elif isinstance(block, SelfAttentionBlock):
                x, _ = block(x, mask, cond)                            # (B, T', d')
            elif isinstance(block, Downsample1D):
                skips.append(x)                                        # stash encoder feature
                x = block(x)                                            # (B, T//2, 2C)
                if mask is not None:
                    mask = F.interpolate(
                        mask.float().unsqueeze(1),
                        size=x.shape[1],
                        mode="nearest",
                    ).squeeze(1).bool()
            elif isinstance(block, Upsample1D):
                x = block(x)                                            # (B, 2T, C//2)
                if mask is not None:
                    mask = F.interpolate(
                        mask.float().unsqueeze(1),
                        size=x.shape[1],
                        mode="nearest",
                    ).squeeze(1).bool()
            elif isinstance(block, SkipConnection1D):
                if not skips:
                    raise RuntimeError("skip block popped an empty skip stack")
                x = block(x, skips.pop())                               # (B, T, d)
                if mask is not None and mask.shape[1] != x.shape[1]:
                    mask = F.interpolate(
                        mask.float().unsqueeze(1),
                        size=x.shape[1],
                        mode="nearest",
                    ).squeeze(1).bool()
            else:                                                      # ConvNeXtBlock, etc.
                x = block(x, cond, mask)                               # (B, T', d')

        if skips:
            raise RuntimeError(f"{len(skips)} skip feature(s) left unused after forward")

        x = self.final_norm(x)                                        # (B, T, hidden)
        x = self.out_proj(x)                                          # (B, T, latent_dim)

        return x                                                      # (B, T, latent_dim)
