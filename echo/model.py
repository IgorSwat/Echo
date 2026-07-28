from echo import config

from echo.modules.conv import ConvNeXtBlock, Downsample1D, Upsample1D
from echo.modules.text_encoder import TextEncoder
from echo.modules.time_encoder import TimeEncoder
from echo.modules.transformer import CrossAttentionBlock, SelfAttentionBlock

from typing import Optional

import torch
import torch.nn as nn


class Echo(nn.Module):
    """
    Echo: text- and time-conditioned audio latent diffusion backbone.

    Pipeline:
      1. Encode time  -> (B, time_embedding_dim) conditioning vector (AdaLN).
      2. Encode text  -> (B, S, text_embedding_dim) conditioning context.
      3. Run the main processing stack (blocks defined in config["blocks"]).
      4. Project hidden -> latent_dim output.
    """

    # Registry mapping the "type" string in a block spec to its module class.
    BLOCK_REGISTRY = {
        "convnext": ConvNeXtBlock,
        "self_attention": SelfAttentionBlock,
        "cross_attention": CrossAttentionBlock,
        "downsample": Downsample1D,
        "upsample": Upsample1D,
    }

    # Block types that accept AdaLN conditioning (use_ada_ln / cond_dim).
    COND_TYPES = {"convnext", "self_attention", "cross_attention"}

    def __init__(self, audio_in_dim: Optional[int] = None) -> None:
        super().__init__()

        self.hidden_dim = config.text_embedding_dim
        self.cond_dim = config.time_embedding_dim
        self.audio_in_dim = audio_in_dim if audio_in_dim is not None else config.latent_dim

        # --- Conditioning streams ---
        self.time_encoder = TimeEncoder(config.time_embedding_dim)
        self.text_encoder = TextEncoder(
            vocab_size=config.text_vocab_size,
            d_model=config.text_embedding_dim,
            d_out=config.text_embedding_dim,
            num_layers=config.text_encoder_num_layers,
            num_heads=config.text_encoder_num_heads,
            ffn_dim=config.text_encoder_ffn_dim,
            ffn_glu=config.text_encoder_ffn_glu,
            kernel_size=config.text_encoder_kernel_size,
            dropout=config.text_encoder_dropout,
            use_rope=config.text_encoder_use_rope,
            conv_use_norm=config.text_encoder_conv_use_norm,
        )

        # --- Main processing stack (built from config["blocks"]) ---
        # The block flow defines its own dim progression starting from latent_dim.
        self.blocks, out_dim = self._build_blocks()

        self.final_norm = nn.LayerNorm(out_dim)
        self.out_proj = nn.Linear(out_dim, config.latent_dim)

        self._init_weights()

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        blocks: list[nn.Module] = []
        cur_dim = self.audio_in_dim
        for spec in config.blocks:
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
            blocks.append(cls(**spec))
            cur_dim = self._infer_out_dim(block_type, spec, cur_dim)
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
        raise ValueError(f"cannot infer output dim for block type '{block_type}'")

    def _init_weights(self) -> None:
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        latent: torch.Tensor,                                       # (B, T, latent_dim)
        time: torch.Tensor,                                         # (B,)
        text_key_padding_mask: Optional[torch.Tensor] = None,       # (B, S) or None
    ) -> torch.Tensor:
        # First encode both text & time
        cond = self.time_encoder(time)                                # (B, cond_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)    # (B, S, hidden)

        x = latent                                                     # (B, T, latent_dim)

        for block in self.blocks:
            if isinstance(block, CrossAttentionBlock):
                x = block(x, text_enc, text_key_padding_mask, cond)   # (B, T, d_model)
            elif isinstance(block, SelfAttentionBlock):
                x = block(x, None, cond)                               # (B, T, d_model)
            elif isinstance(block, (Downsample1D, Upsample1D)):
                x = block(x)                                            # (B, T', d')
            else:                                                      # ConvNeXtBlock, etc.
                x = block(x, cond)                                     # (B, T, d_model)

        x = self.final_norm(x)                                        # (B, T, hidden)
        x = self.out_proj(x)                                          # (B, T, latent_dim)

        return x                                                      # (B, T, latent_dim)
