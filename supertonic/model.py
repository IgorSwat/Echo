from supertonic import config

from supertonic.modules import apply_mask
from supertonic.modules.conv import ConvNeXtStack
from supertonic.modules.text_encoder import TextEncoder
from supertonic.modules.time_encoder import TimeConditionBlock, TimeEncoder
from supertonic.modules.transformer import StyleConditionBlock, TextConditionBlock

from typing import Optional

import torch
import torch.nn as nn


class Supertonic(nn.Module):
    """
    Supertonic-3 vector estimator: text-, style- and time-conditioned audio
    latent flow-matching backbone, with an Echo-style channels-last interface.

    Pipeline:
      1. Encode time  -> (B, time_dim) embedding, added by TimeConditionBlocks.
      2. Encode text  -> (B, S, text_embedding_dim) conditioning context.
      3. Project latent -> hidden_dim, run the main stack (config["blocks"]).
      4. Project hidden -> folded_dim output.

    As in the original, the model operates on FOLDED latents
    (B, T, folded_dim) = (B, T, latent_dim * chunk_compress_factor): the fold
    itself is a data-side reshape (see supertonic/fold.py), so `time` steps and
    latent frames live on the folded grid (6x coarser than the codec frames).

    Differences from Echo (the architectural points under comparison):
      * The backbone is a repeated macro-block of dilated ConvNeXt stacks with
        rotary text cross-attention and tanh-key style attention, all at a
        fixed hidden dim (no self-attention, no dim progression).
      * Time conditioning is additive (project-and-add), not AdaLN.
      * Conditioning is additionally injected via 50 learned style slots
        (GST-style); pass `style` to use a reference voice, otherwise learned
        default tokens are used.
    """

    # Registry mapping the "type" string in a block spec to its module class.
    BLOCK_REGISTRY = {
        "convnext_stack": ConvNeXtStack,
        "time_condition": TimeConditionBlock,
        "text_cross_attention": TextConditionBlock,
        "style_attention": StyleConditionBlock,
    }

    def __init__(self, audio_in_dim: Optional[int] = None) -> None:
        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.audio_in_dim = audio_in_dim if audio_in_dim is not None else config.folded_dim

        # --- Conditioning streams ---
        self.time_encoder = TimeEncoder(config.time_dim, config.time_hdim, config.time_scale)
        self.text_encoder = TextEncoder(
            vocab_size=config.text_vocab_size,
            d_model=config.text_embedding_dim,
            intermediate_dim=config.text_encoder_intermediate_dim,
            kernel_size=config.kernel_size,
            dilations=config.text_encoder_dilations,
            num_heads=config.text_encoder_attn_heads,
            num_layers=config.text_encoder_attn_layers,
            window_size=config.text_encoder_rel_window,
        )

        # --- Main processing stack (built from config["blocks"]) ---
        self.proj_in = nn.Linear(self.audio_in_dim, config.hidden_dim, bias=False)
        self.blocks = self._build_blocks()
        self.proj_out = nn.Linear(config.hidden_dim, config.folded_dim, bias=False)

        # Learned style slots: the keys are shared across voices (only the
        # values differ per speaker in the original); the default values act
        # as the voice when no reference style is passed.
        self.style_key = nn.Parameter(torch.zeros(1, config.n_style, config.style_dim))
        self.default_style = nn.Parameter(torch.zeros(1, config.n_style, config.style_dim))

        # Learned null conditions for classifier-free guidance: replace the
        # text encoder output and the style values for dropped samples.
        self.null_text = nn.Parameter(torch.zeros(1, 1, config.text_embedding_dim))
        self.null_style = nn.Parameter(torch.zeros(1, config.n_style, config.style_dim))

        self._init_weights()

    def _build_blocks(self) -> nn.ModuleList:
        blocks: list[nn.Module] = []
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
            blocks.append(self.BLOCK_REGISTRY[block_type](**spec))
        return nn.ModuleList(blocks)

    def _init_weights(self) -> None:
        nn.init.normal_(self.proj_in.weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=config.init_std)

    def forward(
        self,
        text: torch.Tensor,                                         # (B, S) long
        latent: torch.Tensor,                                       # (B, T, folded_dim) folded
        time: torch.Tensor,                                         # (B,)
        text_key_padding_mask: Optional[torch.Tensor] = None,       # (B, S) bool or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) bool or None (folded grid)
        text_drop_mask: Optional[torch.Tensor] = None,              # (B,) bool or None
        style: Optional[torch.Tensor] = None,                       # (B, n_style, style_dim) or None
    ) -> torch.Tensor:
        B = latent.shape[0]

        # First encode time, text and style
        temb = self.time_encoder(time)                              # (B, time_dim)
        text_enc = self.text_encoder(text, text_key_padding_mask)   # (B, S, text_dim)
        if style is None:
            style = self.default_style.expand(B, -1, -1)            # (B, n_style, style_dim)
        style_keys = self.style_key.expand(B, -1, -1)               # (B, n_style, style_dim)

        # Classifier-free guidance: swap in the learned null conditions for
        # samples whose conditioning is dropped (text and style drop together,
        # as in the original).
        if text_drop_mask is not None and text_drop_mask.any():
            drop = text_drop_mask.view(-1, 1, 1)                    # (B, 1, 1)
            text_enc = torch.where(drop, self.null_text.expand_as(text_enc), text_enc)
            style = torch.where(drop, self.null_style.expand_as(style), style)

        x = apply_mask(self.proj_in(latent), latent_key_padding_mask)  # (B, T, hidden)

        for block in self.blocks:
            if isinstance(block, TextConditionBlock):
                x = block(x, text_enc, text_key_padding_mask, latent_key_padding_mask)
            elif isinstance(block, StyleConditionBlock):
                x = block(x, style_keys, style, latent_key_padding_mask)
            elif isinstance(block, TimeConditionBlock):
                x = block(x, temb, latent_key_padding_mask)
            else:                                                   # ConvNeXtStack
                x = block(x, latent_key_padding_mask)

        x = self.proj_out(x)                                        # (B, T, folded_dim)

        return apply_mask(x, latent_key_padding_mask)               # (B, T, folded_dim)
