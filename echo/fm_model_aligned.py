from echo import config

from echo.fm_model import EchoFM
from echo.nn.transformer_blocks import SelfAttentionBlock

from typing import Optional

import torch
import torch.nn as nn


class EchoFMAligned(EchoFM):
    """
    EchoFM reading frame-aligned phonemes instead of attending over text.

    The alignment is the Viterbi path of the AR stage's CTC head against the
    known phoneme string (``scripts/preprocess/ctc_align.py``): one phoneme id
    or CTC blank per frame of a ``12.5 * ctc_upsample`` Hz grid. Stretched onto
    the latent's grid and embedded, it joins the latent and the prosody on the
    channel axis, so every frame reads the phoneme being spoken at it and the
    text enters before the stem rather than through cross-attention.

    Two things follow. The trunk's attention blocks are plain self-attention,
    since there is no context stream left to read, and the text encoder is gone
    with them -- a frame's phonetic context now comes from the convolutions and
    the self-attention that surround it.

    The first argument to :meth:`forward` and :meth:`sample` is the alignment,
    which is the only form the text takes here.
    """

    def __init__(self) -> None:
        cfg = config.fm_model

        super().__init__(
            audio_in_dim=(config.latent_dim + cfg.prosody_embedding_dim
                          + cfg.align_embedding_dim)
        )

        # Nothing reads the text as a sequence any more.
        del self.text_encoder

        # One row per phoneme plus one for the CTC blank, which is a state of
        # its own -- silence, or a frame the aligner would not commit.
        self.align_blank = config.text_vocab_size
        self.align_embed = nn.Embedding(config.text_vocab_size + 1,
                                        cfg.align_embedding_dim)
        nn.init.normal_(self.align_embed.weight, mean=0.0, std=config.init_std)

    # ---------------
    # Stack assembly
    # ---------------

    def _attention_block(self, dim: int) -> nn.Module:
        cfg = config.fm_model.trunk

        return SelfAttentionBlock(
            d_model=dim,
            num_heads=cfg.num_heads,
            ffn_dim=int(cfg.ffn_mult * dim),
            dropout=cfg.dropout,
            use_rope=cfg.use_rope,
            use_ada_ln=True,
            cond_dim=self.cond_dim,
        )

    # --------------------
    # Text conditioning
    # --------------------

    def embed_align(
        self,
        align: torch.Tensor,                                        # (B, A) long
        frames: int,                                                # T, the latent's length
        align_key_padding_mask: Optional[torch.Tensor] = None,      # (B, A) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
    ) -> torch.Tensor:
        """
        Stretch the alignment onto the latent grid and embed it.
        """

        idx = self._stretch(
            align, frames, align_key_padding_mask, latent_key_padding_mask,
        )                                                            # (B, T)

        return self.align_embed(idx)                                 # (B, T, align_dim)

    # --------------
    # Forward pass
    # --------------

    def forward(
        self,
        align: torch.Tensor,                                        # (B, A) long
        latent: torch.Tensor,                                       # (B, T, latent_dim)
        prosody: torch.Tensor,                                      # (B, K) long
        time: torch.Tensor,                                         # (B,)
        align_key_padding_mask: Optional[torch.Tensor] = None,      # (B, A) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
        prosody_key_padding_mask: Optional[torch.Tensor] = None,    # (B, K) or None
        prosody_drop_mask: Optional[torch.Tensor] = None,           # (B,) bool or None
    ) -> torch.Tensor:
        cond = self.time_encoder(time)                              # (B, cond_dim)
        frames = latent.shape[1]

        prosody_enc = self.embed_prosody(
            prosody, frames, prosody_key_padding_mask, latent_key_padding_mask,
        )                                                           # (B, T, prosody_dim)
        align_enc = self.embed_align(
            align, frames, align_key_padding_mask, latent_key_padding_mask,
        )                                                           # (B, T, align_dim)

        # Classifier-free guidance drops the prosody, as in the baseline; the
        # text stays, so guidance sharpens timbre and delivery, not content.
        if prosody_drop_mask is not None and prosody_drop_mask.any():
            null = self.null_prosody.expand_as(prosody_enc)         # (B, T, prosody_dim)
            prosody_enc = torch.where(
                prosody_drop_mask.view(-1, 1, 1), null, prosody_enc
            )

        x = torch.cat([latent, prosody_enc, align_enc], dim=-1)     # (B, T, audio_in_dim)
        x = self.stem(x)                                            # (B, T, stem_dim)
        mask = latent_key_padding_mask

        for block in self.blocks:
            if isinstance(block, SelfAttentionBlock):
                x, _ = block(x, mask, cond)
            else:                                                   # ConvNeXtBlock
                x = block(x, cond, mask)                            # (B, T, d')

        x = self.final_norm(x)                                      # (B, T, out_dim)

        return self.out_proj(x)                                     # (B, T, latent_dim)
