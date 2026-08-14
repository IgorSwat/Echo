from echo import config

from echo.components.text_encoder import TextEncoder
from echo.components.time_encoder import TimeEncoder

from echo.nn.conv import ConvNeXtBlock
from echo.nn.init import init_weights_
from echo.nn.transformer_blocks import HybridAttentionBlock

from typing import Optional

import torch
import torch.nn as nn


class EchoFM(nn.Module):
    """
    EchoFM: text-, prosody-, distil- and time-conditioned latent flow-matching backbone.

    A pointwise stem lifts the latent (plus its prosody embedding) to the
    trunk's starting width, the trunk alternates runs of ConvNeXt blocks with
    single hybrid attention blocks, and a linear head projects the widest state
    back to the latent.
    """

    # ODE integrators available to :meth:`sample`.
    SOLVERS = ("euler", "midpoint")

    # Latent frames per prosody token: BlueCodec's 44100/512 = 86.13 Hz grid
    # against Mimi's 12.5 Hz one. The two do not divide, which is why the
    # stretch in :meth:`embed_prosody` is built per row from the real lengths;
    # this constant only *chooses* a length when generating.
    LATENT_FRAMES_PER_TOKEN = (44100 / 512) / 12.5                  # 6.890625

    def __init__(self, audio_in_dim: Optional[int] = None) -> None:
        super().__init__()

        cfg = config.fm_model

        self.hidden_dim = cfg.text_embedding_dim
        self.cond_dim = cfg.time_embedding_dim
        self.prosody_dim = cfg.prosody_embedding_dim
        # The prosody embedding rides alongside the latent on the channel axis,
        # so the stem reads that much more than the latent itself.
        self.audio_in_dim = (
            audio_in_dim if audio_in_dim is not None
            else config.latent_dim + self.prosody_dim
        )

        # --- Conditioning streams ---
        self.time_encoder = TimeEncoder(cfg.time_embedding_dim)
        # Sized to the full prosody vocabulary, so pad/bos/eos/mask are embeddable
        # even though only real codebook ids should reach a trained model.
        self.prosody_embed = nn.Embedding(config.prosody_vocab_size, self.prosody_dim)
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

        # The unconditional branch for classifier-free guidance drops the
        # *prosody*, not the text: the tokens are what carries the content, so
        # they are what guidance should sharpen. Dropping them in training is
        # also the only thing that stops the concatenated stream from being
        # sufficient on its own.
        self.null_prosody = nn.Parameter(torch.zeros(1, 1, cfg.prosody_embedding_dim))

        # --- Stem, trunk, head ---
        self.stem, self.stem_dim = self._build_stem()
        self.blocks, out_dim = self._build_blocks()

        self.final_norm = nn.LayerNorm(out_dim)
        self.out_proj = nn.Linear(out_dim, config.latent_dim)

        self._init_weights()

    # ---------------
    # Stack assembly
    # ---------------

    def _build_stem(self) -> tuple[nn.Module, int]:
        cfg = config.fm_model.stem
        hidden = cfg.hidden_dim if cfg.hidden_dim is not None else cfg.dim

        dims = [self.audio_in_dim] + [hidden] * cfg.hidden_layers + [cfg.dim]
        layers: list[nn.Module] = []
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            if i:
                layers.append(nn.GELU())
            layers.append(nn.Linear(d_in, d_out))

        return nn.Sequential(*layers), cfg.dim

    def _build_blocks(self) -> tuple[nn.ModuleList, int]:
        cfg = config.fm_model.trunk
        blocks: list[nn.Module] = []
        dim = self.stem_dim

        def convnext(dim_in: int, dim_out: int) -> nn.Module:
            return ConvNeXtBlock(
                dim_in=dim_in,
                dim_out=dim_out,
                kernel_size=cfg.kernel_size,
                dropout=cfg.dropout,
                use_ada_ln=True,
                cond_dim=self.cond_dim,
            )

        for i, stage in enumerate(cfg.stages):
            if stage.num_conv < 2 or stage.num_conv % 2 != 0:
                raise ValueError(
                    f"stage {i} has num_conv {stage.num_conv}; it must be even, so "
                    f"the attention sits between the two halves of the stage"
                )
            if stage.dim < dim:
                raise ValueError(
                    f"stage {i} narrows the trunk from {dim} to {stage.dim}; "
                    f"widths may only grow, since the head reads the widest one"
                )

            # The stage's first convolution is what widens it; everything after
            # runs at the stage's own width, with the attention in the middle of
            # the convolutions rather than at the end.
            before = stage.num_conv // 2
            for j in range(before):
                blocks.append(convnext(dim if j == 0 else stage.dim, stage.dim))
            dim = stage.dim

            blocks.extend(self._attention_block(dim)
                          for _ in range(stage.num_attention))
            blocks.extend(convnext(dim, dim) for _ in range(stage.num_conv - before))

        return nn.ModuleList(blocks), dim

    def _attention_block(self, dim: int) -> nn.Module:
        cfg = config.fm_model.trunk

        return HybridAttentionBlock(
            d_model=dim,
            d_kv=self.hidden_dim,
            num_heads=cfg.num_heads,
            ffn_dim=int(cfg.ffn_mult * dim),
            dropout=cfg.dropout,
            use_rope=cfg.use_rope,
            rope_norm=cfg.rope_norm,
            use_ada_ln=True,
            cond_dim=self.cond_dim,
        )

    def _init_weights(self) -> None:
        # Submodules initialized themselves; this covers what EchoFM owns directly.
        init_weights_(self.stem)
        nn.init.normal_(self.prosody_embed.weight, mean=0.0, std=config.init_std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.out_proj.bias)

    # -------------------
    # Prosody conditioning
    # -------------------

    def _stretch(
        self,
        tokens: torch.Tensor,                                       # (B, K) long
        frames: int,                                                # T, the latent's length
        token_key_padding_mask: Optional[torch.Tensor] = None,      # (B, K) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
    ) -> torch.Tensor:
        """
        Resample a token grid onto the latent grid, per row from the real lengths.
        """

        B, K = tokens.shape
        device = tokens.device

        if token_key_padding_mask is not None:
            k = token_key_padding_mask.sum(1)                        # (B,)
        else:
            k = torch.full((B,), K, dtype=torch.long, device=device)
        if latent_key_padding_mask is not None:
            t_len = latent_key_padding_mask.sum(1)                   # (B,)
        else:
            t_len = torch.full((B,), frames, dtype=torch.long, device=device)

        k = k.clamp(min=1)
        t_len = t_len.clamp(min=1)

        pos = torch.arange(frames, device=device)[None, :]           # (1, T)
        idx = (pos * k[:, None]) // t_len[:, None]                   # (B, T)
        # Past a row's own end the index runs off; it is clamped to that row's
        # last real token and the position is masked out downstream regardless.
        idx = torch.minimum(idx, (k - 1)[:, None]).clamp(min=0)

        return tokens.gather(1, idx)                                 # (B, T)

    def embed_prosody(
        self,
        prosody: torch.Tensor,                                      # (B, K) long
        frames: int,                                                # T, the latent's length
        prosody_key_padding_mask: Optional[torch.Tensor] = None,    # (B, K) or None
        latent_key_padding_mask: Optional[torch.Tensor] = None,     # (B, T) or None
    ) -> torch.Tensor:
        """
        Stretch the token grid onto the latent grid and embed it.
        """

        idx = self._stretch(
            prosody, frames, prosody_key_padding_mask, latent_key_padding_mask,
        )                                                            # (B, T)

        return self.prosody_embed(idx)                               # (B, T, prosody_dim)

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

        # The prosody stream is frame-aligned, so it joins on the channel axis
        # and travels through the stack as part of the latent.
        prosody_enc = self.embed_prosody(
            prosody, latent.shape[1], prosody_key_padding_mask, latent_key_padding_mask,
        )                                                           # (B, T, prosody_dim)

        # Classifier-free guidance: swap the whole stream for the learned null
        # condition. The row keeps its text and its length, and loses only the
        # tokens -- that is the branch guidance extrapolates away from.
        if prosody_drop_mask is not None and prosody_drop_mask.any():
            null = self.null_prosody.expand_as(prosody_enc)         # (B, T, prosody_dim)
            prosody_enc = torch.where(
                prosody_drop_mask.view(-1, 1, 1), null, prosody_enc
            )

        x = torch.cat([latent, prosody_enc], dim=-1)                # (B, T, audio_in_dim)
        x = self.stem(x)                                            # (B, T, stem_dim)
        mask = latent_key_padding_mask

        for block in self.blocks:
            if isinstance(block, HybridAttentionBlock):
                # `mask` covers the latent it attends over, `text_key_padding_mask`
                # the context it reads.
                x, _ = block(x, text_enc, mask, text_key_padding_mask, cond)
            else:                                                   # ConvNeXtBlock
                x = block(x, cond, mask)                            # (B, T, d')

        x = self.final_norm(x)                                      # (B, T, out_dim)

        return self.out_proj(x)                                     # (B, T, latent_dim)

    # -----------
    # Generation
    # -----------

    def _velocity(
        self,
        text: torch.Tensor,                                         # (1, S) long
        x: torch.Tensor,                                            # (1, T, latent_dim)
        prosody: torch.Tensor,                                      # (1, K) long
        t: torch.Tensor,                                            # (1,)
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        One velocity evaluation at (x, t), with classifier-free guidance.
        """

        if cfg_scale == 1.0:
            return self(text, x, prosody, t)

        drop = torch.tensor([False, True], device=x.device)
        v2 = self(text.repeat(2, 1), x.repeat(2, 1, 1), prosody.repeat(2, 1),
                  t.repeat(2), None, None, None, drop)

        return v2[1:2] + cfg_scale * (v2[0:1] - v2[1:2])            # (1, T, latent_dim)

    def latent_frames(self, tokens: int) -> int:
        """
        How many latent frames a run of ``tokens`` prosody tokens covers.
        """

        return max(1, round(tokens * self.LATENT_FRAMES_PER_TOKEN))

    @torch.no_grad()
    def sample(
        self,
        text: torch.Tensor,                                         # (1, S) long
        prosody: torch.Tensor,                                      # (1, K) long
        steps: int,
        cfg_scale: float = 1.0,
        solver: str = "euler",
        generator: Optional[torch.Generator] = None,
        frames: Optional[int] = None,                               # default: from `prosody`
    ) -> torch.Tensor:
        """Integrate the velocity field from t=0 (noise) to t=1 (data).

        NOTE: euler costs one model evaluation per step, midpoint (RK2) two.
        """
        if solver not in self.SOLVERS:
            raise ValueError(f"solver must be one of {self.SOLVERS}, got {solver!r}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        dt = 1.0 / steps
        T = self.latent_frames(prosody.shape[1]) if frames is None else frames
        x = torch.randn(
            (prosody.shape[0], T, config.latent_dim),
            device=prosody.device, dtype=torch.float32, generator=generator,
        )

        for i in range(steps):
            t0 = torch.full((1,), i / steps, device=text.device)
            if solver == "euler":
                x = x + dt * self._velocity(text, x, prosody, t0, cfg_scale)
            else:                                                   # midpoint (RK2)
                x_mid = x + 0.5 * dt * self._velocity(text, x, prosody, t0, cfg_scale)
                t_mid = torch.full((1,), (i + 0.5) / steps, device=text.device)
                x = x + dt * self._velocity(text, x_mid, prosody, t_mid, cfg_scale)

        return x                                                    # (1, T, latent_dim)
