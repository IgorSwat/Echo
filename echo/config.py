from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainingConfig:
    data_dir: str
    output_dir: str
    batch_size: int
    num_epochs: int
    val_ratio: float
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    grad_clip: float
    num_workers: int
    log_every: int
    save_every: int
    early_stop: int
    seed: int
    # Classifier-free guidance dropout; only the flow-matching run uses it.
    text_dropout: float = 0.0
    # Weight of the CTC auxiliary loss; only the autoregressive run uses it.
    ctc_weight: float = 0.0


@dataclass
class TrainingSections:
    """Per-model training setups, one section per training script."""

    fm: TrainingConfig
    ar: TrainingConfig
    shortcut: TrainingConfig

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingSections:
        return cls(
            fm=TrainingConfig(**d["fm"]),
            ar=TrainingConfig(**d["ar"]),
            shortcut=TrainingConfig(**d["shortcut"]),
        )


@dataclass
class FMModelConfig:
    # Constants
    text_embedding_dim: int
    time_embedding_dim: int

    # Text encoder (Conformer) structural params
    text_encoder_num_layers: int
    text_encoder_num_heads: int
    text_encoder_ffn_dim: int
    text_encoder_ffn_glu: bool
    text_encoder_kernel_size: int
    text_encoder_dropout: float
    text_encoder_use_rope: bool
    text_encoder_conv_use_norm: bool

    # Main processing blocks. Each entry is a dict with a "type" key
    # ("convnext" | "self_attention" | "cross_attention") plus block-specific params.
    blocks: list[dict[str, Any]]

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FMModelConfig:
        te = d["text_encoder"]
        return cls(
            text_embedding_dim=d["text_embedding_dim"],
            time_embedding_dim=d["time_embedding_dim"],
            text_encoder_num_layers=te["num_layers"],
            text_encoder_num_heads=te["num_heads"],
            text_encoder_ffn_dim=te["ffn_dim"],
            text_encoder_ffn_glu=te["ffn_glu"],
            text_encoder_kernel_size=te["kernel_size"],
            text_encoder_dropout=te["dropout"],
            text_encoder_use_rope=te["use_rope"],
            text_encoder_conv_use_norm=te["conv_use_norm"],
            blocks=list(d["blocks"]),
        )


@dataclass
class ARModelConfig:
    # Constants
    emb_dim: int
    hidden_dim: int

    # Text encoder (Conformer) structural params
    text_encoder_d_model: int
    text_encoder_num_layers: int
    text_encoder_num_heads: int
    text_encoder_ffn_dim: int
    text_encoder_ffn_glu: bool
    text_encoder_kernel_size: int
    text_encoder_dropout: float
    text_encoder_use_rope: bool
    text_encoder_conv_use_norm: bool

    # Causal hybrid (self- + cross-attention) decoder
    decoder_num_layers: int
    decoder_num_heads: int
    decoder_ffn_dim: int
    decoder_ffn_glu: bool
    decoder_dropout: float
    decoder_use_rope: bool
    # How cross-attention encodes positions: "query" (each stream normalized by
    # its own length, needs the total up front) or "absolute" (raw indices with
    # a learnable frequency per stream, safe to decode step by step).
    decoder_rope_norm: str

    # Per-token-layer MLP prediction heads
    head_num_layers: int
    head_hidden_dim: int
    head_dropout: float
    # When set, head k > 0 is FiLM-modulated by token layer k - 1 of the same
    # frame, so a frame factorises as P(c0 | h) * P(c1 | c0, h) instead of
    # assuming the layers are conditionally independent given h.
    head_intra_frame_cond: bool = False

    # Auxiliary CTC head over the decoder states, used at training time only.
    # ``ctc_upsample`` widens the frame grid before the head: CTC needs at least
    # one frame per target phoneme, and the 12.5 Hz token grid is *below* the
    # phoneme rate of natural speech, so on the raw grid the loss is undefined
    # for nearly every utterance.
    ctc_enabled: bool = False
    ctc_upsample: int = 2

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ARModelConfig:
        te, dec, hd = d["text_encoder"], d["decoder"], d["head"]
        ctc = d.get("ctc", {})
        return cls(
            emb_dim=d["emb_dim"],
            hidden_dim=d["hidden_dim"],
            text_encoder_d_model=te["d_model"],
            text_encoder_num_layers=te["num_layers"],
            text_encoder_num_heads=te["num_heads"],
            text_encoder_ffn_dim=te["ffn_dim"],
            text_encoder_ffn_glu=te["ffn_glu"],
            text_encoder_kernel_size=te["kernel_size"],
            text_encoder_dropout=te["dropout"],
            text_encoder_use_rope=te["use_rope"],
            text_encoder_conv_use_norm=te["conv_use_norm"],
            decoder_num_layers=dec["num_layers"],
            decoder_num_heads=dec["num_heads"],
            decoder_ffn_dim=dec["ffn_dim"],
            decoder_ffn_glu=dec["ffn_glu"],
            decoder_dropout=dec["dropout"],
            decoder_use_rope=dec["use_rope"],
            decoder_rope_norm=dec.get("rope_norm", "query"),
            head_num_layers=hd["num_layers"],
            head_hidden_dim=hd["hidden_dim"],
            head_dropout=hd["dropout"],
            head_intra_frame_cond=hd.get("intra_frame_cond", False),
            ctc_enabled=ctc.get("enabled", False),
            ctc_upsample=ctc.get("upsample", 2),
        )


@dataclass
class ShortcutModelConfig:
    # Per-token-layer embedding width; the trunk runs at 2*emb_dim.
    emb_dim: int
    d_out: int
    dropout: float

    # ConvNeXt blocks per stage.
    blocks_per_stage: int

    # One entry per stage: (kernel_size, upsampling factor applied *after*
    # that stage's blocks; 1.0 means no interpolation).
    stages: list[tuple[int, float]]

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShortcutModelConfig:
        return cls(
            emb_dim=d["emb_dim"],
            d_out=d["d_out"],
            dropout=d["dropout"],
            blocks_per_stage=d["blocks_per_stage"],
            stages=[(s["kernel_size"], float(s["upsample"])) for s in d["stages"]],
        )


@dataclass
class EchoConfig:
    # Constants shared by every model
    latent_dim: int
    text_vocab_size: int
    prosody_vocab_size: int
    init_std: float

    # Limits
    text_len_limit: int

    # Special tokens
    text_pad: int
    prosody_pad: int
    prosody_bos: int
    prosody_eos: int
    prosody_mask: int

    # Flow-matching model (EchoFM)
    fm_model: FMModelConfig

    # Autoregressive prosody model (EchoAR)
    ar_model: ARModelConfig

    # Mimi tokens -> Blue latent shortcut model (EchoShortcut)
    shortcut_model: ShortcutModelConfig

    # Training hyperparameters, per model
    training: TrainingSections

    # Factory method
    @classmethod
    def from_json(cls, path: str | Path) -> EchoConfig:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        return cls(
            latent_dim=d["latent_dim"],
            text_vocab_size=d["vocab_size"]["text"],
            prosody_vocab_size=d["vocab_size"]["prosody"],
            init_std=d["init_std"],
            text_len_limit=d["limits"]["text_len"],
            text_pad=d["special_tokens"]["text_pad"],
            prosody_pad=d["special_tokens"]["prosody_pad"],
            prosody_bos=d["special_tokens"]["prosody_bos"],
            prosody_eos=d["special_tokens"]["prosody_eos"],
            prosody_mask=d["special_tokens"]["prosody_mask"],
            fm_model=FMModelConfig.from_dict(d["fm_model"]),
            ar_model=ARModelConfig.from_dict(d["ar_model"]),
            shortcut_model=ShortcutModelConfig.from_dict(d["shortcut_model"]),
            training=TrainingSections.from_dict(d["training"]),
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)
