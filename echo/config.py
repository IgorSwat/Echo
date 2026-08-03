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


@dataclass
class TrainingSections:
    """Per-model training setups, one section per training script."""

    fm: TrainingConfig
    ar: TrainingConfig

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingSections:
        return cls(
            fm=TrainingConfig(**d["fm"]),
            ar=TrainingConfig(**d["ar"]),
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

    # Causal convolutional front-end over the token embeddings
    conv_num_layers: int
    conv_kernel_size: int
    conv_dropout: float
    conv_use_norm: bool

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

    # Factory method
    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ARModelConfig:
        te, cv, dec, hd = d["text_encoder"], d["conv"], d["decoder"], d["head"]
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
            conv_num_layers=cv["num_layers"],
            conv_kernel_size=cv["kernel_size"],
            conv_dropout=cv["dropout"],
            conv_use_norm=cv["use_norm"],
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

    # Flow-matching model (EchoFM)
    fm_model: FMModelConfig

    # Autoregressive prosody model (EchoAR)
    ar_model: ARModelConfig

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
            fm_model=FMModelConfig.from_dict(d["fm_model"]),
            ar_model=ARModelConfig.from_dict(d["ar_model"]),
            training=TrainingSections.from_dict(d["training"]),
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)
