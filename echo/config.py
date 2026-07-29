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
    seed: int
    text_dropout: float


@dataclass
class EchoConfig:
    # Constants
    latent_dim: int
    text_vocab_size: int
    text_embedding_dim: int
    time_embedding_dim: int
    init_std: float

    # Special tokens
    text_pad: int

    # Text encoder (Conformer) structural params
    text_encoder_num_layers: int
    text_encoder_num_heads: int
    text_encoder_ffn_dim: int
    text_encoder_ffn_glu: bool
    text_encoder_kernel_size: int
    text_encoder_dropout: float
    text_encoder_use_rope: bool
    text_encoder_conv_use_norm: bool

    # Main processing blocks (Echo). Each entry is a dict with a "type" key
    # ("convnext" | "self_attention" | "cross_attention") plus block-specific params.
    blocks: list[dict[str, Any]]

    # Training hyperparameters
    training: TrainingConfig

    # Factory method
    @classmethod
    def from_json(cls, path: str | Path) -> EchoConfig:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        te = d["text_encoder"]
        return cls(
            latent_dim=d["latent_dim"],
            text_vocab_size=d["vocab_size"]["text"],
            text_embedding_dim=d["text_embedding_dim"],
            time_embedding_dim=d["time_embedding_dim"],
            init_std=d["init_std"],
            text_pad=d["special_tokens"]["text_pad"],
            text_encoder_num_layers=te["num_layers"],
            text_encoder_num_heads=te["num_heads"],
            text_encoder_ffn_dim=te["ffn_dim"],
            text_encoder_ffn_glu=te["ffn_glu"],
            text_encoder_kernel_size=te["kernel_size"],
            text_encoder_dropout=te["dropout"],
            text_encoder_use_rope=te["use_rope"],
            text_encoder_conv_use_norm=te["conv_use_norm"],
            blocks=list(d["blocks"]),
            training=TrainingConfig(**d["training"]),
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)
