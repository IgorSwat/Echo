from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EchoConfig:
    # Latent geometry
    latent_dim: int
    text_vocab_size: int

    # Sequence bounds
    max_text_length: int
    max_audio_length: int

    # Special tokens (text only; audio uses continuous latents, no discrete tokens)
    text_pad_id: int
    bos_id: int
    ref_text_eos_id: int
    ref_audio_eos_id: int
    text_eos_id: int

    # Dimensions
    embedding_dim: int
    intermediate_dim: int
    init_std: float

    # Decoder
    decoder_num_layers: int
    decoder_num_heads: int
    decoder_hidden_dim: int
    decoder_ffn_dim: int
    decoder_ffn_glu: bool
    decoder_dropout: float

    # Training
    training_learning_rate: float
    training_batch_size: int
    training_num_epochs: int
    training_warmup_fraction: float
    training_weight_decay: float
    training_grad_clip: float
    training_log_interval: int
    training_save_interval: int
    training_val_fraction: float
    training_phonemes_csv: str
    training_latent_dir: str
    training_output_dir: str

    @property
    def max_seq_len(self) -> int:
        return 2 * self.max_text_length + 2 * self.max_audio_length + 4

    @property
    def codec_embedding_dim(self) -> int:
        return self.embedding_dim

    # Factory method
    @classmethod
    def from_json(cls, path: str | Path) -> EchoConfig:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        return cls(
            latent_dim=d["latent_dim"],
            text_vocab_size=d["vocab_size"]["text"],
            max_text_length=d["limits"]["text_seq_len"],
            max_audio_length=d["limits"]["audio_seq_len"],
            text_pad_id=d["special_tokens"]["text_pad"],
            bos_id=d["special_tokens"]["bos"],
            ref_text_eos_id=d["special_tokens"]["ref_text_eos"],
            ref_audio_eos_id=d["special_tokens"]["ref_audio_eos"],
            text_eos_id=d["special_tokens"]["text_eos"],
            embedding_dim=d["embedding_dim"],
            intermediate_dim=d["intermediate_dim"],
            init_std=d["init_std"],
            decoder_num_layers=d["decoder"]["no_layers"],
            decoder_num_heads=d["decoder"]["no_heads"],
            decoder_hidden_dim=d["decoder"]["hidden_dim"],
            decoder_ffn_dim=d["decoder"]["ffn_dim"],
            decoder_ffn_glu=d["decoder"]["ffn_glu"],
            decoder_dropout=d["decoder"]["dropout"],
            training_learning_rate=d["training"]["learning_rate"],
            training_batch_size=d["training"]["batch_size"],
            training_num_epochs=d["training"]["num_epochs"],
            training_warmup_fraction=d["training"]["warmup_fraction"],
            training_weight_decay=d["training"]["weight_decay"],
            training_grad_clip=d["training"]["grad_clip"],
            training_log_interval=d["training"]["log_interval"],
            training_save_interval=d["training"]["save_interval"],
            training_val_fraction=d["training"]["val_fraction"],
            training_phonemes_csv=d["training"]["phonemes_csv"],
            training_latent_dir=d["training"]["latent_dir"],
            training_output_dir=d["training"]["output_dir"],
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)