from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EchoConfig:
    # Codebook geometry
    num_codebooks: int
    text_vocab_size: int
    audio_vocab_size: int

    # Sequence bounds
    max_text_length: int
    max_audio_length: int

    # Special tokens
    text_pad_id: int
    audio_pad_id: int
    bos_id: int
    ref_text_eos_id: int
    ref_codec_eos_id: int
    text_eos_id: int
    eos_id: int

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

    # Codec embedding
    codec_token_embedding_dim: int
    codec_mlp_hidden_dim: int
    codec_mlp_num_layers: int
    codec_mlp_dropout: float

    # Codebook projector
    codebook_projector_no_context_chunks: int
    codebook_projector_d_model: int
    codebook_projector_num_layers: int
    codebook_projector_num_heads: int
    codebook_projector_ffn_dim: int
    codebook_projector_dropout: float

    # Training
    training_learning_rate: float
    training_batch_size: int
    training_num_epochs: int
    training_warmup_fraction: float
    training_weight_decay: float
    training_grad_clip: float
    training_weighted_loss: bool
    training_loss_decay: float
    training_scheduled_sampling_start_epoch: int
    training_scheduled_sampling_probability_increment: float
    training_scheduled_sampling_max_probability: float
    training_log_interval: int
    training_save_interval: int
    training_val_fraction: float
    training_phonemes_csv: str
    training_codec_dir: str
    training_output_dir: str

    @property
    def max_seq_len(self) -> int:
        return 2 * self.max_text_length + 2 * self.max_audio_length + 4

    @property
    def codec_logit_dim(self) -> int:
        return self.audio_vocab_size + 1

    @property
    def codec_vocab_size(self) -> int:
        return self.audio_vocab_size

    @property
    def codec_embedding_dim(self) -> int:
        return self.embedding_dim

    # Factory method
    @classmethod
    def from_json(cls, path: str | Path) -> EchoConfig:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        return cls(
            num_codebooks=d["num_codebooks"],
            text_vocab_size=d["vocab_size"]["text"],
            audio_vocab_size=d["vocab_size"]["audio"],
            max_text_length=d["limits"]["text_seq_len"],
            max_audio_length=d["limits"]["audio_seq_len"],
            text_pad_id=d["special_tokens"]["text_pad"],
            audio_pad_id=d["special_tokens"]["audio_pad"],
            bos_id=d["special_tokens"]["bos"],
            ref_text_eos_id=d["special_tokens"]["ref_text_eos"],
            ref_codec_eos_id=d["special_tokens"]["ref_codec_eos"],
            text_eos_id=d["special_tokens"]["text_eos"],
            eos_id=d["special_tokens"]["eos"],
            embedding_dim=d["embedding_dim"],
            intermediate_dim=d["intermediate_dim"],
            init_std=d["init_std"],
            decoder_num_layers=d["decoder"]["no_layers"],
            decoder_num_heads=d["decoder"]["no_heads"],
            decoder_hidden_dim=d["decoder"]["hidden_dim"],
            decoder_ffn_dim=d["decoder"]["ffn_dim"],
            decoder_ffn_glu=d["decoder"]["ffn_glu"],
            decoder_dropout=d["decoder"]["dropout"],
            codec_token_embedding_dim=d["codec_embedding"]["token_embedding_dim"],
            codec_mlp_hidden_dim=d["codec_embedding"]["mlp_hidden_dim"],
            codec_mlp_num_layers=d["codec_embedding"]["mlp_num_layers"],
            codec_mlp_dropout=d["codec_embedding"]["mlp_dropout"],
            codebook_projector_no_context_chunks=d["codebook_projector"]["no_context_chunks"],
            codebook_projector_d_model=d["codebook_projector"]["d_model"],
            codebook_projector_num_layers=d["codebook_projector"]["num_layers"],
            codebook_projector_num_heads=d["codebook_projector"]["num_heads"],
            codebook_projector_ffn_dim=d["codebook_projector"]["ffn_dim"],
            codebook_projector_dropout=d["codebook_projector"]["dropout"],
            training_learning_rate=d["training"]["learning_rate"],
            training_batch_size=d["training"]["batch_size"],
            training_num_epochs=d["training"]["num_epochs"],
            training_warmup_fraction=d["training"]["warmup_fraction"],
            training_weight_decay=d["training"]["weight_decay"],
            training_grad_clip=d["training"]["grad_clip"],
            training_weighted_loss=d["training"]["weighted_loss"],
            training_loss_decay=d["training"]["loss_decay"],
            training_scheduled_sampling_start_epoch=d["training"]["scheduled_sampling_start_epoch"],
            training_scheduled_sampling_probability_increment=d["training"]["scheduled_sampling_probability_increment"],
            training_scheduled_sampling_max_probability=d["training"]["scheduled_sampling_max_probability"],
            training_log_interval=d["training"]["log_interval"],
            training_save_interval=d["training"]["save_interval"],
            training_val_fraction=d["training"]["val_fraction"],
            training_phonemes_csv=d["training"]["phonemes_csv"],
            training_codec_dir=d["training"]["codec_dir"],
            training_output_dir=d["training"]["output_dir"],
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)
