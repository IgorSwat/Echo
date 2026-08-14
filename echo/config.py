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

    # Classifier-free guidance dropout; only the flow-matching run uses it. The
    # prosody stream is what gets dropped, since that is what guidance sharpens.
    prosody_dropout: float = 0.0

    # Weight of the CTC auxiliary loss; only the autoregressive run uses it.
    ctc_weight: float = 0.0

    # Input corruption for the non-autoregressive run: this fraction of the
    # conditioning codes is replaced by random codebook entries before the model
    # reads them. At inference layer 0 arrives from the AR stage and every layer
    # above it from this model's own draws, so a share of the conditioning is
    # always wrong; training on a perfect stack is training on a distribution
    # that never occurs.
    token_noise: float = 0.0

    # History corruption for the autoregressive run: from
    # ``history_mask_start_epoch`` onwards, a growing fraction of the *input*
    # frames is replaced before the model reads them, so it learns to recover
    # from a history that is not ground truth. A max of 0 disables the schedule.
    history_mask_max: float = 0.0
    history_mask_start_epoch: int = 1
    history_mask_step: float = 0.02
    # Corruption comes in contiguous runs: isolated frames are trivially
    # interpolated from their neighbours, while real drift arrives in bursts.
    history_mask_span_min: int = 2
    history_mask_span_max: int = 5
    # Share of corrupted frames that become the mask token; the rest are replaced
    # by random codec tokens, which is what an AR error actually looks like — a
    # plausible wrong frame rather than a flag saying "ignore me".
    history_mask_token_frac: float = 0.3


@dataclass
class TrainingSections:
    """Per-model training setups, one section per training script."""

    fm: TrainingConfig
    ar: TrainingConfig
    nar: TrainingConfig

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrainingSections:
        return cls(
            fm=TrainingConfig(**d["fm"]),
            ar=TrainingConfig(**d["ar"]),
            nar=TrainingConfig(**d["nar"]),
        )


@dataclass
class FMModelConfig:
    text_embedding_dim: int
    time_embedding_dim: int

    # Width of the prosody token embedding. These are the AR stage's layer-0
    # (Mimi semantic) tokens: embedded, stretched onto the latent's frame grid,
    # and concatenated to the latent along the channel axis, so the stack's
    # input is ``latent_dim + prosody_embedding_dim`` wide.
    prosody_embedding_dim: int

    # Text encoder (Conformer)
    text_encoder_num_layers: int
    text_encoder_num_heads: int
    text_encoder_ffn_dim: int
    text_encoder_ffn_glu: bool
    text_encoder_kernel_size: int
    text_encoder_dropout: float
    text_encoder_use_rope: bool
    text_encoder_conv_use_norm: bool

    # Main processing stack. Each entry is a dict with a "type" key naming a
    # block in EchoFM.BLOCK_REGISTRY, plus that block's own parameters.
    blocks: list[dict[str, Any]]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FMModelConfig:
        te = d["text_encoder"]
        return cls(
            text_embedding_dim=d["text_embedding_dim"],
            time_embedding_dim=d["time_embedding_dim"],
            prosody_embedding_dim=d["prosody_embedding_dim"],
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
    emb_dim: int
    hidden_dim: int

    # Text encoder (Conformer)
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

    # Auxiliary CTC head over the decoder states, used at training time only.
    # ``ctc_upsample`` widens the frame grid before the head: CTC needs at least
    # one frame per target phoneme, and the 12.5 Hz token grid is *below* the
    # phoneme rate of natural speech, so on the raw grid the loss is undefined
    # for nearly every utterance.
    ctc_enabled: bool = False
    ctc_upsample: int = 2

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
            ctc_enabled=ctc.get("enabled", False),
            ctc_upsample=ctc.get("upsample", 2),
        )


@dataclass
class NARModelConfig:
    # Codec layers the stage covers, layer 0 included. It reads layer 0 from the
    # AR stage and writes the ``num_layers - 1`` above it, so the number also
    # decides how many layers the dataset has to load.
    num_layers: int

    emb_dim: int
    hidden_dim: int
    # Width of the layer-index embedding, which is the AdaLN conditioning: one
    # network writes every layer, and this is all that tells it which.
    cond_dim: int

    # Text encoder (Conformer)
    text_encoder_d_model: int
    text_encoder_num_layers: int
    text_encoder_num_heads: int
    text_encoder_ffn_dim: int
    text_encoder_ffn_glu: bool
    text_encoder_kernel_size: int
    text_encoder_dropout: float
    text_encoder_use_rope: bool
    text_encoder_conv_use_norm: bool

    # Bidirectional hybrid (self- + cross-attention) encoder
    encoder_num_layers: int
    encoder_num_heads: int
    encoder_ffn_dim: int
    encoder_ffn_glu: bool
    encoder_dropout: float
    encoder_use_rope: bool
    encoder_rope_norm: str

    # Per-layer MLP prediction heads (one per written codebook)
    head_num_layers: int
    head_hidden_dim: int
    head_dropout: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> NARModelConfig:
        te, enc, hd = d["text_encoder"], d["encoder"], d["head"]
        return cls(
            num_layers=d["num_layers"],
            emb_dim=d["emb_dim"],
            hidden_dim=d["hidden_dim"],
            cond_dim=d["cond_dim"],
            text_encoder_d_model=te["d_model"],
            text_encoder_num_layers=te["num_layers"],
            text_encoder_num_heads=te["num_heads"],
            text_encoder_ffn_dim=te["ffn_dim"],
            text_encoder_ffn_glu=te["ffn_glu"],
            text_encoder_kernel_size=te["kernel_size"],
            text_encoder_dropout=te["dropout"],
            text_encoder_use_rope=te["use_rope"],
            text_encoder_conv_use_norm=te["conv_use_norm"],
            encoder_num_layers=enc["num_layers"],
            encoder_num_heads=enc["num_heads"],
            encoder_ffn_dim=enc["ffn_dim"],
            encoder_ffn_glu=enc["ffn_glu"],
            encoder_dropout=enc["dropout"],
            encoder_use_rope=enc["use_rope"],
            encoder_rope_norm=enc.get("rope_norm", "query"),
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

    # Where latent normalization statistics come from: "dataset" (one fixed
    # per-channel affine map for the whole corpus, from latent_stats.npz) or
    # "instance" (each utterance normalized by its own per-channel statistics,
    # taken from its distil so the transform can be inverted at inference).
    latent_norm: str

    # Limits
    text_len_limit: int

    # Special tokens
    text_pad: int
    prosody_pad: int
    prosody_bos: int
    prosody_eos: int
    prosody_mask: int

    # Per-model architecture, then per-model training hyperparameters
    fm_model: FMModelConfig
    ar_model: ARModelConfig
    nar_model: NARModelConfig
    training: TrainingSections

    @classmethod
    def from_json(cls, path: str | Path) -> EchoConfig:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)

        return cls(
            latent_dim=d["latent_dim"],
            text_vocab_size=d["vocab_size"]["text"],
            prosody_vocab_size=d["vocab_size"]["prosody"],
            init_std=d["init_std"],
            latent_norm=d.get("latent_norm", "dataset"),
            text_len_limit=d["limits"]["text_len"],
            text_pad=d["special_tokens"]["text_pad"],
            prosody_pad=d["special_tokens"]["prosody_pad"],
            prosody_bos=d["special_tokens"]["prosody_bos"],
            prosody_eos=d["special_tokens"]["prosody_eos"],
            prosody_mask=d["special_tokens"]["prosody_mask"],
            fm_model=FMModelConfig.from_dict(d["fm_model"]),
            ar_model=ARModelConfig.from_dict(d["ar_model"]),
            nar_model=NARModelConfig.from_dict(d["nar_model"]),
            training=TrainingSections.from_dict(d["training"]),
        )


# Default instance loaded from the canonical config JSON.
config: EchoConfig = EchoConfig.from_json(
    Path(__file__).resolve().parent.parent / "models" / "config.json"
)
