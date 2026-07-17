#!/usr/bin/env python3
"""Train the AR decoder on (phonemes, first-codebook audio tokens) pairs.

The dataset consists of:

* ``<data-dir>/phonemes.csv``  -- ``name.npz|phoneme_string`` per line.
* ``<data-dir>/codecs/<name>``  -- ``.npz`` files with a ``codes`` array of
  shape ``(16, T)``; only the first codebook (``codes[0]``) is used.

Training is standard causal language modelling with teacher forcing.  The
model sees ``[text, <SEP>, audio]`` and is optimised to predict, for every
audio position, the *next* audio token, and at the last real audio position
the ``<EOS>`` token (which controls the generated length).  The loss is a
weighted cross-entropy: every prediction step contributes equally and the
single EOS step additionally receives a configurable extra weight.

Padding is used for batching: text is right-padded with ``TEXT_PAD_ID`` and
audio with ``AUDIO_PAD_ID`` (an extra learned row of the audio embedding).
A key-padding mask prevents attention to padded positions.

Usage:
    python scripts/train_ar_decoder.py --data-dir data/norbi \
        --phoneme-vocab checkpoints/phoneme_vocab.json \
        --output-dir models/ar_decoder

    # quick smoke run on a tiny subset
    python scripts/train_ar_decoder.py --data-dir data/norbi \
        --phoneme-vocab checkpoints/phoneme_vocab.json \
        --max-samples 64 --epochs 1 --batch-size 8 --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import Dataset, DataLoader  # noqa: E402

from echo.ar_decoder import ARDecoder  # noqa: E402
from echo.config import (  # noqa: E402
    AR_BATCH_SIZE,
    AR_BETAS,
    AR_D_FF,
    AR_D_MODEL,
    AR_DROPOUT,
    AR_EOS_LOSS_WEIGHT,
    AR_GRAD_CLIP,
    AR_LABEL_SMOOTHING,
    AR_LEARNING_RATE,
    AR_LOG_EVERY,
    AR_N_HEADS,
    AR_N_LAYERS,
    AR_NUM_EPOCHS,
    AR_NUM_WORKERS,
    AR_SAVE_EVERY,
    AR_SEED,
    AR_VAL_EVERY,
    AR_VAL_FRACTION,
    AR_WARMUP_STEPS,
    AR_WEIGHT_DECAY,
    AUDIO_PAD_ID,
    CODEC_VOCAB_SIZE,
    EOS_TOKEN_ID,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    TEXT_PAD_ID,
)
from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)

IGNORE_INDEX = -100  # target value ignored by F.cross_entropy


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class PhonemeTokenizer:
    """Map a phoneme string to a list of token ids, one per code point.

    The phoneme vocab (``checkpoints/phoneme_vocab.json``) maps individual
    Unicode code points -- including combining marks such as ``˜`` and
    diacritics -- to ids.  Iterating the Python string yields code points,
    so each character becomes exactly one token, matching the vocab.
    """

    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = stoi

    @classmethod
    def from_file(cls, path: Path | str) -> "PhonemeTokenizer":
        with open(path) as f:
            vocab = json.load(f)
        return cls(vocab["stoi"])

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for ch in text:
            idx = self.stoi.get(ch)
            if idx is None:
                raise KeyError(f"Unknown phoneme character {ch!r} (U+{ord(ch):04X}) "
                               f"in text: {text[:40]!r}...")
            # Skip the special pad/bos/eos tokens if they ever appear -- they
            # are not part of the real phoneme sequence.
            if idx in (0, 1, 2):
                continue
            ids.append(idx)
        return ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class CodecDataset(Dataset):
    """Preloaded ``(phoneme_ids, first-codebook audio ids)`` pairs."""

    def __init__(
        self,
        data_dir: Path | str,
        tokenizer: PhonemeTokenizer,
        max_samples: Optional[int] = None,
        max_text_length: int = MAX_TEXT_LENGTH,
        max_audio_length: int = MAX_AUDIO_LENGTH,
    ) -> None:
        data_dir = Path(data_dir)
        csv_path = data_dir / "phonemes.csv"
        codecs_dir = data_dir / "codecs"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Missing {csv_path}")
        if not codecs_dir.is_dir():
            raise FileNotFoundError(f"Missing {codecs_dir}")

        self.text_list: list[np.ndarray] = []
        self.audio_list: list[np.ndarray] = []
        self.names: list[str] = []

        skipped = 0
        with open(csv_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                name, phoneme_str = line.split("|", 1)
                try:
                    text_ids = np.asarray(tokenizer.encode(phoneme_str), dtype=np.int64)
                except KeyError as e:
                    print_error(f"Skipping {name}: {e}")
                    skipped += 1
                    continue

                codec_path = codecs_dir / name
                if not codec_path.is_file():
                    skipped += 1
                    continue
                try:
                    codes = np.load(codec_path)["codes"]
                except Exception as e:  # noqa: BLE001
                    print_error(f"Skipping {name}: failed to load codes: {e}")
                    skipped += 1
                    continue
                audio_ids = codes[0].astype(np.int64)  # first codebook

                # Filter over-length samples.
                if len(text_ids) == 0 or len(text_ids) > max_text_length:
                    skipped += 1
                    continue
                if len(audio_ids) == 0 or len(audio_ids) > max_audio_length:
                    skipped += 1
                    continue
                # Sanity: codec token range.
                if audio_ids.min() < 0 or audio_ids.max() >= CODEC_VOCAB_SIZE:
                    print_error(f"Skipping {name}: codec token out of range "
                                f"[0,{CODEC_VOCAB_SIZE})")
                    skipped += 1
                    continue

                self.text_list.append(text_ids)
                self.audio_list.append(audio_ids)
                self.names.append(name)

                if max_samples is not None and len(self.text_list) >= max_samples:
                    break

        print_info("Loaded samples", str(len(self.text_list)), Colors.OKCYAN)
        if skipped:
            print_info("Skipped samples", str(skipped), Colors.WARNING)

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.text_list[idx], self.audio_list[idx]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def make_collate():
    """Return a picklable collate callable (use :class:`CodecCollate`)."""
    return CodecCollate()


class CodecCollate:
    """Top-level (picklable) collate_fn for ``DataLoader`` workers.

    Pads a batch of ``(phoneme_ids, audio_ids)`` to common lengths and builds
    all the tensors the training loop needs:

    * ``text_tokens``     -- ``(B, Tt)`` right-padded with ``TEXT_PAD_ID``.
    * ``audio_tokens``    -- ``(B, Ta)`` right-padded with ``AUDIO_PAD_ID``.
    * ``key_padding_mask``-- ``(B, Tt+1+Ta)`` bool, ``True`` = ignore.
    * ``targets``         -- ``(B, Ta+1)`` teacher-forcing targets (real codec
      ids at the first ``la`` positions, ``EOS_TOKEN_ID`` at position ``la``).
    * ``valid_mask``      -- ``(B, Ta+1)`` bool, ``True`` where loss is computed.
    * ``eos_mask``        -- ``(B, Ta+1)`` bool, ``True`` only at the EOS step.
    * ``text_lens``/``audio_lens`` -- original lengths.
    * ``text_pad_start`` -- ``Tt``, the index of ``<SEP>`` in the full sequence.
    """

    def __call__(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        texts = [b[0] for b in batch]
        audios = [b[1] for b in batch]
        B = len(batch)

        Tt = max(len(t) for t in texts)
        Ta = max(len(a) for a in audios)

        text_tokens = np.full((B, Tt), TEXT_PAD_ID, dtype=np.int64)
        audio_tokens = np.full((B, Ta), AUDIO_PAD_ID, dtype=np.int64)
        text_lens = np.zeros(B, dtype=np.int64)
        audio_lens = np.zeros(B, dtype=np.int64)

        for i, (t, a) in enumerate(zip(texts, audios)):
            lt, la = len(t), len(a)
            text_tokens[i, :lt] = t
            audio_tokens[i, :la] = a
            text_lens[i] = lt
            audio_lens[i] = la

        T_total = Tt + 1 + Ta
        # key_padding_mask: True = ignore in attention.
        key_padding_mask = np.zeros((B, T_total), dtype=bool)
        for i in range(B):
            lt = int(text_lens[i])
            la = int(audio_lens[i])
            # padded text positions (right of the real text).
            key_padding_mask[i, lt:Tt] = True
            # padded audio positions (right of the real audio).
            key_padding_mask[i, Tt + 1 + la:] = True
            # SEP at index Tt is never padded.

        # Targets aligned with logits at positions [SEP, audio_0, ..., audio_{Ta-1}].
        # That slice has length Ta + 1.
        targets = np.full((B, Ta + 1), IGNORE_INDEX, dtype=np.int64)
        valid_mask = np.zeros((B, Ta + 1), dtype=bool)  # positions we compute loss on
        eos_mask = np.zeros((B, Ta + 1), dtype=bool)   # the EOS-prediction position
        for i in range(B):
            la = int(audio_lens[i])
            a = audios[i]
            # position j (0-indexed in this slice) corresponds to logits at
            # [SEP, audio_0, ..., audio_{Ta-1}][j].  It predicts the *next*
            # token:
            #   j = 0  (SEP)      -> audio[0]
            #   j = k  (audio[k-1]) -> audio[k]
            #   j = la  (audio[la-1]) -> <EOS>
            # So targets[0..la-1] = audio[0..la-1], targets[la] = EOS.
            targets[i, :la] = a
            targets[i, la] = EOS_TOKEN_ID
            valid_mask[i, : la + 1] = True
            eos_mask[i, la] = True

        return {
            "text_tokens": torch.from_numpy(text_tokens),
            "audio_tokens": torch.from_numpy(audio_tokens),
            "key_padding_mask": torch.from_numpy(key_padding_mask),
            "targets": torch.from_numpy(targets),
            "valid_mask": torch.from_numpy(valid_mask),
            "eos_mask": torch.from_numpy(eos_mask),
            "text_lens": torch.from_numpy(text_lens),
            "audio_lens": torch.from_numpy(audio_lens),
            "text_pad_start": Tt,  # index of SEP in the full sequence
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    logits: torch.Tensor,    # (B, Ta+1, V)
    targets: torch.Tensor,   # (B, Ta+1)  int64, IGNORE_INDEX for invalid
    valid_mask: torch.Tensor,  # (B, Ta+1) bool
    eos_mask: torch.Tensor,   # (B, Ta+1) bool
    label_smoothing: float,
    eos_weight: float,
) -> torch.Tensor:
    """Weighted cross-entropy averaged over valid prediction steps.

    Every valid step has base weight 1; the EOS-prediction step additionally
    gets ``eos_weight``.  Padding steps carry weight 0 and are ignored.
    """
    B, Tp, V = logits.shape
    flat_logits = logits.reshape(B * Tp, V)
    flat_targets = targets.reshape(B * Tp)

    # F.cross_entropy with ignore_index returns 0 for ignored positions.
    per_token = F.cross_entropy(
        flat_logits, flat_targets,
        reduction="none",
        label_smoothing=label_smoothing,
        ignore_index=IGNORE_INDEX,
    ).view(B, Tp)

    # Per-position weights (base 1, EOS extra, padding 0).
    weights = torch.ones_like(per_token)
    weights = weights.masked_fill(eos_mask, eos_weight)
    weights = weights * valid_mask.to(weights.dtype)

    loss = (per_token * weights).sum() / weights.sum().clamp_min(1.0)
    return loss


# ---------------------------------------------------------------------------
# Optimizer & scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: ARDecoder, lr: float, weight_decay: float,
                    betas: tuple) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay on non-bias / non-norm params."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("bias") or "norm" in name.lower() \
                or "embedding" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def build_scheduler(optimizer: torch.optim.Optimizer,
                    warmup_steps: int, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay to 0."""
    warmup = max(1, warmup_steps)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def select_device(prefer: str) -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    # auto fallback
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model: ARDecoder, loader: DataLoader, device: torch.device,
             label_smoothing: float, eos_weight: float
             ) -> tuple[float, float, float]:
    """Return ``(weighted_loss, audio_loss, eos_loss)``.

    * ``weighted_loss`` -- the same weighted CE that training optimises
      (base weight 1 per valid step, EOS step additionally weighted by
      ``eos_weight``).  Use this to compare against the training loss and
      to select the best checkpoint.
    * ``audio_loss``    -- mean CE over valid *non-EOS* (audio-token)
      positions only.  Unweighted.
    * ``eos_loss``      -- mean CE over the EOS-prediction positions only.
      Unweighted.

    Reporting ``audio`` and ``eos`` separately (rather than a single
    "all valid" mean) makes it easy to see whether length prediction is
    keeping up with token prediction.
    """
    model.eval()
    w_sum, w_cnt = 0.0, 0.0      # weighted total
    aud_sum, aud_cnt = 0.0, 0.0   # audio-only
    eos_sum, eos_cnt = 0.0, 0.0   # eos-only
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(batch["text_tokens"], batch["audio_tokens"],
                        key_padding_mask=batch["key_padding_mask"])
        Tt = batch["text_pad_start"]
        Ta = batch["audio_tokens"].shape[1]
        pred_logits = logits[:, Tt:Tt + Ta + 1, :]

        per_token = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)),
            batch["targets"].reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
            ignore_index=IGNORE_INDEX,
        ).view(pred_logits.shape[0], -1)

        valid = batch["valid_mask"].to(per_token.dtype)
        eos = batch["eos_mask"].to(per_token.dtype)
        audio = valid * (1.0 - eos)  # valid audio-token positions (exclude EOS)

        # Weighted loss (matches training: weight 1 on audio, eos_weight on EOS).
        weights = valid.clone()
        weights = torch.where(eos.bool(), torch.full_like(weights, eos_weight), weights)
        w_sum += (per_token * weights).sum().item()
        w_cnt += weights.sum().item()

        # Audio-only (unweighted).
        aud_sum += (per_token * audio).sum().item()
        aud_cnt += audio.sum().item()

        # EOS-only (unweighted).
        eos_sum += (per_token * eos).sum().item()
        eos_cnt += eos.sum().item()

    model.train()
    weighted_loss = w_sum / max(1.0, w_cnt)
    audio_loss = aud_sum / max(1.0, aud_cnt)
    eos_loss = eos_sum / max(1.0, eos_cnt)
    return weighted_loss, audio_loss, eos_loss


def save_checkpoint(path: Path, model: ARDecoder,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler.LambdaLR,
                     step: int, epoch: int, best_val: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "best_val_loss": best_val,
    }, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the AR decoder.")
    p.add_argument("--data-dir", type=str, default="data/norbi")
    p.add_argument("--phoneme-vocab", type=str, default="checkpoints/phoneme_vocab.json")
    p.add_argument("--output-dir", type=str, default="models/ar_decoder")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    # overrides for hyperparams
    p.add_argument("--epochs", type=int, default=AR_NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=AR_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=AR_LEARNING_RATE)
    p.add_argument("--warmup-steps", type=int, default=AR_WARMUP_STEPS)
    p.add_argument("--weight-decay", type=float, default=AR_WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=AR_GRAD_CLIP)
    p.add_argument("--label-smoothing", type=float, default=AR_LABEL_SMOOTHING)
    p.add_argument("--eos-weight", type=float, default=AR_EOS_LOSS_WEIGHT)
    p.add_argument("--num-workers", type=int, default=AR_NUM_WORKERS)
    p.add_argument("--val-fraction", type=float, default=AR_VAL_FRACTION)
    p.add_argument("--val-every", type=int, default=AR_VAL_EVERY)
    p.add_argument("--save-every", type=int, default=AR_SAVE_EVERY)
    p.add_argument("--log-every", type=int, default=AR_LOG_EVERY)
    p.add_argument("--seed", type=int, default=AR_SEED)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Use only the first N samples (for smoke tests).")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 autocast on CUDA.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_header("AR Decoder - Training")
    print_separator()

    # ---- seed --------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device("auto" if args.device == "auto" else args.device)
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Data dir", str(args.data_dir))
    print_info("Output dir", str(args.output_dir))
    print_info("Batch size", str(args.batch_size))
    print_info("Learning rate (peak)", str(args.lr))
    print_info("Warmup steps", str(args.warmup_steps))
    print_info("Epochs", str(args.epochs))
    print_info("Label smoothing", str(args.label_smoothing))
    print_info("EOS loss weight", str(args.eos_weight))
    print_info("bf16 autocast", str(args.bf16))

    # ---- tokenizer + dataset ----------------------------------------------
    print_section("Loading data")
    tokenizer = PhonemeTokenizer.from_file(args.phoneme_vocab)
    print_info("Phoneme vocab size", str(len(tokenizer.stoi)))
    dataset = CodecDataset(args.data_dir, tokenizer, max_samples=args.max_samples)
    n = len(dataset)
    if n == 0:
        print_error("No samples loaded; aborting.")
        sys.exit(1)

    # ---- train / val split -------------------------------------------------
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(n, generator=g).numpy()
    n_val = max(1, int(n * args.val_fraction))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_ds = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_ds = torch.utils.data.Subset(dataset, val_idx.tolist())
    print_info("Train samples", str(len(train_ds)), Colors.OKCYAN)
    print_info("Val samples", str(len(val_ds)), Colors.OKCYAN)

    collate = make_collate()
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=args.num_workers,
        pin_memory=pin, drop_last=False,
    )

    # ---- model ------------------------------------------------------------
    print_section("Model")
    model = ARDecoder(
        d_model=AR_D_MODEL, n_heads=AR_N_HEADS, d_ff=AR_D_FF,
        n_layers=AR_N_LAYERS, dropout=AR_DROPOUT,
    ).to(device)
    print_info("Total parameters", f"{model.num_parameters():,}")
    print_info("Non-embedding parameters",
               f"{model.num_parameters(exclude_embeddings=True):,}", Colors.OKCYAN)

    # ---- optimizer / scheduler -------------------------------------------
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    optimizer = build_optimizer(model, args.lr, args.weight_decay, AR_BETAS)
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)
    print_info("Steps per epoch", str(steps_per_epoch))
    print_info("Total optimizer steps", str(total_steps))

    # ---- resume -----------------------------------------------------------
    start_epoch, global_step, best_val = 0, 0, float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        best_val = ckpt.get("best_val_loss", float("inf"))
        print_info("Resumed from", str(args.resume), Colors.OKCYAN)
        print_info("  at epoch/step", f"{start_epoch}/{global_step}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- training loop ----------------------------------------------------
    print_section("Training")
    print_separator("═", 60)
    use_amp = args.bf16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t0 = time.perf_counter()
    running_loss, running_cnt = 0.0, 0

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                logits = model(batch["text_tokens"], batch["audio_tokens"],
                               key_padding_mask=batch["key_padding_mask"])
                Tt = batch["text_pad_start"]
                Ta = batch["audio_tokens"].shape[1]
                pred_logits = logits[:, Tt:Tt + Ta + 1, :]
                loss = compute_loss(
                    pred_logits, batch["targets"], batch["valid_mask"],
                    batch["eos_mask"], args.label_smoothing, args.eos_weight,
                )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += loss.item()
            running_cnt += 1

            if global_step % args.log_every == 0:
                avg = running_loss / running_cnt
                lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t0
                print(f"  step {global_step:>7} | epoch {epoch} | "
                      f"loss {avg:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                running_loss, running_cnt = 0.0, 0

            if global_step % args.val_every == 0:
                val_loss, aud_loss, eos_loss = evaluate(
                    model, val_loader, device,
                    args.label_smoothing, args.eos_weight,
                )
                flag = " (new best)" if val_loss < best_val else ""
                print(f"  [val] step {global_step:>7} | "
                      f"loss {val_loss:.4f} | audio {aud_loss:.4f} | "
                      f"eos {eos_loss:.4f}{flag}")
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(output_dir / "best.pt", model, optimizer,
                                    scheduler, global_step, epoch, best_val)

            if global_step % args.save_every == 0:
                save_checkpoint(output_dir / "latest.pt", model, optimizer,
                                scheduler, global_step, epoch, best_val)

        # end of epoch checkpoint
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler,
                        global_step, epoch + 1, best_val)
        val_loss, aud_loss, eos_loss = evaluate(model, val_loader, device,
                                                   args.label_smoothing, args.eos_weight)
        flag = " (new best)" if val_loss < best_val else ""
        print(f"  [epoch {epoch} val] loss {val_loss:.4f} | "
              f"audio {aud_loss:.4f} | eos {eos_loss:.4f}{flag}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, scheduler,
                            global_step, epoch + 1, best_val)

    print_separator("═", 60)
    print_success(f"Training done. Best val loss: {best_val:.4f}")
    print_info("Checkpoints", str(output_dir))


if __name__ == "__main__":
    main()
