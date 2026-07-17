#!/usr/bin/env python3
"""Train the NAR decoder on the upper Mimi codebooks (layers 2..16).

The dataset is the same as the AR decoder's:

* ``<data-dir>/phonemes.csv``  -- ``name.npz|phoneme_string`` per line.
* ``<data-dir>/codecs/<name>``  -- ``.npz`` files with a ``codes`` array of
  shape ``(16, T)``.  Here **all 16 codebook layers** are used.

Training mirrors the AR decoder (teacher forcing, causal transformer stack),
with one key difference: instead of predicting the *next* token of a single
codebook, the NAR decoder predicts, for every frame, the token of codebook
``i`` given the text and **all lower codebooks** ``0..i-1``:

    forward([text, <SEP>, Σ_{k<i} code_emb_k(codes[k])])  ->  logits for codes[i]

There is no shift and no EOS: the audio position ``j`` predicts codebook ``i``
at the *same* frame ``j`` (non-autoregressive across the layer axis).

Every step evaluates all target codebooks ``i = 1..15`` and combines their
per-frame cross-entropies into a single **weighted average**: codebook ``i``
is weighted by ``layer_weight_decay ** (i-1)`` so the 2nd codebook (``i=1``)
has the highest impact and each subsequent codebook contributes geometrically
less.  To keep memory bounded, the per-codebook losses are back-propagated one
at a time (gradient accumulation) rather than held in a single graph.

Usage:
    python scripts/train_nar_decoder.py --data-dir data/norbi \
        --phoneme-vocab checkpoints/phoneme_vocab.json \
        --output-dir models/nar_decoder

    # quick smoke run on a tiny subset
    python scripts/train_nar_decoder.py --data-dir data/norbi \
        --max-samples 64 --epochs 1 --batch-size 4 --num-workers 0
"""

from __future__ import annotations

import argparse
import json
import math
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

from echo.nar_decoder import NARDecoder  # noqa: E402
from echo.config import (  # noqa: E402
    AUDIO_PAD_ID,
    CODEC_VOCAB_SIZE,
    MAX_AUDIO_LENGTH,
    MAX_TEXT_LENGTH,
    NAR_BATCH_SIZE,
    NAR_BETAS,
    NAR_D_FF,
    NAR_D_MODEL,
    NAR_DROPOUT,
    NAR_GRAD_CLIP,
    NAR_LABEL_SMOOTHING,
    NAR_LAYER_WEIGHT_DECAY,
    NAR_LEARNING_RATE,
    NAR_LOG_EVERY,
    NAR_N_HEADS,
    NAR_N_LAYERS,
    NAR_NUM_EPOCHS,
    NAR_NUM_WORKERS,
    NAR_SAVE_EVERY,
    NAR_SEED,
    NAR_VAL_EVERY,
    NAR_VAL_FRACTION,
    NAR_WARMUP_STEPS,
    NAR_WEIGHT_DECAY,
    NUM_CODEBOOKS,
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
# Tokenizer (identical to the AR trainer)
# ---------------------------------------------------------------------------


class PhonemeTokenizer:
    """Map a phoneme string to token ids, one per Unicode code point."""

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
            if idx in (0, 1, 2):  # skip <pad>/<bos>/<eos>
                continue
            ids.append(idx)
        return ids


# ---------------------------------------------------------------------------
# Dataset -- returns ALL codec layers
# ---------------------------------------------------------------------------


class CodecDataset(Dataset):
    """Preloaded ``(phoneme_ids, full-codec-grid)`` pairs.

    ``full-codec-grid`` is a ``(num_layers, T)`` int64 array (all 16 Mimi
    codebooks), unlike the AR trainer which keeps only the first codebook.
    """

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
        self.codes_list: list[np.ndarray] = []
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
                    codes = np.load(codec_path)["codes"].astype(np.int64)  # (L, T)
                except Exception as e:  # noqa: BLE001
                    print_error(f"Skipping {name}: failed to load codes: {e}")
                    skipped += 1
                    continue

                if codes.ndim != 2 or codes.shape[0] < NUM_CODEBOOKS:
                    print_error(f"Skipping {name}: expected >= {NUM_CODEBOOKS} codec "
                                f"layers, got shape {codes.shape}")
                    skipped += 1
                    continue
                codes = codes[:NUM_CODEBOOKS]  # (16, T)
                T = codes.shape[1]

                # Filter over-length / empty samples.
                if len(text_ids) == 0 or len(text_ids) > max_text_length:
                    skipped += 1
                    continue
                if T == 0 or T > max_audio_length:
                    skipped += 1
                    continue
                # Sanity: every codebook token must be a real codec id.
                if codes.min() < 0 or codes.max() >= CODEC_VOCAB_SIZE:
                    print_error(f"Skipping {name}: codec token out of range "
                                f"[0,{CODEC_VOCAB_SIZE})")
                    skipped += 1
                    continue

                self.text_list.append(text_ids)
                self.codes_list.append(codes)
                self.names.append(name)

                if max_samples is not None and len(self.text_list) >= max_samples:
                    break

        print_info("Loaded samples", str(len(self.text_list)), Colors.OKCYAN)
        if skipped:
            print_info("Skipped samples", str(skipped), Colors.WARNING)

    def __len__(self) -> int:
        return len(self.text_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.text_list[idx], self.codes_list[idx]


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def make_collate():
    return CodecCollate()


class CodecCollate:
    """Pad a batch of ``(phoneme_ids, codec_grid)`` and build the tensors.

    * ``text_tokens``      -- ``(B, Tt)`` right-padded with ``TEXT_PAD_ID``.
    * ``codec_tokens``     -- ``(B, 16, Ta)`` right-padded with ``AUDIO_PAD_ID``.
    * ``key_padding_mask`` -- ``(B, Tt+1+Ta)`` bool, ``True`` = ignore.
    * ``audio_valid``      -- ``(B, Ta)`` bool, ``True`` on real audio frames
      (the positions where loss is computed, for every target layer).
    * ``text_pad_start``   -- ``Tt``, the index of ``<SEP>`` in the sequence.
    """

    def __call__(self, batch: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        texts = [b[0] for b in batch]
        grids = [b[1] for b in batch]  # each (16, T_i)
        B = len(batch)
        L = grids[0].shape[0]

        Tt = max(len(t) for t in texts)
        Ta = max(g.shape[1] for g in grids)

        text_tokens = np.full((B, Tt), TEXT_PAD_ID, dtype=np.int64)
        codec_tokens = np.full((B, L, Ta), AUDIO_PAD_ID, dtype=np.int64)
        audio_valid = np.zeros((B, Ta), dtype=bool)
        text_lens = np.zeros(B, dtype=np.int64)
        audio_lens = np.zeros(B, dtype=np.int64)

        for i, (t, g) in enumerate(zip(texts, grids)):
            lt, la = len(t), g.shape[1]
            text_tokens[i, :lt] = t
            codec_tokens[i, :, :la] = g
            audio_valid[i, :la] = True
            text_lens[i] = lt
            audio_lens[i] = la

        T_total = Tt + 1 + Ta
        key_padding_mask = np.zeros((B, T_total), dtype=bool)
        for i in range(B):
            lt = int(text_lens[i])
            la = int(audio_lens[i])
            key_padding_mask[i, lt:Tt] = True          # padded text
            key_padding_mask[i, Tt + 1 + la:] = True    # padded audio
            # SEP at index Tt is never padded.

        return {
            "text_tokens": torch.from_numpy(text_tokens),
            "codec_tokens": torch.from_numpy(codec_tokens),
            "key_padding_mask": torch.from_numpy(key_padding_mask),
            "audio_valid": torch.from_numpy(audio_valid),
            "text_lens": torch.from_numpy(text_lens),
            "audio_lens": torch.from_numpy(audio_lens),
            "text_pad_start": Tt,
        }


# ---------------------------------------------------------------------------
# Per-layer loss
# ---------------------------------------------------------------------------


def layer_logits_and_target(
    model: NARDecoder, batch: dict, target_layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model for one target codebook and return ``(audio_logits, target)``.

    ``audio_logits`` is ``(B, Ta, V)`` -- the logits at the audio positions
    only (no shift; position ``j`` predicts codebook ``target_layer`` at frame
    ``j``).  ``target`` is ``(B, Ta)`` with ``IGNORE_INDEX`` on padded frames.
    """
    text = batch["text_tokens"]
    codec = batch["codec_tokens"]                 # (B, L, Ta)
    Tt = batch["text_pad_start"]
    Ta = codec.shape[2]

    prev = codec[:, :target_layer, :]             # layers 0..target_layer-1
    logits = model(text, prev, layer_idx=target_layer,
                   key_padding_mask=batch["key_padding_mask"])
    # Sequence layout: [text(Tt), <SEP>(1), audio(Ta)]  ->  audio at Tt+1 ...
    audio_logits = logits[:, Tt + 1: Tt + 1 + Ta, :]

    target = codec[:, target_layer, :].clone()
    target[~batch["audio_valid"]] = IGNORE_INDEX
    return audio_logits, target


def layer_ce(audio_logits: torch.Tensor, target: torch.Tensor,
             label_smoothing: float, reduction: str) -> torch.Tensor:
    return F.cross_entropy(
        audio_logits.reshape(-1, audio_logits.size(-1)),
        target.reshape(-1),
        ignore_index=IGNORE_INDEX,
        label_smoothing=label_smoothing,
        reduction=reduction,
    )


def layer_weights(num_pred_layers: int, decay: float) -> list[float]:
    """Normalized geometric weights: index ``k`` (codebook ``k+1``) -> decay**k."""
    raw = [decay ** k for k in range(num_pred_layers)]
    s = sum(raw)
    return [r / s for r in raw]


# ---------------------------------------------------------------------------
# Optimizer & scheduler (same conventions as the AR trainer)
# ---------------------------------------------------------------------------


def build_optimizer(model: NARDecoder, lr: float, weight_decay: float,
                    betas: tuple) -> torch.optim.Optimizer:
    """AdamW with decoupled weight decay on non-bias / non-norm / non-embedding."""
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


def build_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int,
                    total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
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
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


@torch.no_grad()
def evaluate(model: NARDecoder, loader: DataLoader, device: torch.device,
             weights: list[float], label_smoothing: float
             ) -> tuple[float, list[float]]:
    """Return ``(weighted_loss, per_layer_loss)``.

    ``per_layer_loss[i]`` is the mean CE for target codebook ``i`` (index 0 is
    unused / 0.0 since codebook 0 is produced by the AR model).  The weighted
    loss uses the same normalized geometric weights as training.
    """
    model.eval()
    n_layers = NUM_CODEBOOKS
    sums = [0.0] * n_layers
    cnts = [0.0] * n_layers
    for batch in loader:
        batch = move_batch(batch, device)
        for i in range(1, n_layers):
            audio_logits, target = layer_logits_and_target(model, batch, i)
            ce_sum = layer_ce(audio_logits, target, label_smoothing, reduction="sum")
            cnt = (target != IGNORE_INDEX).sum()
            sums[i] += ce_sum.item()
            cnts[i] += cnt.item()

    model.train()
    per_layer = [0.0] * n_layers
    weighted = 0.0
    for i in range(1, n_layers):
        per_layer[i] = sums[i] / max(1.0, cnts[i])
        weighted += weights[i - 1] * per_layer[i]  # weights already normalized
    return weighted, per_layer


def save_checkpoint(path: Path, model: NARDecoder,
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
    p = argparse.ArgumentParser(description="Train the NAR decoder (codebooks 2..16).")
    p.add_argument("--data-dir", type=str, default="data/norbi")
    p.add_argument("--phoneme-vocab", type=str, default="checkpoints/phoneme_vocab.json")
    p.add_argument("--output-dir", type=str, default="models/nar_decoder")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint to resume from.")
    # overrides for hyperparams
    p.add_argument("--epochs", type=int, default=NAR_NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=NAR_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=NAR_LEARNING_RATE)
    p.add_argument("--warmup-steps", type=int, default=NAR_WARMUP_STEPS)
    p.add_argument("--weight-decay", type=float, default=NAR_WEIGHT_DECAY)
    p.add_argument("--grad-clip", type=float, default=NAR_GRAD_CLIP)
    p.add_argument("--label-smoothing", type=float, default=NAR_LABEL_SMOOTHING)
    p.add_argument("--layer-weight-decay", type=float, default=NAR_LAYER_WEIGHT_DECAY,
                   help="Geometric decay for per-codebook loss weights "
                        "(codebook i weight = decay**(i-1); 1.0 = equal weight).")
    p.add_argument("--num-workers", type=int, default=NAR_NUM_WORKERS)
    p.add_argument("--val-fraction", type=float, default=NAR_VAL_FRACTION)
    p.add_argument("--val-every", type=int, default=NAR_VAL_EVERY)
    p.add_argument("--save-every", type=int, default=NAR_SAVE_EVERY)
    p.add_argument("--log-every", type=int, default=NAR_LOG_EVERY)
    p.add_argument("--seed", type=int, default=NAR_SEED)
    p.add_argument("--max-samples", type=int, default=None,
                   help="Use only the first N samples (for smoke tests).")
    p.add_argument("--bf16", action="store_true",
                   help="Use bfloat16 autocast on CUDA.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print_header("NAR Decoder - Training")
    print_separator()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device("auto" if args.device == "auto" else args.device)
    n_pred_layers = NUM_CODEBOOKS - 1  # codebooks 1..15
    weights = layer_weights(n_pred_layers, args.layer_weight_decay)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Data dir", str(args.data_dir))
    print_info("Output dir", str(args.output_dir))
    print_info("Batch size", str(args.batch_size))
    print_info("Learning rate (peak)", str(args.lr))
    print_info("Warmup steps", str(args.warmup_steps))
    print_info("Epochs", str(args.epochs))
    print_info("Label smoothing", str(args.label_smoothing))
    print_info("Predicted codebooks", f"1..{NUM_CODEBOOKS - 1} ({n_pred_layers} layers)")
    print_info("Layer weight decay", str(args.layer_weight_decay))
    print_info("Layer weights (2nd..last)",
               "[" + ", ".join(f"{w:.3f}" for w in weights) + "]", Colors.OKCYAN)
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
    model = NARDecoder(
        d_model=NAR_D_MODEL, n_heads=NAR_N_HEADS, d_ff=NAR_D_FF,
        n_layers=NAR_N_LAYERS, dropout=NAR_DROPOUT,
    ).to(device)
    print_info("Total parameters", f"{model.num_parameters():,}")
    print_info("Non-embedding parameters",
               f"{model.num_parameters(exclude_embeddings=True):,}", Colors.OKCYAN)

    # ---- optimizer / scheduler -------------------------------------------
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    optimizer = build_optimizer(model, args.lr, args.weight_decay, NAR_BETAS)
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

    def run_validation(tag: str) -> None:
        nonlocal best_val
        val_loss, per_layer = evaluate(model, val_loader, device,
                                       weights, args.label_smoothing)
        flag = " (new best)" if val_loss < best_val else ""
        print(f"  [{tag}] loss {val_loss:.4f} | "
              f"cb2 {per_layer[1]:.4f} | cb{NUM_CODEBOOKS} {per_layer[-1]:.4f}{flag}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(output_dir / "best.pt", model, optimizer,
                            scheduler, global_step, start_epoch, best_val)

    for epoch in range(start_epoch, args.epochs):
        for batch in train_loader:
            batch = move_batch(batch, device)

            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            # One target codebook at a time; back-prop each (gradient
            # accumulation) so only a single forward graph is alive at once.
            for i in range(1, NUM_CODEBOOKS):
                w = weights[i - 1]
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                    audio_logits, target = layer_logits_and_target(model, batch, i)
                    loss_i = layer_ce(audio_logits, target, args.label_smoothing,
                                      reduction="mean")
                    weighted_i = loss_i * w
                scaler.scale(weighted_i).backward()
                step_loss += weighted_i.item()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

            running_loss += step_loss
            running_cnt += 1

            if global_step % args.log_every == 0:
                avg = running_loss / running_cnt
                lr = scheduler.get_last_lr()[0]
                elapsed = time.perf_counter() - t0
                print(f"  step {global_step:>7} | epoch {epoch} | "
                      f"loss {avg:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                running_loss, running_cnt = 0.0, 0

            if global_step % args.val_every == 0:
                run_validation(f"val step {global_step}")

            if global_step % args.save_every == 0:
                save_checkpoint(output_dir / "latest.pt", model, optimizer,
                                scheduler, global_step, epoch, best_val)

        # end of epoch checkpoint + validation
        save_checkpoint(output_dir / "latest.pt", model, optimizer, scheduler,
                        global_step, epoch + 1, best_val)
        run_validation(f"epoch {epoch} val")

    print_separator("═", 60)
    print_success(f"Training done. Best val loss: {best_val:.4f}")
    print_info("Checkpoints", str(output_dir))


if __name__ == "__main__":
    main()
