from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import math
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from __style__ import Colors, print_info, print_separator, print_success

from echo import config
from echo.config import TrainingConfig
from echo.tokenizer import Tokenizer
from echo.training.collate import collate_fn


# Mimi: 24 kHz audio on a 12.5 Hz token grid, i.e. 1920 samples per codec frame.
MIMI_SR = 24000
MIMI_FPS = 12.5
MIMI_FRAME = 1920

# BlueCodec: 44.1 kHz, 512-sample hop, 24 latent channels.
BLUE_SR = 44100
BLUE_HOP = 512
BLUE_CHANNELS = 24

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}


# ---------
# Runtime
# ---------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def sync_device(device: torch.device) -> None:
    """Drain the queue so wall-clock timings reflect work actually finished."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# -------
# Files
# -------

def load_pairs_csv(path: str | Path) -> list[tuple[str, str]]:
    """Parse ``<name>|<value>`` lines, as written by the phonemizer."""
    pairs: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            name, value = line.split("|", 1)
            pairs.append((name.strip(), value.strip()))

    return pairs


def find_audio_files(
    directory: Path,
    extra_exts: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Every audio file under `directory`, sorted, optionally capped."""
    exts = set(AUDIO_EXTS)
    if extra_exts:
        exts.update(e.lower() for e in extra_exts)

    files = sorted(
        p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )

    return files[:limit] if limit is not None else files


def save_npz(out_path: Path, **arrays: np.ndarray) -> None:
    """Write a compressed ``.npz``, creating the parent directory as needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)


def load_tokenizer() -> Tokenizer:
    return Tokenizer(REPO_ROOT / "models" / "phoneme_vocab.json")


def print_run_summary(n_ok: int, n_fail: int, elapsed: float) -> None:
    """The closing block the preprocessing scripts share."""
    print_separator("═", 60)
    print_info("Files processed", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


# -------------
# Checkpoints
# -------------

def load_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    device: torch.device,
    what: str,
    strict: bool = True,
) -> dict:
    """Load a training checkpoint into `model` and return its raw payload.

    ``strict=False`` reports a partial match instead of failing, which lets an
    older checkpoint be evaluated against a model that has grown since.
    """
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    if not strict:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  {what}: {len(missing)} missing, {len(unexpected)} unexpected keys")
        return ckpt

    try:
        # EchoAR carries a training-only CTC head that inference never runs, so
        # it knows how to accept a checkpoint saved without one.
        if hasattr(model, "load_weights"):
            model.load_weights(state)
        else:
            model.load_state_dict(state)
    except RuntimeError as e:
        hint = ""
        if any("theta" in k for k in list(state) + [n for n, _ in model.named_parameters()]):
            hint = ("\nHint: cross-attention rope_norm in models/config.json must match what "
                    "the checkpoint was trained with ('query' stores `theta`, 'absolute' "
                    "stores `theta_q`/`theta_k`).")
        raise SystemExit(f"Could not load the {what} checkpoint {path}:\n{e}{hint}")

    return ckpt


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    val_loss: float,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "epoch": epoch,
            "val_loss": val_loss,
        },
        path,
    )


# ----------
# Training
# ----------

def split_loaders(
    dataset: Dataset, cfg: TrainingConfig
) -> tuple[DataLoader, DataLoader, Dataset, Dataset]:
    """Deterministic train/val split plus a loader for each half."""
    val_len = int(len(dataset) * cfg.val_ratio)
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, train_set, val_set


def build_optimizer(
    model: torch.nn.Module, cfg: TrainingConfig, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW with linear warmup into a cosine decay to zero."""
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def lr_at(step: int) -> float:
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)


# ------------------------
# Latent normalization
# ------------------------

Stats = tuple[torch.Tensor, torch.Tensor]


def load_latent_stats(stats_path: Path, device: torch.device) -> Stats | None:
    """
    Per-channel mean/std of shape ``(latent_dim,)``, or None if absent.
    """

    if not stats_path.is_file():
        return None

    stats = np.load(stats_path)

    return (
        torch.from_numpy(stats["mean"].astype(np.float32)).to(device),  # (C,)
        torch.from_numpy(stats["std"].astype(np.float32)).to(device),   # (C,)
    )


def norm_stats(distil: torch.Tensor, stats: Stats | None) -> Stats | None:
    """
    The (mean, std) that normalize a distil and denormalize the model's output.
    """
    if config.latent_norm == "instance":
        return (
            distil.mean(dim=-2, keepdim=True),
            distil.std(dim=-2, keepdim=True).clamp_min(1e-5),
        )

    return stats


# --------
# Codecs
# --------

def decode_mimi(mimi, codes: torch.Tensor) -> torch.Tensor:
    """
    Mimi-decode ``(B, layers, T)`` token ids into a ``(B, T_audio)`` waveform.
    """

    out = mimi.decode(codes)
    
    # decode() returns either a tuple or a MimiDecoderOutput.
    audio = out[0] if isinstance(out, tuple) else out.audio_values

    return audio.squeeze(1)                                          # (B, T_audio)


# ---------------
# Output filtering
# ---------------

def lowpass(audio: torch.Tensor, cutoff: float, sample_rate: int) -> torch.Tensor:
    """Zero-phase low-pass at ``cutoff`` Hz, leaving the sample rate untouched.

    This is a filter, not a resample: every sample is kept, only content above
    ``cutoff`` is removed. It exists because the source corpus is band-limited
    well below BlueCodec's 22 kHz Nyquist — LJSpeech is a 24 kHz recording, so
    nothing above 12 kHz is real, and the codec's decoder fills that empty top
    band with broadband noise anyway, audible as a sizzle riding on the speech.

    ``sosfiltfilt`` runs the filter forwards and backwards, so the result has no
    group delay and any trims applied around this step stay sample-accurate.
    """
    from scipy.signal import butter, sosfiltfilt                     # arrives with librosa

    sos = butter(8, cutoff / (sample_rate / 2), btype="low", output="sos")
    filtered = sosfiltfilt(sos, audio.numpy(), axis=-1).copy()       # negative strides -> copy

    return torch.from_numpy(filtered).float()
