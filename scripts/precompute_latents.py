#!/usr/bin/env python3
"""Precompute BlueCodec latents for a directory of audio files.

Encodes each audio file with the Blue audio codec and saves the resulting
continuous latent grid to ``--output-dir``. Output files are ``.npz`` (zlib
compressed) containing a single ``latents`` array of shape
``(num_channels, T_latent)`` of float32 values.

Usage:
    python scripts/precompute_latents.py --audio-dir data/audio --output-dir data/latents
    python scripts/precompute_latents.py --audio-dir data/audio --output-dir data/latents --batch-size 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch  # noqa: E402
import torchaudio  # noqa: E402
from tqdm import tqdm  # noqa: E402
from bluecodec import BlueCodec  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)


# Audio file extensions we look for.
_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

# BlueCodec operates at 44.1 kHz.
_TARGET_SR = 44100

# Number of latent channels output by BlueCodec.
_NUM_CHANNELS = 24


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_latents(latents: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, latents=latents.astype(np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute BlueCodec latents for a directory of audio files.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing input audio files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write .npz latent files to.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding (default: 8).")
    parser.add_argument("--ext", action="append", help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N audio files (for testing).")
    args = parser.parse_args()

    print_header("BlueCodec Latents - Precompute")
    print_separator()

    # --- Device -------------------------------------------------------------
    device = _select_device()
    print_section("Device")
    print_info("Selected", str(device), Colors.OKCYAN)

    # --- Discover audio files -----------------------------------------------
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    if not audio_dir.is_dir():
        print_error(f"Audio directory not found: {audio_dir}")
        sys.exit(1)

    exts = set(_AUDIO_EXTS)
    if args.ext:
        exts.update(e.lower() for e in args.ext)

    files = sorted(
        p for p in audio_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print_error(f"No audio files found in {audio_dir}")
        sys.exit(1)

    print_section("Input")
    print_info("Audio dir", str(audio_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Batch size", str(args.batch_size))
    print_info("Files found", str(len(files)))

    # --- Load model ---------------------------------------------------------
    print_section("Loading BlueCodec model")
    t_model = time.perf_counter()
    codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    model_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{_TARGET_SR} Hz")
    print_info("Latent channels", str(_NUM_CHANNELS))
    print_info("Model load time", f"{model_time:.3f}s", Colors.OKCYAN)

    # --- Process in batches -------------------------------------------------
    print_section("Encoding")
    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0

    progress = tqdm(range(0, len(files), args.batch_size), desc="Encoding", unit="batch")
    for bi in progress:
        batch_files = files[bi : bi + args.batch_size]

        # Load + resample to 44.1 kHz.
        audios = []
        for p in batch_files:
            try:
                a, sr = torchaudio.load(str(p), backend="soundfile")
                if sr != _TARGET_SR:
                    a = torchaudio.functional.resample(a, sr, _TARGET_SR)
                audios.append(a)
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to load {p.name}: {e}")
                audios.append(None)

        # Drop any loads that failed.
        ok_idx = [i for i, a in enumerate(audios) if a is not None]
        if not ok_idx:
            n_fail += len(batch_files)
            continue

        ok_files = [batch_files[i] for i in ok_idx]
        ok_audios = [audios[i] for i in ok_idx]
        n_fail += len(batch_files) - len(ok_files)

        # Encode each file individually (BlueCodec encoder may not support
        # variable-length batched input).
        for j, (src, audio) in enumerate(zip(ok_files, ok_audios)):
            try:
                audio_dev = audio.to(device)
                with torch.no_grad():
                    latents = codec.encode(audio_dev)
                latents_np = latents.detach().cpu().numpy().astype(np.float32)
                # Squeeze batch dim: (1, C, T) -> (C, T)
                latents_np = latents_np.squeeze(0)

                rel = src.relative_to(audio_dir)
                out_path = output_dir / rel.with_suffix(".npz")
                _save_latents(latents_np, out_path)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to encode/save {src.name}: {e}")
                n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    elapsed = time.perf_counter() - t_total

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Files processed", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()