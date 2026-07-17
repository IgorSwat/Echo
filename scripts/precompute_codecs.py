#!/usr/bin/env python3
"""Precompute Mimi codec tokens for a directory of audio files.

Encodes each audio file with the Kyutai Mimi codec and saves the resulting
discrete token grid to ``--output-dir``. Output files are ``.npz`` (zlib
compressed) containing a single ``codes`` array of shape
``(num_layers, T_audio)`` of int32 token ids in ``[0, 2048)``.

Usage:
    python scripts/precompute_codecs.py --audio-dir data/audio --output-dir data/codecs
    python scripts/precompute_codecs.py --audio-dir data/audio --output-dir data/codecs --batch-size 8
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
import librosa  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import MimiModel, AutoFeatureExtractor  # noqa: E402

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

# Mimi operates at 24 kHz. We resample all inputs to this rate.
_TARGET_SR = 24000


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _pad_batch(audios: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of 1D arrays to a common length and return (batch, lengths).

    Padding is zero-valued; a boolean mask of the same length marks real
    samples (False = real, True = padded), matching the ``attention_mask``
    convention used by the HuggingFace feature extractor.
    """
    lengths = np.array([len(a) for a in audios], dtype=np.int64)
    max_len = int(lengths.max())
    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    mask = np.ones((len(audios), max_len), dtype=bool)
    for i, a in enumerate(audios):
        L = len(a)
        batch[i, :L] = a
        mask[i, :L] = False
    return batch, mask


def _save_codes(codes: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, codes=codes.astype(np.int32))


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute Mimi codec tokens for a directory of audio files.")
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing input audio files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write .npz codec files to.")
    parser.add_argument("--layers", type=int, default=16, help="Number of codec layers to keep (default: 16).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for encoding (default: 8).")
    parser.add_argument("--ext", action="append", help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N audio files (for testing).")
    args = parser.parse_args()

    print_header("Mimi Codec - Precompute")
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
    print_info("Codec layers", str(args.layers))
    print_info("Batch size", str(args.batch_size))
    print_info("Files found", str(len(files)))

    # --- Load model ---------------------------------------------------------
    print_section("Loading Mimi model")
    t_model = time.perf_counter()
    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    model_sr = feature_extractor.sampling_rate
    assert model_sr == _TARGET_SR, f"Expected Mimi sample rate {_TARGET_SR}, got {model_sr}"
    model_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{model_sr} Hz")
    print_info("Model load time", f"{model_time:.3f}s", Colors.OKCYAN)

    # --- Process in batches -------------------------------------------------
    print_section("Encoding")
    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0

    progress = tqdm(range(0, len(files), args.batch_size), desc="Encoding", unit="batch")
    for bi in progress:
        batch_files = files[bi : bi + args.batch_size]

        # Load + resample to 24 kHz mono.
        audios = []
        for p in batch_files:
            try:
                a, _ = librosa.load(str(p), sr=_TARGET_SR, mono=True)
                audios.append(a)
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to load {p.name}: {e}")
                audios.append(None)

        # Drop any loads that failed; record the failure and keep the rest
        # aligned with ``batch_files`` so we can still save the successes.
        ok_idx = [i for i, a in enumerate(audios) if a is not None]
        if not ok_idx:
            n_fail += len(batch_files)
            continue

        ok_files = [batch_files[i] for i in ok_idx]
        ok_audios = [audios[i] for i in ok_idx]
        n_fail += len(batch_files) - len(ok_files)

        # Pad to a common length within the batch.
        batch_arr, _ = _pad_batch(ok_audios)
        inputs = feature_extractor(
            raw_audio=batch_arr.tolist(),
            sampling_rate=_TARGET_SR,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)
        # Mimi encode: returns audio_codes of shape (B, num_layers, T_codec).
        with torch.no_grad():
            enc = model.encode(input_values)
        codes = enc.audio_codes[:, : args.layers, :]   # (B, layers, T_codec)
        codes_np = codes.detach().cpu().numpy().astype(np.int32)

        # Per-file save: rel path under audio_dir -> same rel path under output_dir.
        for j, src in enumerate(ok_files):
            rel = src.relative_to(audio_dir)
            out_path = output_dir / rel.with_suffix(".npz")
            try:
                _save_codes(codes_np[j], out_path)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to save {out_path}: {e}")
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
