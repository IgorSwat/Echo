#!/usr/bin/env python3
"""Precompute distillation latents: Mimi encode → truncate → decode → BlueCodec encode.

Pipeline: audio -> Mimi codec (first --layers layers) -> decoded audio -> BlueCodec latents.

Output files are ``.npz`` (zlib compressed) containing a single ``latents`` array
of shape ``(num_channels, T_latent)`` of float32 values.

Usage:
    python scripts/precompute_distils.py --audio-dir data/audio --output-dir data/distils
    python scripts/precompute_distils.py --audio-dir data/audio --output-dir data/distils --layers 4 --batch-size 4
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
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
import librosa  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import MimiModel, AutoFeatureExtractor  # noqa: E402
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


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

_MIMI_SR = 24000
_BLUE_SR = 44100
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
    parser = argparse.ArgumentParser(
        description="Precompute distillation latents: Mimi → truncate → decode → BlueCodec."
    )
    parser.add_argument("--audio-dir", type=str, required=True, help="Directory containing input audio files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write .npz latent files to.")
    parser.add_argument("--layers", type=int, default=16, help="Number of Mimi codec layers to keep (default: 16).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for Mimi encoding (default: 8).")
    parser.add_argument("--ext", action="append", help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N audio files (for testing).")
    args = parser.parse_args()

    print_header("Distillation Latents - Precompute (Mimi → BlueCodec)")
    print_separator()

    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

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
    print_info("Mimi layers", str(args.layers))
    print_info("Batch size", str(args.batch_size))
    print_info("Files found", str(len(files)))

    # --- Load Mimi -----------------------------------------------------------
    print_section("Loading Mimi model")
    t_model = time.perf_counter()
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    mimi_sr = feature_extractor.sampling_rate
    assert mimi_sr == _MIMI_SR, f"Expected Mimi sample rate {_MIMI_SR}, got {mimi_sr}"
    mimi_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{_MIMI_SR} Hz")
    print_info("Model load time", f"{mimi_time:.3f}s", Colors.OKCYAN)

    # --- Load BlueCodec ------------------------------------------------------
    print_section("Loading BlueCodec model")
    t_model = time.perf_counter()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    blue_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{_BLUE_SR} Hz")
    print_info("Latent channels", str(_NUM_CHANNELS))
    print_info("Model load time", f"{blue_time:.3f}s", Colors.OKCYAN)

    # --- Process in batches -------------------------------------------------
    print_section("Processing")
    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0

    progress = tqdm(range(0, len(files), args.batch_size), desc="Processing", unit="batch")
    for bi in progress:
        batch_files = files[bi : bi + args.batch_size]

        # Step 1: Load + resample to 24 kHz mono (Mimi input).
        audios = []
        for p in batch_files:
            try:
                a, _ = librosa.load(str(p), sr=_MIMI_SR, mono=True)
                audios.append(a)
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to load {p.name}: {e}")
                audios.append(None)

        ok_idx = [i for i, a in enumerate(audios) if a is not None]
        if not ok_idx:
            n_fail += len(batch_files)
            continue

        ok_files = [batch_files[i] for i in ok_idx]
        ok_audios = [audios[i] for i in ok_idx]
        n_fail += len(batch_files) - len(ok_files)

        # Step 2: Mimi encode (batched).
        lengths = np.array([len(a) for a in ok_audios], dtype=np.int64)
        max_len = int(lengths.max())
        batch_arr = np.zeros((len(ok_audios), max_len), dtype=np.float32)
        for i, a in enumerate(ok_audios):
            batch_arr[i, : len(a)] = a

        inputs = feature_extractor(
            raw_audio=batch_arr.tolist(),
            sampling_rate=_MIMI_SR,
            return_tensors="pt",
        )
        input_values = inputs["input_values"].to(device)

        with torch.no_grad():
            enc = mimi.encode(input_values)
        codes = enc.audio_codes[:, : args.layers, :]  # (B, layers, T_codec)

        # Step 3: Mimi decode back to audio (batched).
        with torch.no_grad():
            dec_out = mimi.decode(codes)
        # dec_out returns (audio_values, ...) or MimiDecoderOutput
        if isinstance(dec_out, tuple):
            decoded_audio = dec_out[0]
        else:
            decoded_audio = dec_out.audio_values
        # decoded_audio: (B, 1, T_audio) — trim to original lengths
        decoded_audio = decoded_audio.squeeze(1)  # (B, T_audio)

        # Step 4: Resample each decoded audio to 44.1 kHz and encode with BlueCodec.
        for j, src in enumerate(ok_files):
            try:
                orig_len = int(lengths[j])
                audio_mimi = decoded_audio[j, :orig_len].unsqueeze(0)  # (1, T)
                # Resample 24kHz -> 44.1kHz
                audio_441 = torchaudio.functional.resample(audio_mimi, _MIMI_SR, _BLUE_SR)

                audio_dev = audio_441.to(device)
                with torch.no_grad():
                    latents = blue.encode(audio_dev)
                latents_np = latents.detach().cpu().numpy().astype(np.float32)
                latents_np = latents_np.squeeze(0)  # (1, C, T) -> (C, T)

                rel = src.relative_to(audio_dir)
                out_path = output_dir / rel.with_suffix(".npz")
                _save_latents(latents_np, out_path)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to process {src.name}: {e}")
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