#!/usr/bin/env python3
"""Precompute BlueCodec latents for a directory of audio files.

Each output is a compressed `.npz` holding one `latents` array of shape
`(num_channels, T_latent)`.

Usage:
    python scripts/preprocess/latents.py --audio-dir data/audio --output-dir data/latents
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
import torch
import torchaudio
from bluecodec import BlueCodec
from tqdm import tqdm

from __common__ import (
    BLUE_SR, find_audio_files, print_run_summary, save_npz, select_device,
)
from __style__ import Colors, print_error, print_header, print_info, print_section, print_separator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute BlueCodec latents for a directory of audio files."
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing input audio files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write .npz latent files to.")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Files loaded per batch (default: 8).")
    parser.add_argument("--ext", action="append",
                        help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N audio files.")
    args = parser.parse_args()

    audio_dir, output_dir = Path(args.audio_dir), Path(args.output_dir)
    if not audio_dir.is_dir():
        print_error(f"Audio directory not found: {audio_dir}")
        sys.exit(1)
    files = find_audio_files(audio_dir, args.ext, args.limit)
    if not files:
        print_error(f"No audio files found in {audio_dir}")
        sys.exit(1)

    device = select_device()

    print_header("BlueCodec Latents - Precompute")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Audio dir", str(audio_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Files found", f"{len(files)} (batches of {args.batch_size})")

    codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    print_section("Encoding")
    t_total = time.perf_counter()
    n_ok = n_fail = 0

    progress = tqdm(range(0, len(files), args.batch_size), desc="Encoding", unit="batch")
    for bi in progress:
        # Load + resample first, dropping whatever fails to read.
        loaded: list[tuple[Path, torch.Tensor]] = []
        for src in files[bi: bi + args.batch_size]:
            try:
                audio, sr = torchaudio.load(str(src), backend="soundfile")
                if sr != BLUE_SR:
                    audio = torchaudio.functional.resample(audio, sr, BLUE_SR)
                loaded.append((src, audio))
            except Exception as e:                                   # noqa: BLE001
                print_error(f"Failed to load {src.name}: {e}")
                n_fail += 1

        # One at a time: the BlueCodec encoder takes no variable-length batch.
        for src, audio in loaded:
            try:
                with torch.no_grad():
                    latents = codec.encode(audio.to(device))
                save_npz(
                    output_dir / src.relative_to(audio_dir).with_suffix(".npz"),
                    latents=latents.detach().cpu().numpy().astype(np.float32).squeeze(0),
                )
                n_ok += 1
            except Exception as e:                                   # noqa: BLE001
                print_error(f"Failed to encode/save {src.name}: {e}")
                n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    print_run_summary(n_ok, n_fail, time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
