#!/usr/bin/env python3
"""Precompute Mimi codec tokens for a directory of audio files.

Each output is a compressed `.npz` holding one `codes` array of shape
`(num_layers, T_codec)` of int32 ids in `[0, 2048)`.

Usage:
    python scripts/preprocess/codecs.py --audio-dir data/audio --output-dir data/codecs
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel

from __common__ import (
    MIMI_SR, find_audio_files, print_run_summary, save_npz, select_device,
)
from __style__ import Colors, print_error, print_header, print_info, print_section, print_separator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Mimi codec tokens for a directory of audio files."
    )
    parser.add_argument("--audio-dir", type=str, required=True,
                        help="Directory containing input audio files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write .npz codec files to.")
    parser.add_argument("--layers", type=int, default=16,
                        help="Codec layers to keep (default: 16).")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for encoding (default: 8).")
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

    print_header("Mimi Codec - Precompute")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Audio dir", str(audio_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Codec layers", str(args.layers))
    print_info("Files found", f"{len(files)} (batches of {args.batch_size})")

    model = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    assert feature_extractor.sampling_rate == MIMI_SR, (
        f"Expected Mimi sample rate {MIMI_SR}, got {feature_extractor.sampling_rate}"
    )

    print_section("Encoding")
    t_total = time.perf_counter()
    n_ok = n_fail = 0

    progress = tqdm(range(0, len(files), args.batch_size), desc="Encoding", unit="batch")
    for bi in progress:
        # Load + resample to 24 kHz mono, dropping whatever fails to read.
        loaded: list[tuple[Path, np.ndarray]] = []
        for src in files[bi: bi + args.batch_size]:
            try:
                audio, _ = librosa.load(str(src), sr=MIMI_SR, mono=True)
                loaded.append((src, audio))
            except Exception as e:                                   # noqa: BLE001
                print_error(f"Failed to load {src.name}: {e}")
                n_fail += 1

        if not loaded:
            continue

        # Zero-pad to a common length within the batch.
        batch = np.zeros((len(loaded), max(len(a) for _, a in loaded)), dtype=np.float32)
        for i, (_, audio) in enumerate(loaded):
            batch[i, : len(audio)] = audio

        inputs = feature_extractor(
            raw_audio=batch.tolist(), sampling_rate=MIMI_SR, return_tensors="pt",
        )
        with torch.no_grad():
            enc = model.encode(inputs["input_values"].to(device))
        codes = enc.audio_codes[:, : args.layers, :].cpu().numpy().astype(np.int32)

        # Mirror each file's path under audio_dir into output_dir.
        for i, (src, _) in enumerate(loaded):
            out_path = output_dir / src.relative_to(audio_dir).with_suffix(".npz")
            try:
                save_npz(out_path, codes=codes[i])
                n_ok += 1
            except Exception as e:                                   # noqa: BLE001
                print_error(f"Failed to save {out_path}: {e}")
                n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    print_run_summary(n_ok, n_fail, time.perf_counter() - t_total)


if __name__ == "__main__":
    main()
