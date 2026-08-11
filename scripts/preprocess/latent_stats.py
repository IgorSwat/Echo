#!/usr/bin/env python3
"""Compute per-channel normalization statistics over a latent dataset.

Scans every ``.npz`` written by ``precompute_latents.py`` and accumulates running
per-channel sums, so the whole dataset never has to fit in memory. Writes a
single ``latent_stats.npz`` holding ``mean`` and ``std`` arrays of shape
``(num_channels,)``, which :class:`echo.training.dataset.EchoDataset` uses to
z-score latents at load time.

Usage:
    python scripts/preprocess/latent_stats.py --latent-dir data/latents
    python scripts/preprocess/latent_stats.py --latent-dir data/latents \\
        --output data/latents/latent_stats.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time

import numpy as np
from tqdm import tqdm

from __style__ import (
    Colors,
    print_error,
    print_header,
    print_info,
    print_section,
    print_separator,
    print_success,
)

# Floor on the per-channel variance, to avoid dividing by zero on a dead channel.
_EPS = 1e-6

_STATS_FILENAME = "latent_stats.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std statistics over a latent dataset."
    )
    parser.add_argument("--latent-dir", type=str, required=True,
                        help="Directory of .npz latent files (each with a 'latents' key "
                             "of shape (C, T)).")
    parser.add_argument("--output", type=str, default=None,
                        help=f"Output stats path (default: <latent-dir>/{_STATS_FILENAME}).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only scan the first N latent files (for testing).")
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir)
    if not latent_dir.is_dir():
        print_error(f"Latent directory not found: {latent_dir}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else latent_dir / _STATS_FILENAME

    print_header("BlueCodec Latents - Compute Stats")
    print_separator()

    # --- Discover latent files ----------------------------------------------
    # A pre-existing stats file inside the same directory is not input.
    files = [p for p in sorted(latent_dir.rglob("*.npz")) if p.name != _STATS_FILENAME]
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print_error(f"No .npz latent files found in {latent_dir}")
        sys.exit(1)

    print_section("Input")
    print_info("Latent dir", str(latent_dir), Colors.OKCYAN)
    print_info("Output", str(out_path), Colors.OKCYAN)
    print_info("Files found", str(len(files)))

    # --- Accumulate running per-channel stats -------------------------------
    # Each file contributes a (C, T) array; per-channel sum, sum of squares and
    # the total frame count are enough to derive mean/var/std at the end.
    channel_sum: np.ndarray | None = None                            # (C,)
    channel_sumsq: np.ndarray | None = None                          # (C,)
    num_channels: int | None = None
    total_count = 0
    n_ok, n_fail = 0, 0

    t_total = time.perf_counter()
    progress = tqdm(files, desc="Scanning", unit="file")
    for p in progress:
        try:
            arr = np.load(p)["latents"]                              # (C, T)
            if arr.ndim != 2:
                print_error(f"Unexpected shape {arr.shape} in {p.name} (expected (C, T))")
                n_fail += 1
                continue

            C, T = arr.shape
            if num_channels is None:
                num_channels = C
                channel_sum = np.zeros(C, dtype=np.float64)
                channel_sumsq = np.zeros(C, dtype=np.float64)
            elif C != num_channels:
                print_error(f"Channel mismatch in {p.name}: {C} != {num_channels}")
                n_fail += 1
                continue

            arr64 = arr.astype(np.float64)
            channel_sum += arr64.sum(axis=1)
            channel_sumsq += (arr64 * arr64).sum(axis=1)
            total_count += T
            n_ok += 1
        except Exception as e:                                       # noqa: BLE001
            print_error(f"Failed to read {p.name}: {e}")
            n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    elapsed = time.perf_counter() - t_total

    if n_ok == 0 or total_count == 0 or channel_sum is None:
        print_error("No latent data accumulated; nothing to write.")
        sys.exit(1)

    mean = channel_sum / total_count
    var = np.maximum(channel_sumsq / total_count - mean * mean, _EPS)
    std = np.sqrt(var)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32))

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Files scanned", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    print_info("Total frames", str(total_count), Colors.OKCYAN)
    print_info("Channels", str(num_channels), Colors.OKCYAN)
    print_info("Elapsed", f"{elapsed:.2f}s", Colors.OKCYAN)

    print_section("Per-channel statistics")
    for c in range(num_channels):
        print_info(f"ch{c:02d}", f"mean={mean[c]:+.4f}  std={std[c]:.4f}", Colors.OKCYAN)
    print_success(f"Saved stats -> {out_path}")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
