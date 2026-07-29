#!/usr/bin/env python3
"""Compute per-channel normalization statistics over a latent dataset.

Scans every ``.npz`` file produced by ``precompute_latents.py`` and accumulates
running per-channel sums so the whole dataset does not need to fit in memory.
Writes a single ``latent_stats.npz`` containing ``mean`` and ``std`` arrays of
shape ``(num_channels,)`` which :class:`echo.training.dataset.EchoDataset` uses
to z-score the latents at load time.

Usage:
    python scripts/compute_latent_stats.py --latent-dir data/latents
    python scripts/compute_latent_stats.py --latent-dir data/latents --output data/latents/latent_stats.npz
"""

from __future__ import annotations

import argparse
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

from tqdm import tqdm  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)


# Floor on the per-channel variance to avoid division by zero for dead
# channels.
_EPS = 1e-6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-channel mean/std statistics over a latent dataset."
    )
    parser.add_argument(
        "--latent-dir", type=str, required=True,
        help="Directory containing .npz latent files (each with a 'latents' key of shape (C, T)).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output stats file path (default: <latent-dir>/latent_stats.npz).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scan the first N latent files (for testing).",
    )
    args = parser.parse_args()

    latent_dir = Path(args.latent_dir)
    if not latent_dir.is_dir():
        print_error(f"Latent directory not found: {latent_dir}")
        sys.exit(1)

    out_path = Path(args.output) if args.output else latent_dir / "latent_stats.npz"

    print_header("BlueCodec Latents - Compute Stats")
    print_separator()

    # --- Discover latent files ---------------------------------------------
    files = sorted(latent_dir.rglob("*.npz"))
    # Exclude a pre-existing stats file if it lives inside the same dir.
    files = [p for p in files if p.name != "latent_stats.npz"]
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        print_error(f"No .npz latent files found in {latent_dir}")
        sys.exit(1)

    print_section("Input")
    print_info("Latent dir", str(latent_dir), Colors.OKCYAN)
    print_info("Output", str(out_path), Colors.OKCYAN)
    print_info("Files found", str(len(files)))

    # --- Accumulate running per-channel stats ------------------------------
    # Each file contributes an array of shape (C, T). We accumulate per-channel
    # sum, sum of squares, and total frame count, then derive mean/var/std.
    channel_sum: np.ndarray | None = None      # (C,)
    channel_sumsq: np.ndarray | None = None    # (C,)
    total_count = 0                            # total frames across dataset
    num_channels: int | None = None

    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0

    progress = tqdm(files, desc="Scanning", unit="file")
    for p in progress:
        try:
            arr = np.load(p)["latents"]        # (C, T)
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
        except Exception as e:  # noqa: BLE001
            print_error(f"Failed to read {p.name}: {e}")
            n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    elapsed = time.perf_counter() - t_total

    if n_ok == 0 or total_count == 0 or channel_sum is None:
        print_error("No latent data accumulated; nothing to write.")
        sys.exit(1)

    mean = channel_sum / total_count
    var = channel_sumsq / total_count - mean * mean
    var = np.maximum(var, _EPS)
    std = np.sqrt(var)

    # --- Write stats -------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, mean=mean.astype(np.float32), std=std.astype(np.float32))

    # --- Summary -----------------------------------------------------------
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
