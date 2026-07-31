#!/usr/bin/env python3
"""Benchmark Echo inference latency on random data.

Measures forward-pass wall-clock time for combinations of text and audio
sequence lengths.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --warmup 5 --iters 20
    python scripts/benchmark.py --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch

from __style__ import (
    Colors,
    print_header,
    print_info,
    print_section,
    print_separator,
    print_test_title,
)

from echo import config
from echo.fm_model import EchoFM

T_TEXT_VALUES = [32, 64, 128]
T_AUDIO_VALUES = [200, 400, 600, 800]


def _fmt_ms(secs: float) -> str:
    return f"{secs * 1000:.1f} ms"


def _time_forward(
    model: EchoFM,
    text: torch.Tensor,
    latent: torch.Tensor,
    time_tensor: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> float:
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(text, latent, time_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Timed
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            model(text, latent, time_tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Echo inference.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = EchoFM().to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())

    print_test_title("Echo — Inference Benchmark")
    print_info("Device", device)
    print_info("Dtype", next(model.parameters()).dtype)
    print_info("Parameters", f"{total_params:,}")
    print_info("Warmup iters", args.warmup)
    print_info("Timed iters", args.iters)
    print_info("Batch size", args.batch_size)
    print_info("Latent dim", config.latent_dim)
    print_info("Text emb dim", config.text_embedding_dim)

    # --- Results table ---
    print_section("Latency (ms per forward pass)")

    # Header
    col_w = 12
    label_w = 14
    header_cells = [f"{'T_audio':>{label_w}}"] + [f"{t:>{col_w}}" for t in T_TEXT_VALUES]
    print(f"  {'T_text ->':>{label_w}}" + "".join(c for c in header_cells[1:]))
    print(f"  {'':>{label_w}}" + "".join(f"{'T_text=' + str(t):>{col_w}}" for t in T_TEXT_VALUES))
    print("  " + "─" * (label_w + col_w * len(T_TEXT_VALUES)))

    for t_audio in T_AUDIO_VALUES:
        row_cells = [f"{t_audio:>{label_w}}"]
        for t_text in T_TEXT_VALUES:
            B = args.batch_size
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            latent = torch.randn(B, t_audio, config.latent_dim, device=device)
            time_tensor = torch.rand(B, device=device)

            secs = _time_forward(
                model, text, latent, time_tensor,
                warmup=args.warmup, iters=args.iters, device=device,
            )
            row_cells.append(f"{_fmt_ms(secs):>{col_w}}")
        print("  " + "".join(row_cells))

    print_separator()

    # --- Throughput summary ---
    print_section("Throughput (samples/sec at batch_size={})".format(args.batch_size))

    fastest = 0.0
    slowest = float("inf")
    for t_audio in T_AUDIO_VALUES:
        for t_text in T_TEXT_VALUES:
            B = args.batch_size
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            latent = torch.randn(B, t_audio, config.latent_dim, device=device)
            time_tensor = torch.rand(B, device=device)

            secs = _time_forward(
                model, text, latent, time_tensor,
                warmup=args.warmup, iters=args.iters, device=device,
            )
            throughput = B / secs
            fastest = max(fastest, throughput)
            slowest = min(slowest, throughput)

    print_info("Fastest", f"{fastest:.1f} samples/s", Colors.OKGREEN)
    print_info("Slowest", f"{slowest:.1f} samples/s", Colors.WARNING)


if __name__ == "__main__":
    main()
