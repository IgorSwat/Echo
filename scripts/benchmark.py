#!/usr/bin/env python3
"""Benchmark Echo inference latency on random data.

Measures forward-pass wall-clock time for combinations of text and audio
sequence lengths (EchoFM), and for both prefill and single-step decode
(EchoAR).

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --warmup 5 --iters 20
    python scripts/benchmark.py --device cuda
    python scripts/benchmark.py --model ar
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

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
from echo.ar_model import EchoAR
from echo.fm_model import EchoFM

T_TEXT_VALUES = [32, 64, 128]
T_AUDIO_VALUES = [200, 400, 600, 800]
T_TOKEN_VALUES = [200, 400, 600, 800]

COL_W = 12
LABEL_W = 14


def _fmt_ms(secs: float) -> str:
    return f"{secs * 1000:.1f} ms"


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _time(fn: Callable[[], object], warmup: int, iters: int, device: torch.device) -> float:
    """Average wall-clock seconds per call of `fn`."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    _sync(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
    _sync(device)
    elapsed = time.perf_counter() - start

    return elapsed / iters


def _print_grid(
    row_label: str,
    row_values: list[int],
    col_label: str,
    col_values: list[int],
    secs: dict[tuple[int, int], float],
) -> None:
    """Latency table: one row per `row_values` entry, one column per `col_values`."""
    print(f"  {row_label:>{LABEL_W}}" + "".join(
        f"{col_label + '=' + str(c):>{COL_W}}" for c in col_values
    ))
    print("  " + "─" * (LABEL_W + COL_W * len(col_values)))

    for r in row_values:
        cells = [f"{r:>{LABEL_W}}"] + [
            f"{_fmt_ms(secs[(r, c)]):>{COL_W}}" for c in col_values
        ]
        print("  " + "".join(cells))


def _print_throughput(secs: dict[tuple[int, int], float], batch_size: int) -> None:
    throughputs = [batch_size / s for s in secs.values()]
    print_info("Fastest", f"{max(throughputs):.1f} samples/s", Colors.OKGREEN)
    print_info("Slowest", f"{min(throughputs):.1f} samples/s", Colors.WARNING)


def benchmark_fm(args: argparse.Namespace, device: torch.device) -> None:
    model = EchoFM().to(device)
    model.eval()

    print_header("EchoFM — flow-matching backbone")
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Latent dim", config.latent_dim)
    print_info("Text emb dim", config.fm_model.text_embedding_dim)

    # --- Results table ---
    print_section("Latency (ms per forward pass)")

    B = args.batch_size
    secs: dict[tuple[int, int], float] = {}
    for t_audio in T_AUDIO_VALUES:
        for t_text in T_TEXT_VALUES:
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            latent = torch.randn(B, t_audio, config.latent_dim, device=device)
            time_tensor = torch.rand(B, device=device)

            secs[(t_audio, t_text)] = _time(
                lambda: model(text, latent, time_tensor),
                warmup=args.warmup, iters=args.iters, device=device,
            )

    _print_grid("T_audio", T_AUDIO_VALUES, "T_text", T_TEXT_VALUES, secs)

    print_separator()

    # --- Throughput summary ---
    print_section(f"Throughput (samples/sec at batch_size={B})")
    _print_throughput(secs, B)


def benchmark_ar(args: argparse.Namespace, device: torch.device) -> None:
    model = EchoAR().to(device)
    model.eval()

    print_header("EchoAR — autoregressive prosody model")
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Prosody vocab", f"{config.prosody_vocab_size:,}")
    print_info("Hidden dim", model.hidden_dim)

    B = args.batch_size
    V = config.prosody_vocab_size
    layers = model.NUM_TOKEN_LAYERS

    # --- Prefill: the whole token sequence in one pass, text encoded inline ---
    print_section("Prefill latency (ms per forward pass, text encoder included)")

    secs: dict[tuple[int, int], float] = {}
    for t_tokens in T_TOKEN_VALUES:
        for t_text in T_TEXT_VALUES:
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            x = torch.randint(0, V, (B, t_tokens, layers), device=device)

            secs[(t_tokens, t_text)] = _time(
                lambda: model(x, text),
                warmup=args.warmup, iters=args.iters, device=device,
            )

    _print_grid("T_tokens", T_TOKEN_VALUES, "T_text", T_TEXT_VALUES, secs)

    print_separator()
    print_section(f"Prefill throughput (samples/sec at batch_size={B})")
    _print_throughput(secs, B)

    # --- Decode: one token step, reusing an already-encoded text context ---
    print_section("Single-step decode (T=1, text context precomputed)")

    x_step = torch.randint(0, V, (B, 1, layers), device=device)

    print(f"  {'T_text':>{LABEL_W}}{'latency':>{COL_W}}{'steps/s':>{COL_W}}")
    print("  " + "─" * (LABEL_W + COL_W * 2))

    for t_text in T_TEXT_VALUES:
        text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
        with torch.no_grad():
            context = model.encode_text(text)                    # (B, S, d_text)

        step_secs = _time(
            lambda: model(x_step, context=context),
            warmup=args.warmup, iters=args.iters, device=device,
        )
        print(f"  {t_text:>{LABEL_W}}{_fmt_ms(step_secs):>{COL_W}}{1.0 / step_secs:>{COL_W}.1f}")

    print_separator()
    print_info(
        "Note",
        "no KV cache yet: a step attends over the given T only, so decode cost "
        "does not grow with history",
        Colors.WARNING,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Echo inference.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "fm", "ar"],
                        help="which model to benchmark (default: all)")
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

    print_test_title("Echo — Inference Benchmark")
    print_info("Device", device)
    print_info("Warmup iters", args.warmup)
    print_info("Timed iters", args.iters)
    print_info("Batch size", args.batch_size)

    if args.model in ("all", "fm"):
        print()
        benchmark_fm(args, device)
    if args.model in ("all", "ar"):
        print()
        benchmark_ar(args, device)


if __name__ == "__main__":
    main()
