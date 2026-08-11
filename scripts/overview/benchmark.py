#!/usr/bin/env python3
"""Benchmark Echo inference latency on random data, in pipeline order.

For EchoAR: prefill over a whole token sequence, and a single cached decode step
— the state real generation spends almost all its time in. For EchoFM: one
forward pass over combinations of text and audio sequence lengths.

Usage:
    python scripts/overview/benchmark.py
    python scripts/overview/benchmark.py --warmup 5 --iters 20
    python scripts/overview/benchmark.py --device cuda
    python scripts/overview/benchmark.py --model ar
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from typing import Callable

import torch

from __common__ import select_device, sync_device
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


def _time(fn: Callable[[], object], warmup: int, iters: int, device: torch.device) -> float:
    """Average wall-clock seconds per call of `fn`."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
    sync_device(device)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            fn()
    sync_device(device)

    return (time.perf_counter() - start) / iters


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
    # Intra-frame conditioning makes the lower token layers a required input.
    needs_cond = model.predictor.uses_cond

    # --- Prefill: the whole token sequence in one pass, text encoded inline ---
    print_section("Prefill latency (ms per forward pass, text encoder included)")

    secs: dict[tuple[int, int], float] = {}
    for t_tokens in T_TOKEN_VALUES:
        for t_text in T_TEXT_VALUES:
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            x = torch.randint(0, V, (B, t_tokens, layers), device=device)
            cond = x[..., :-1] if needs_cond else None

            secs[(t_tokens, t_text)] = _time(
                lambda: model(x, text, cond_tokens=cond),
                warmup=args.warmup, iters=args.iters, device=device,
            )

    _print_grid("T_tokens", T_TOKEN_VALUES, "T_text", T_TEXT_VALUES, secs)

    print_separator()
    print_section(f"Prefill throughput (samples/sec at batch_size={B})")
    _print_throughput(secs, B)

    # --- Decode: one cached step appended to an existing history ---
    # This is what `generate` actually does per frame, and where nearly all of
    # inference is spent: the text context and the first T_hist frames are
    # already in the caches, so only the new frame is embedded and decoded.
    print_section("Single-step decode (T=1, KV cache warm, text context precomputed)")

    secs = {}
    for t_hist in T_TOKEN_VALUES:
        for t_text in T_TEXT_VALUES:
            text = torch.randint(0, config.text_vocab_size, (B, t_text), device=device)
            x_hist = torch.randint(0, V, (B, t_hist, layers), device=device)

            with torch.no_grad():
                context = model.encode_text(text)                    # (B, S, d_text)
                _, caches = model(
                    x_hist, context=context,
                    cond_tokens=x_hist[..., :-1] if needs_cond else None,
                    return_cache=True,
                )

            # The step reads the full sequence but `start_pos` drops everything
            # the caches already cover, leaving one frame to compute.
            x_step = torch.randint(0, V, (B, t_hist + 1, layers), device=device)
            cond_step = x_step[..., :-1] if needs_cond else None

            secs[(t_hist, t_text)] = _time(
                lambda: model(x_step, context=context, kv_cache=caches,
                              start_pos=t_hist, cond_tokens=cond_step, return_cache=True),
                warmup=args.warmup, iters=args.iters, device=device,
            )

    _print_grid("T_hist", T_TOKEN_VALUES, "T_text", T_TEXT_VALUES, secs)

    print_separator()
    print_section("Decode throughput (steps/sec)")
    print_info("Fastest", f"{1.0 / min(secs.values()):.1f} steps/s", Colors.OKGREEN)
    print_info("Slowest", f"{1.0 / max(secs.values()):.1f} steps/s", Colors.WARNING)
    print_info(
        "Note",
        "cost grows with T_hist only through attention over the cached keys",
        Colors.WARNING,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Echo inference.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--model", type=str, default="all", choices=["all", "ar", "fm"],
                        help="which model to benchmark (default: all)")
    args = parser.parse_args()

    device = select_device() if args.device == "auto" else torch.device(args.device)

    print_test_title("Echo — Inference Benchmark")
    print_info("Device", device)
    print_info("Warmup iters", args.warmup)
    print_info("Timed iters", args.iters)
    print_info("Batch size", args.batch_size)

    # Pipeline order: EchoAR writes the prosody tokens EchoFM then renders.
    if args.model in ("all", "ar"):
        print()
        benchmark_ar(args, device)
    if args.model in ("all", "fm"):
        print()
        benchmark_fm(args, device)


if __name__ == "__main__":
    main()
