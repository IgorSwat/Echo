#!/usr/bin/env python3
"""Benchmark Echo model inference speed across different sequence lengths."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo import config
from echo.model import Echo
from scripts.__style__ import Colors, print_header, print_section, print_separator, print_info


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 5
REPEATS = 20
BATCH = 1


def _elapsed(start_event, end_event) -> float:
    if DEVICE == "cuda":
        start_event.synchronize()
        end_event.synchronize()
        return start_event.elapsed_time(end_event) / 1000.0
    return end_event - start_event


def _timer():
    if DEVICE == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    return None, None


def _record(event) -> None:
    if DEVICE == "cuda":
        event.record()
    else:
        pass


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_it(fn, *args, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    for _ in range(warmup):
        fn(*args)
        _sync()
    start_event, end_event = _timer()
    times: list[float] = []
    for _ in range(repeats):
        if DEVICE == "cuda":
            start_event.record()
        else:
            t0 = time.perf_counter()
        fn(*args)
        if DEVICE == "cuda":
            end_event.record()
            _sync()
            times.append(start_event.elapsed_time(end_event) / 1000.0)
        else:
            t1 = time.perf_counter()
            _sync()
            times.append(t1 - t0)
    return sum(times) / len(times)


def main() -> None:
    print_header(f"Echo model — inference benchmark ({DEVICE.upper()})")
    print_info("Batch size", BATCH)
    print_info("Warmup runs", WARMUP)
    print_info("Measured runs", REPEATS)
    print()

    model = Echo().to(DEVICE).eval()
    torch.set_float32_matmul_precision("high")

    text_lens = [32, 64, 128, 256]
    audio_lens = [32, 64, 128]

    # ---- Forward (teacher-forcing) ----
    print_section("Forward pass (teacher-forcing)")
    header = f"  {'text len':>8}  {'audio len':>9}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        for alen in audio_lens:
            text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()
            audio = torch.randint(0, config.CODEC_VOCAB_SIZE, (BATCH, alen, config.NUM_CODEBOOKS), device=DEVICE).long()

            def _fwd():
                return model(text, audio)

            avg = _time_it(_fwd)
            total_tokens = tlen + 1 + alen  # text + sep + audio
            tok_per_s = total_tokens / avg
            print(f"  {tlen:>8}  {alen:>9}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Prefill ----
    print_section("Prefill (warm-start decoding)")
    header = f"  {'text len':>8}  {'audio len':>9}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        for alen in audio_lens:
            text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()
            audio = torch.randint(0, config.CODEC_VOCAB_SIZE, (BATCH, alen, config.NUM_CODEBOOKS), device=DEVICE).long()

            def _prefill():
                return model.prefill(text, audio)

            avg = _time_it(_prefill)
            total_tokens = tlen + 1 + alen
            tok_per_s = total_tokens / avg
            print(f"  {tlen:>8}  {alen:>9}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Step (iterative decoding) ----
    print_section("Step (single-frame iterative decode)")
    header = f"  {'seq len':>8}  {'time/step':>12}  {'real-time factor':>17}"
    print(header)
    print_separator(width=len(header))

    # Time per step is roughly constant but let's measure at different cache sizes
    for tlen in [32, 128, 512]:
        text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()
        audio = torch.randint(0, config.CODEC_VOCAB_SIZE, (BATCH, 64, config.NUM_CODEBOOKS), device=DEVICE).long()
        pre = model.prefill(text, audio)
        kv_cache = pre.kv_cache
        start_pos = model._prefill_audio_len

        frame = torch.randint(0, config.CODEC_VOCAB_SIZE, (BATCH, config.NUM_CODEBOOKS), device=DEVICE).long()

        def _step():
            return model.step(frame, position=start_pos + 1, kv_cache=kv_cache)

        avg = _time_it(_step, warmup=3, repeats=REPEATS)
        # Real-time factor: 1 frame = 12.5ms at 80Hz codec => how much faster is generation vs real-time?
        frame_duration = 0.0125  # seconds at 80 Hz
        rtf = avg / frame_duration
        print(f"  {tlen:>8}  {_fmt_time(avg):>12}  {rtf:>17.3f}")

    print()
    print_info("Note", "Real-time factor < 1 means faster than real-time", Colors.OKCYAN)


if __name__ == "__main__":
    main()