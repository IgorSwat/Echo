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


FIXED_REF_TEXT_LEN = 32
FIXED_REF_AUDIO_LEN = 96

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
WARMUP = 5
REPEATS = 20
BATCH = 1


def _fmt_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def _sync() -> None:
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def _time_it(fn, *args, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    for _ in range(warmup):
        fn(*args)
        _sync()
    times: list[float] = []
    for _ in range(repeats):
        if DEVICE == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(*args)
            end.record()
            _sync()
            times.append(start.elapsed_time(end) / 1000.0)
        else:
            t0 = time.perf_counter()
            fn(*args)
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
    print_info("Ref text len (fixed)", FIXED_REF_TEXT_LEN, Colors.OKGREEN)
    print_info("Ref audio len (fixed)", FIXED_REF_AUDIO_LEN, Colors.OKGREEN)
    print()

    model = Echo().to(DEVICE).eval()
    torch.set_float32_matmul_precision("high")

    ref_text = torch.randint(
        0, config.text_vocab_size, (BATCH, FIXED_REF_TEXT_LEN), device=DEVICE
    ).long()
    ref_audio = torch.randint(
        0, config.audio_pad_id, (BATCH, FIXED_REF_AUDIO_LEN, config.num_codebooks), device=DEVICE
    ).long()

    text_lens = [32, 64, 128, 256]
    audio_lens = [32, 64, 128]

    # ---- Forward (teacher-forcing) ----
    print_section("Forward pass (teacher-forcing)")
    header = f"  {'text len':>8}  {'audio len':>9}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        for alen in audio_lens:
            text = torch.randint(0, config.text_vocab_size, (BATCH, tlen), device=DEVICE).long()
            audio = torch.randint(0, config.audio_pad_id, (BATCH, alen, config.num_codebooks), device=DEVICE).long()
            text_lengths = torch.full((BATCH,), tlen, dtype=torch.long, device=DEVICE)
            audio_lengths = torch.full((BATCH,), alen, dtype=torch.long, device=DEVICE)

            def _fwd(text=text, audio=audio):
                return model(ref_text, ref_audio, text, audio, text_lengths, audio_lengths)

            avg = _time_it(_fwd)
            total_tokens = 1 + FIXED_REF_TEXT_LEN + 1 + FIXED_REF_AUDIO_LEN + 1 + tlen + 1 + alen
            tok_per_s = total_tokens / avg
            print(f"  {tlen:>8}  {alen:>9}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Prefill ----
    print_section("Prefill (warm-start decoding)")
    header = f"  {'text len':>8}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        text = torch.randint(0, config.text_vocab_size, (BATCH, tlen), device=DEVICE).long()

        def _prefill(text=text):
            return model.prefill(ref_text, ref_audio, text)

        avg = _time_it(_prefill)
        total_tokens = 1 + FIXED_REF_TEXT_LEN + 1 + FIXED_REF_AUDIO_LEN + 1 + tlen + 1
        tok_per_s = total_tokens / avg
        print(f"  {tlen:>8}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Step (iterative decoding) ----
    print_section("Step (single-frame iterative decode)")
    header = f"  {'text len':>8}  {'time/step':>12}  {'real-time factor':>17}"
    print(header)
    print_separator(width=len(header))

    for tlen in [32, 128, 256]:
        text = torch.randint(0, config.text_vocab_size, (BATCH, tlen), device=DEVICE).long()
        _, kv_cache = model.prefill(ref_text, ref_audio, text)
        start_pos = 1 + FIXED_REF_TEXT_LEN + 1 + FIXED_REF_AUDIO_LEN + 1 + tlen + 1

        frame = torch.randint(0, config.audio_pad_id, (BATCH, config.num_codebooks), device=DEVICE).long()

        def _step():
            return model.step(frame, position=start_pos, kv_cache=kv_cache)

        avg = _time_it(_step, warmup=3, repeats=REPEATS)
        frame_duration = 0.0125  # seconds at 80 Hz
        rtf = avg / frame_duration
        print(f"  {tlen:>8}  {_fmt_time(avg):>12}  {rtf:>17.3f}")

    print()
    print_info("Note", "Real-time factor < 1 means faster than real-time", Colors.OKCYAN)


if __name__ == "__main__":
    main()
