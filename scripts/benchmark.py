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

    # Fixed reference prompt used across all measurements. The reference provides
    # voice/style conditioning; the target text + target audio are varied below.
    ref_text_len = 32
    ref_audio_len = 64
    text_lens = [32, 64, 128, 256]
    audio_lens = [32, 64, 128]

    ref_text = torch.randint(
        0, config.TEXT_VOCAB_SIZE, (BATCH, ref_text_len), device=DEVICE
    ).long()
    ref_audio = torch.randint(
        0, config.CODEC_PAD_ID, (BATCH, ref_audio_len, config.NUM_CODEBOOKS), device=DEVICE
    ).long()

    print_info("Ref text len", str(ref_text_len))
    print_info("Ref audio len", str(ref_audio_len))
    print()

    # ---- Forward (teacher-forcing) ----
    print_section("Forward pass (teacher-forcing)")
    header = f"  {'text len':>8}  {'audio len':>9}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        for alen in audio_lens:
            text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()
            audio = torch.randint(0, config.CODEC_PAD_ID, (BATCH, alen, config.NUM_CODEBOOKS), device=DEVICE).long()
            text_lengths = torch.full((BATCH,), tlen, dtype=torch.long, device=DEVICE)
            audio_lengths = torch.full((BATCH,), alen, dtype=torch.long, device=DEVICE)

            def _fwd(text=text, audio=audio):
                return model(ref_text, ref_audio, text, audio, text_lengths, audio_lengths)

            avg = _time_it(_fwd)
            # <BOS> + ref_text + <REF_TEXT_EOS> + ref_audio + <REF_CODEC_EOS>
            # + text + <TEXT_EOS> + audio
            total_tokens = 1 + ref_text_len + 1 + ref_audio_len + 1 + tlen + 1 + alen
            tok_per_s = total_tokens / avg
            print(f"  {tlen:>8}  {alen:>9}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Prefill ----
    print_section("Prefill (warm-start decoding)")
    header = f"  {'text len':>8}  {'time':>10}  {'tokens/s':>10}"
    print(header)
    print_separator(width=len(header))

    for tlen in text_lens:
        text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()

        def _prefill(text=text):
            return model.prefill(ref_text, ref_audio, text)

        avg = _time_it(_prefill)
        # Prefill consumes the full prompt + text but no target audio.
        total_tokens = 1 + ref_text_len + 1 + ref_audio_len + 1 + tlen + 1
        tok_per_s = total_tokens / avg
        print(f"  {tlen:>8}  {_fmt_time(avg):>10}  {tok_per_s:>10.0f}")

    print()

    # ---- Step (iterative decoding) ----
    print_section("Step (single-frame iterative decode)")
    header = f"  {'text len':>8}  {'time/step':>12}  {'real-time factor':>17}"
    print(header)
    print_separator(width=len(header))

    # Time per step is roughly constant but let's measure at different cache sizes
    # (driven by the target text length; the reference prompt is held fixed).
    for tlen in [32, 128, 256]:
        text = torch.randint(0, config.TEXT_VOCAB_SIZE, (BATCH, tlen), device=DEVICE).long()
        _, kv_cache = model.prefill(ref_text, ref_audio, text)
        start_pos = 1 + ref_text_len + 1 + ref_audio_len + 1 + tlen + 1

        frame = torch.randint(0, config.CODEC_PAD_ID, (BATCH, config.NUM_CODEBOOKS), device=DEVICE).long()

        def _step():
            return model.step(frame, position=start_pos, kv_cache=kv_cache)

        avg = _time_it(_step, warmup=3, repeats=REPEATS)
        # Real-time factor: 1 frame = 12.5ms at 80Hz codec => how much faster is generation vs real-time?
        frame_duration = 0.0125  # seconds at 80 Hz
        rtf = avg / frame_duration
        print(f"  {tlen:>8}  {_fmt_time(avg):>12}  {rtf:>17.3f}")

    print()
    print_info("Note", "Real-time factor < 1 means faster than real-time", Colors.OKCYAN)


if __name__ == "__main__":
    main()
