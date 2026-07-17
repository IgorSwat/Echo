#!/usr/bin/env python3
"""Benchmark inference speed of the AR and NAR decoders.

AR decoder
    For each combination of text length {32, 64, 128, 256} and audio length
    {32, 64, 128}:
      1. **Forward**   – full ``forward(text, audio)`` pass.
      2. **Prefill**   – single ``forward_step`` over the whole prompt.
      3. **Decode**    – token-by-token generation with the KV cache.

NAR decoder
    For each combination of text length {32, 64, 128, 256} and audio length
    {32, 64, 128} with a single previous layer (L=1, predicting layer 1):
      1. **Forward**   – ``forward(text, prev_codec, layer_idx=1)``.
      2. **Prefill**   – ``forward_step`` prefill over the whole prompt.

All measurements use ``batch_size = 1``, ``eval()``, dropout disabled.

Run:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch  # noqa: E402

from echo.ar_decoder import ARDecoder, KVCache, TokenType as ARTokenType  # noqa: E402
from echo.nar_decoder import KVCache as NARKVCache, TokenType as NARTokenType  # noqa: E402
from echo.nar_decoder import NARDecoder  # noqa: E402
from echo.config import (  # noqa: E402
    AR_D_FF, AR_D_MODEL, AR_N_HEADS, AR_N_LAYERS,
    NAR_D_FF, NAR_D_MODEL, NAR_N_HEADS, NAR_N_LAYERS,
    CODEC_VOCAB_SIZE, TEXT_VOCAB_SIZE,
)
from __style__ import (  # noqa: E402
    Colors, print_header, print_section, print_info, print_separator,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TEXT_LENGTHS = [32, 64, 128, 256]
AUDIO_LENGTHS = [32, 64, 128]

BATCH_SIZE: int = 1
N_DECODE_TOKENS: int = 64
N_WARMUP: int = 3
N_ITERS: int = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _benchmark(fn: Callable[[], None], n_warmup: int, n_iters: int,
               device: torch.device) -> float:
    for _ in range(n_warmup):
        fn()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    _sync(device)
    t1 = time.perf_counter()
    return (t1 - t0) / n_iters


# ---------------  AR helpers  -----------------------------------------------


def _ar_build_prompt(
    text_tokens: torch.Tensor,
    audio_tokens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, Tt = text_tokens.shape
    Ta = audio_tokens.shape[1]
    sep_id = torch.zeros(B, 1, dtype=torch.long, device=device)
    ids = torch.cat([text_tokens, sep_id, audio_tokens], dim=1)
    types_ = torch.cat([
        torch.full((B, Tt), ARTokenType.TEXT, dtype=torch.long, device=device),
        torch.full((B, 1), ARTokenType.SEP, dtype=torch.long, device=device),
        torch.full((B, Ta), ARTokenType.AUDIO, dtype=torch.long, device=device),
    ], dim=1)
    pos = torch.cat([
        torch.arange(Tt, device=device).unsqueeze(0).expand(B, -1),
        torch.zeros(B, 1, dtype=torch.long, device=device),
        torch.arange(1, Ta + 1, device=device).unsqueeze(0).expand(B, -1),
    ], dim=1)
    return ids, types_, pos


# ---------------  NAR helpers  -----------------------------------------------


def _nar_build_prompt(
    text_tokens: torch.Tensor,
    codec_tokens: torch.Tensor,    # (B, Ta)
    layer_idx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (ids, types, codec_layer_ids, pos) for NAR prefill."""
    B, Tt = text_tokens.shape
    Ta = codec_tokens.shape[1]

    sep_id = torch.zeros(B, 1, dtype=torch.long, device=device)
    ids = torch.cat([text_tokens, sep_id, codec_tokens], dim=1)

    types_ = torch.cat([
        torch.full((B, Tt), NARTokenType.TEXT, dtype=torch.long, device=device),
        torch.full((B, 1), NARTokenType.SEP, dtype=torch.long, device=device),
        torch.full((B, Ta), NARTokenType.AUDIO, dtype=torch.long, device=device),
    ], dim=1)

    # AUDIO tokens come from layer (layer_idx - 1).
    codec_layers = torch.full((B, Ta), layer_idx - 1, dtype=torch.long, device=device)
    codec_layers_full = torch.cat([
        torch.full((B, Tt + 1), -1, dtype=torch.long, device=device),  # not used
        codec_layers,
    ], dim=1)

    pos = torch.cat([
        torch.arange(Tt, device=device).unsqueeze(0).expand(B, -1),
        torch.zeros(B, 1, dtype=torch.long, device=device),
        torch.arange(1, Ta + 1, device=device).unsqueeze(0).expand(B, -1),
    ], dim=1)

    return ids, types_, codec_layers_full, pos


# ---------------------------------------------------------------------------
# AR benchmark
# ---------------------------------------------------------------------------


def ar_benchmark_forward(model: ARDecoder, tt: torch.Tensor, at: torch.Tensor,
                         device: torch.device) -> float:
    def fn() -> None: model(tt, at)
    return _benchmark(fn, N_WARMUP, N_ITERS, device)


def ar_benchmark_prefill(model: ARDecoder, ids: torch.Tensor, types_: torch.Tensor,
                         pos: torch.Tensor, device: torch.device) -> float:
    def fn() -> None: model.forward_step(ids, types_, pos, kv_cache=None)
    return _benchmark(fn, N_WARMUP, N_ITERS, device)


def ar_benchmark_decode(model: ARDecoder, ids: torch.Tensor, types_: torch.Tensor,
                        pos: torch.Tensor, device: torch.device,
                        ) -> tuple[float, float]:
    B = ids.shape[0]; plen = ids.shape[1]; nl = model.n_layers
    def run() -> float:
        kc = KVCache(nl)
        model.forward_step(ids, types_, pos, kv_cache=kc)
        dt = torch.randint(0, CODEC_VOCAB_SIZE, (B, N_DECODE_TOKENS), device=device)
        dty = torch.full((B, N_DECODE_TOKENS), ARTokenType.AUDIO, dtype=torch.long, device=device)
        dp = torch.arange(plen, plen + N_DECODE_TOKENS, device=device).unsqueeze(0).expand(B, -1)
        _sync(device); t0 = time.perf_counter()
        for i in range(N_DECODE_TOKENS):
            model.forward_step(dt[:, i:i+1], dty[:, i:i+1], dp[:, i:i+1], kv_cache=kc)
        _sync(device); t1 = time.perf_counter()
        return t1 - t0
    for _ in range(N_WARMUP): run()
    times = [run() for _ in range(N_ITERS)]
    m = sum(times) / len(times)
    return m / N_DECODE_TOKENS, N_DECODE_TOKENS / m


# ---------------------------------------------------------------------------
# NAR benchmark
# ---------------------------------------------------------------------------


def nar_benchmark_forward(model: NARDecoder, tt: torch.Tensor,
                          ct: torch.Tensor, layer_idx: int,
                          device: torch.device) -> float:
    # ct: (B, 1, Ta) — single previous layer stacked
    def fn() -> None: model(tt, ct, layer_idx)
    return _benchmark(fn, N_WARMUP, N_ITERS, device)


def nar_benchmark_prefill(model: NARDecoder, ids: torch.Tensor, types_: torch.Tensor,
                          cls: torch.Tensor, pos: torch.Tensor, layer_idx: int,
                          device: torch.device) -> float:
    def fn() -> None: model.forward_step(ids, types_, cls, pos, kv_cache=None,
                                          target_layer=layer_idx)
    return _benchmark(fn, N_WARMUP, N_ITERS, device)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print_header("AR + NAR Decoder – Inference Benchmark")
    print_separator()

    device = _select_device()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Batch size", str(BATCH_SIZE))
    print_info("Warm-up / timed iters", f"{N_WARMUP} / {N_ITERS}")

    # ==========================  AR  ========================================
    print_section("AR Decoder")
    ar = ARDecoder(d_model=AR_D_MODEL, n_heads=AR_N_HEADS, d_ff=AR_D_FF,
                   n_layers=AR_N_LAYERS, dropout=0.0).to(device).eval()
    print_info("Total params", f"{ar.num_parameters():,}")
    print_info("Non-emb params", f"{ar.num_parameters(exclude_embeddings=True):,}",
               Colors.OKCYAN)

    h = (f"{'Text':>5}  {'Audio':>5}  {'Total':>5}  │"
         f"{'Forward':>10}  {'Prefill':>10}  {'Decode/tok':>12}  {'Decode':>10}")
    u = (f"{'(len)':>5}  {'(len)':>5}  {'(len)':>5}  │"
         f"{'(ms)':>10}  {'(ms)':>10}  {'(ms)':>12}  {'(tok/s)':>10}")
    print(); print(f"{Colors.BOLD}{h}{Colors.ENDC}"); print(f"{Colors.BOLD}{u}{Colors.ENDC}")
    print_separator("┼", len(h))

    for t_len in TEXT_LENGTHS:
        for a_len in AUDIO_LENGTHS:
            tt = torch.randint(0, TEXT_VOCAB_SIZE, (BATCH_SIZE, t_len), device=device)
            at = torch.randint(0, CODEC_VOCAB_SIZE, (BATCH_SIZE, a_len), device=device)
            ids, types_, pos = _ar_build_prompt(tt, at, device)
            with torch.no_grad():
                fwd = ar_benchmark_forward(ar, tt, at, device)
                pre = ar_benchmark_prefill(ar, ids, types_, pos, device)
                dpt, dtps = ar_benchmark_decode(ar, ids, types_, pos, device)
            tl = t_len + 1 + a_len
            print(f"{t_len:>5}  {a_len:>5}  {tl:>5}  │"
                  f"{fwd*1e3:>10.2f}  {pre*1e3:>10.2f}  "
                  f"{dpt*1e3:>12.2f}  {dtps:>10.1f}")
        print_separator("┬", len(h))
    print_separator("═", len(h))
    print()

    # ==========================  NAR  ========================================
    print_section("NAR Decoder")
    nar = NARDecoder(d_model=NAR_D_MODEL, n_heads=NAR_N_HEADS, d_ff=NAR_D_FF,
                     n_layers=NAR_N_LAYERS, dropout=0.0).to(device).eval()
    print_info("Total params", f"{nar.num_parameters():,}")
    print_info("Non-emb params", f"{nar.num_parameters(exclude_embeddings=True):,}",
               Colors.OKCYAN)

    nh = (f"{'Text':>5}  {'Audio':>5}  {'Total':>5}  │"
          f"{'Forward':>10}  {'Prefill':>10}")
    nu = (f"{'(len)':>5}  {'(len)':>5}  {'(len)':>5}  │"
          f"{'(ms)':>10}  {'(ms)':>10}")
    print(); print(f"{Colors.BOLD}{nh}{Colors.ENDC}"); print(f"{Colors.BOLD}{nu}{Colors.ENDC}")
    print_separator("┼", len(nh))

    LYR = 1  # single previous layer → predict layer 1
    for t_len in TEXT_LENGTHS:
        for a_len in AUDIO_LENGTHS:
            tt = torch.randint(0, TEXT_VOCAB_SIZE, (BATCH_SIZE, t_len), device=device)
            ct = torch.randint(0, CODEC_VOCAB_SIZE, (BATCH_SIZE, 1, a_len), device=device)
            ids, types_, cls, pos = _nar_build_prompt(
                tt, ct[:, 0, :] if ct.ndim == 3 else ct, LYR, device)
            ct_stacked = ct if ct.ndim == 3 else ct.unsqueeze(1)
            with torch.no_grad():
                fwd = nar_benchmark_forward(nar, tt, ct_stacked, LYR, device)
                pre = nar_benchmark_prefill(nar, ids, types_, cls, pos, LYR, device)
            tl = t_len + 1 + a_len
            print(f"{t_len:>5}  {a_len:>5}  {tl:>5}  │"
                  f"{fwd*1e3:>10.2f}  {pre*1e3:>10.2f}")
        print_separator("┬", len(nh))
    print_separator("═", len(nh))
    print()


if __name__ == "__main__":
    main()