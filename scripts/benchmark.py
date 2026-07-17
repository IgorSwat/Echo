"""Benchmark inference speed of Echo modules on Apple Silicon (MPS).

Runs warmup + timed forward passes and reports per-iteration latency
(mean / median / p95) and throughput (tokens/sec).

Usage:
    python scripts/benchmark.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import mean, median

import torch

# Make the ``echo`` package importable when running this script directly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo.config import AudioEncoderConfig  # noqa: E402
from echo.config import TextEncoderConfig  # noqa: E402
from echo.config import DecoderPlannerConfig  # noqa: E402
from echo.config import DecoderExecutorConfig  # noqa: E402
from echo.codec_embedding import CodecEmbedding  # noqa: E402
from echo.audio_encoder import AudioEncoder  # noqa: E402
from echo.text_encoder import TextEncoder  # noqa: E402
from echo.decoder import DecoderPlanner  # noqa: E402
from echo.decoder import DecoderExecutor  # noqa: E402


# ============================================================================
# Benchmark config
# ============================================================================
WARMUP_ITERS: int = 5
TIMED_ITERS: int = 20

# Sequence lengths to benchmark.
SEQ_LENS: list[int] = [128, 512, 1024]
BATCH_SIZE: int = 1


# ============================================================================
# Timing utilities
# ============================================================================
def _benchmark(
    model: torch.nn.Module,
    args: tuple,
    device: torch.device,
    warmup: int,
    iters: int,
) -> list[float]:
    """Run ``warmup`` untimed + ``iters`` timed forward passes; return latencies (s)."""
    model = model.to(device).eval()

    with torch.no_grad():
        # Warmup.
        for _ in range(warmup):
            _ = model(*[a.to(device) for a in args])
        torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize() if device.type == "cuda" else None

        # Timed.
        latencies = []
        for _ in range(iters):
            args_dev = [a.to(device) for a in args]
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = model(*args_dev)

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    return latencies


def _percentile(data: list[float], pct: float) -> float:
    """Simple percentile (``pct`` in 0-100)."""
    s = sorted(data)
    k = int(round((pct / 100.0) * (len(s) - 1)))
    return s[k]


# ============================================================================
# Entry point
# ============================================================================
def _run_benchmark(
    label: str,
    model: torch.nn.Module,
    make_args,
    device: torch.device,
) -> None:
    """Build args per sequence length, run timing, print a results table."""
    header = f"{'seq_len':>8}  {'mean (ms)':>10}  {'median (ms)':>12}  {'p95 (ms)':>10}  {'tok/s':>10}"
    print("=" * len(header))
    print(f"{label} benchmark  |  device={device}  batch={BATCH_SIZE}")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for T in SEQ_LENS:
        args = make_args(T)

        latencies = _benchmark(
            model, args, device,
            warmup=WARMUP_ITERS, iters=TIMED_ITERS,
        )

        mean_ms = mean(latencies) * 1e3
        med_ms = median(latencies) * 1e3
        p95_ms = _percentile(latencies, 95) * 1e3
        tok_per_s = (BATCH_SIZE * T) / mean(latencies)

        print(f"{T:>8}  {mean_ms:>10.2f}  {med_ms:>12.2f}  {p95_ms:>10.2f}  {tok_per_s:>10.0f}")

    print()


# ============================================================================
# Entry point
# ============================================================================
def main() -> None:
    assert torch.backends.mps.is_available(), "MPS is not available on this device."
    device = torch.device("mps")

    # --- AudioEncoder ---------------------------------------------------------
    audio_cfg = AudioEncoderConfig()
    exec_cfg = DecoderExecutorConfig()
    codec_embedding = CodecEmbedding(
        vocab_size=audio_cfg.vocab_size,
        num_codebooks=audio_cfg.num_codebooks,
        embedding_dim=audio_cfg.embedding_dim,
        mask_token_id=audio_cfg.mask_token_id,
    )
    audio_model = AudioEncoder(audio_cfg, codec_embedding=codec_embedding)

    def audio_args(T: int) -> tuple:
        return (torch.randint(0, audio_cfg.vocab_size, (BATCH_SIZE, audio_cfg.num_codebooks, T), dtype=torch.long),)

    _run_benchmark("AudioEncoder", audio_model, audio_args, device)

    # --- TextEncoder ---------------------------------------------------------
    text_cfg = TextEncoderConfig()
    text_model = TextEncoder(text_cfg)

    def text_args(T: int) -> tuple:
        return (torch.randint(0, text_cfg.vocab_size, (BATCH_SIZE, T), dtype=torch.long),)

    _run_benchmark("TextEncoder", text_model, text_args, device)

    # --- DecoderPlanner ------------------------------------------------------
    dec_cfg = DecoderPlannerConfig()
    dec_model = DecoderPlanner(dec_cfg)

    def dec_args(T: int) -> tuple:
        H = torch.randn(BATCH_SIZE, T, dec_cfg.d_plan)
        text_enc = torch.randn(BATCH_SIZE, T // 2, dec_cfg.d_hid)
        audio_enc = torch.randn(BATCH_SIZE, T, dec_cfg.d_hid)
        return (H, text_enc, audio_enc)

    _run_benchmark("DecoderPlanner", dec_model, dec_args, device)

    # --- DecoderExecutor -----------------------------------------------------
    exec_model = DecoderExecutor(exec_cfg, codec_embedding=codec_embedding)

    def exec_args(T: int) -> tuple:
        planner_state = torch.randn(BATCH_SIZE, exec_cfg.d_plan)
        # full grid: context_steps real rows + chunk_size rows with a mix of
        # real tokens and MASK tokens.
        CTX, CS, C = exec_cfg.context_steps, exec_cfg.chunk_size, exec_cfg.num_codebooks
        grid = torch.randint(0, exec_cfg.vocab_size, (BATCH_SIZE, CTX + CS, C), dtype=torch.long)
        # mask ~half of the chunk rows for a realistic MLM-style input.
        mask = torch.rand(BATCH_SIZE, CS, C) < 0.5
        grid[:, CTX:, :][mask] = exec_cfg.mask_token_id
        return (planner_state, grid)

    _run_benchmark("DecoderExecutor", exec_model, exec_args, device)


if __name__ == "__main__":
    main()
