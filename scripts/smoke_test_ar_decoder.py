#!/usr/bin/env python3
"""Smoke test for the AR decoder.

Verifies:
  1. Model constructs and parameter count is under the 60 M budget
     (excluding embeddings).
  2. ``forward(text_tokens, audio_tokens)`` produces correctly shaped
     logits.
  3. ``forward_step`` prefill produces logits that match ``forward``
     (KV cache prefill == full forward pass).
  4. ``forward_step`` single‑token decode produces logits that match the
     corresponding slice of the prefill output (incremental KV cache
     consistency).

Run:
    python scripts/smoke_test_ar_decoder.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from echo.ar_decoder import ARDecoder, KVCache, TokenType
from echo.config import (
    AR_D_FF,
    AR_D_MODEL,
    AR_N_HEADS,
    AR_N_LAYERS,
    CODEC_VOCAB_SIZE,
    EOS_TOKEN_ID,
    OUTPUT_VOCAB_SIZE,
    TEXT_VOCAB_SIZE,
)
from __style__ import (
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
    print_result,
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    print_header("AR Decoder – Smoke Test")
    print_separator()

    device = _select_device()
    print_section("Device")
    print_info("Selected", str(device), Colors.OKCYAN)

    # ------------------------------------------------------------------
    # 1. Build model & count parameters
    # ------------------------------------------------------------------
    print_section("1. Model construction")

    torch.manual_seed(42)
    model = ARDecoder(
        d_model=AR_D_MODEL,
        n_heads=AR_N_HEADS,
        d_ff=AR_D_FF,
        n_layers=AR_N_LAYERS,
        dropout=0.0,  # deterministic for consistency checks
    ).to(device)
    model.eval()

    total_params = model.num_parameters(exclude_embeddings=False)
    non_emb_params = model.num_parameters(exclude_embeddings=True)
    print_info("d_model", str(AR_D_MODEL))
    print_info("n_heads", str(AR_N_HEADS))
    print_info("d_ff", str(AR_D_FF))
    print_info("n_layers", str(AR_N_LAYERS))
    print_info("Total parameters", f"{total_params:,}")
    print_info("Non‑embedding parameters", f"{non_emb_params:,}", Colors.OKCYAN)

    budget = 60_000_000
    if non_emb_params >= budget:
        print_error(f"Non‑embedding params ({non_emb_params:,}) exceed 60 M budget")
        sys.exit(1)
    else:
        print_success(f"Non‑embedding params under 60 M budget "
                      f"({non_emb_params / budget * 100:.1f}%)")

    # ------------------------------------------------------------------
    # 2. Forward pass shape check
    # ------------------------------------------------------------------
    print_section("2. forward() shape check")

    B, T_text, T_audio = 2, 16, 64
    text_tokens = torch.randint(0, TEXT_VOCAB_SIZE, (B, T_text), device=device)
    audio_tokens = torch.randint(0, CODEC_VOCAB_SIZE, (B, T_audio), device=device)

    with torch.no_grad():
        logits = model(text_tokens, audio_tokens)

    expected_T = T_text + 1 + T_audio
    expected_shape = (B, expected_T, OUTPUT_VOCAB_SIZE)
    print_info("Input", f"text={tuple(text_tokens.shape)}, audio={tuple(audio_tokens.shape)}")
    print_info("Output shape", str(tuple(logits.shape)), Colors.OKCYAN)
    print_info("Expected shape", str(expected_shape))

    assert tuple(logits.shape) == expected_shape, (
        f"Shape mismatch: got {tuple(logits.shape)}, expected {expected_shape}"
    )
    print_success("forward() output shape correct")

    # ------------------------------------------------------------------
    # 3. Prefill consistency: forward_step vs forward
    # ------------------------------------------------------------------
    print_section("3. Prefill consistency (forward_step vs forward)")

    # Build token_ids, token_types, positions for the full sequence
    # [text(t_0..t_{Tt-1}), <SEP>, audio(a_0..a_{Ta-1})]
    sep_id = torch.zeros(B, 1, dtype=torch.long, device=device)  # dummy id; SEP uses learned vector
    all_ids = torch.cat([text_tokens, sep_id, audio_tokens], dim=1)

    text_types = torch.full((B, T_text), TokenType.TEXT, dtype=torch.long, device=device)
    sep_type = torch.full((B, 1), TokenType.SEP, dtype=torch.long, device=device)
    audio_types = torch.full((B, T_audio), TokenType.AUDIO, dtype=torch.long, device=device)
    all_types = torch.cat([text_types, sep_type, audio_types], dim=1)

    text_pos = torch.arange(T_text, device=device).unsqueeze(0).expand(B, -1)
    sep_pos = torch.zeros(B, 1, dtype=torch.long, device=device)  # audio pos index 0
    audio_pos = torch.arange(1, T_audio + 1, device=device).unsqueeze(0).expand(B, -1)
    all_pos = torch.cat([text_pos, sep_pos, audio_pos], dim=1)

    with torch.no_grad():
        step_logits, kv_cache = model.forward_step(all_ids, all_types, all_pos, kv_cache=None)

    print_info("forward_step output", str(tuple(step_logits.shape)))
    print_info("Cache length after prefill", str(kv_cache.sequence_length))

    assert kv_cache.sequence_length == expected_T, (
        f"Cache length {kv_cache.sequence_length} != expected {expected_T}"
    )

    max_diff = (step_logits - logits).abs().max().item()
    print_info("Max |forward_step − forward|", f"{max_diff:.2e}", Colors.OKCYAN)
    assert max_diff < 1e-4, f"Prefill logits differ by {max_diff:.2e} (expected < 1e-4)"
    print_success("Prefill logits match forward()")

    # ------------------------------------------------------------------
    # 4. Single‑token decode consistency
    # ------------------------------------------------------------------
    print_section("4. Single‑token decode consistency")

    # Simulate generating one more audio token.
    # The “ground truth” next‑step logits come from re‑running forward with
    # an extra audio token appended.
    extra_audio = torch.randint(0, CODEC_VOCAB_SIZE, (B, 1), device=device)
    with torch.no_grad():
        full_logits_ext = model(text_tokens, torch.cat([audio_tokens, extra_audio], dim=1))
    # The logits for the *new* token (last position) in the extended forward
    # correspond to predicting the token *after* extra_audio.  In the cached
    # forward_step, feeding extra_audio should produce the same logits.
    expected_decode_logits = full_logits_ext[:, -1:, :]  # (B, 1, OUTPUT_VOCAB_SIZE)

    # Feed the extra audio token through forward_step using the existing cache.
    extra_types = torch.full((B, 1), TokenType.AUDIO, dtype=torch.long, device=device)
    extra_pos = torch.full((B, 1), T_audio + 1, dtype=torch.long, device=device)

    with torch.no_grad():
        decode_logits, kv_cache = model.forward_step(
            extra_audio, extra_types, extra_pos, kv_cache=kv_cache
        )

    print_info("Decode logits shape", str(tuple(decode_logits.shape)))
    print_info("Cache length after decode", str(kv_cache.sequence_length))

    assert kv_cache.sequence_length == expected_T + 1, (
        f"Cache length {kv_cache.sequence_length} != expected {expected_T + 1}"
    )

    decode_diff = (decode_logits - expected_decode_logits).abs().max().item()
    print_info("Max |decode − expected|", f"{decode_diff:.2e}", Colors.OKCYAN)
    assert decode_diff < 1e-4, f"Decode logits differ by {decode_diff:.2e} (expected < 1e-4)"
    print_success("Single‑token decode logits match extended forward()")

    # ------------------------------------------------------------------
    # 5. EOS token output check
    # ------------------------------------------------------------------
    print_section("5. EOS token in output vocabulary")

    # The last logit dimension should be the EOS slot.
    print_info("OUTPUT_VOCAB_SIZE", str(OUTPUT_VOCAB_SIZE))
    print_info("EOS_TOKEN_ID (logit index)", str(EOS_TOKEN_ID))
    assert logits.shape[-1] == OUTPUT_VOCAB_SIZE
    assert EOS_TOKEN_ID == CODEC_VOCAB_SIZE  # 2048, right after codec tokens
    print_success("EOS token correctly positioned in output logits")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    tests_passed = 5
    print_result(tests_passed, tests_passed)


if __name__ == "__main__":
    main()
