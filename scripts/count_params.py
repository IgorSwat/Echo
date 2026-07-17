"""Count and report the parameter budget of the Echo model pipeline.

Runs every module that has been implemented so far, prints a per-module and
per-submodule breakdown, and reports the pipeline total. As new modules are
added (text encoder, decoder, ...), register them in ``PIPELINE`` below and
they will be picked up automatically.

Usage:
    python scripts/count_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the ``echo`` package importable when running this script directly,
# regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch.nn as nn  # noqa: E402

from echo.config import AudioEncoderConfig  # noqa: E402
from echo.config import TextEncoderConfig  # noqa: E402
from echo.config import DecoderPlannerConfig  # noqa: E402
from echo.config import DecoderExecutorConfig  # noqa: E402
from echo.codec_embedding import CodecEmbedding  # noqa: E402
from echo.audio_encoder import AudioEncoder  # noqa: E402
from echo.text_encoder import TextEncoder  # noqa: E402
from echo.decoder import DecoderPlanner  # noqa: E402
from echo.decoder import DecoderExecutor  # noqa: E402


# =================
# Pipeline registry
# =================

def _build_pipeline() -> list[tuple[str, nn.Module]]:
    audio_cfg = AudioEncoderConfig()
    exec_cfg = DecoderExecutorConfig()

    # One codec embedding table shared by the audio encoder and the decoder
    # executor (includes the MASK slot).
    codec_embedding = CodecEmbedding(
        vocab_size=audio_cfg.vocab_size,
        num_codebooks=audio_cfg.num_codebooks,
        embedding_dim=audio_cfg.embedding_dim,
        mask_token_id=audio_cfg.mask_token_id,
    )

    return [
        ("AudioEncoder", AudioEncoder(audio_cfg, codec_embedding=codec_embedding)),
        ("TextEncoder", TextEncoder(TextEncoderConfig())),
        ("DecoderPlanner", DecoderPlanner(DecoderPlannerConfig())),
        ("DecoderExecutor", DecoderExecutor(exec_cfg, codec_embedding=codec_embedding)),
    ]


# =================
# Reporting helpers
# =================

def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _fmt(n: int) -> str:
    return f"{n / 1e6:8.3f} M"


def _print_breakdown(name: str, module: nn.Module, depth: int, max_depth: int) -> int:
    """Print ``module`` and its direct children up to ``max_depth``.

    ``nn.ModuleList`` children are collapsed into a single summary line
    (count x per-block params) instead of being expanded element by element.
    Zero-parameter modules (e.g. Dropout) are omitted for brevity.
    """
    pad = "  " * depth
    total = _count(module)
    print(f"{pad}{name:<28} {_fmt(total)}")

    if depth >= max_depth:
        return total

    for child_name, child in module.named_children():
        if _count(child) == 0:
            continue
        # Collapse ModuleList (e.g. transformer blocks) into one summary line.
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            _print_collapsed(child_name, child, depth + 1, max_depth)
        else:
            _print_breakdown(child_name, child, depth + 1, max_depth)
    return total


def _print_collapsed(name: str, module_list: nn.ModuleList, depth: int, max_depth: int) -> None:
    """Print a ModuleList. When all blocks have equal param counts, collapse to
    a single ``N x per_block`` summary line. When blocks differ (e.g. the
    forehead's layers have growing in/out dims), print each block's total on
    its own line so the breakdown is honest.

    In both cases the first block is additionally expanded one level deeper so
    the user can see what makes up each repeated block.
    """
    pad = "  " * depth
    n = len(module_list)
    total = _count(module_list)
    per_block_counts = [_count(b) for b in module_list]
    all_equal = len(set(per_block_counts)) == 1

    if all_equal:
        print(f"{pad}{name:<28} {_fmt(total)}   ({n} x {_fmt(per_block_counts[0])} each)")
    else:
        print(f"{pad}{name:<28} {_fmt(total)}   ({n} blocks, varying sizes:)")
        for i, c in enumerate(per_block_counts):
            print(f"{pad}  {'block ' + str(i):<26} {_fmt(c)}")

    # Expand the first block one level deeper to show its composition.
    first = module_list[0]
    if depth + 1 < max_depth:
        for child_name, child in first.named_children():
            if _count(child) == 0:
                continue
            if isinstance(child, nn.ModuleList) and len(child) > 0:
                _print_collapsed(child_name, child, depth + 2, max_depth)
            else:
                _print_breakdown(child_name, child, depth + 2, max_depth)


# ===========
# Entry point
# ===========

def main() -> None:
    pipeline = _build_pipeline()

    print("=" * 60)
    print("Echo pipeline - parameter count")
    print("=" * 60)

    for name, model in pipeline:
        print()
        _print_breakdown(name, model, depth=0, max_depth=6)

    grand_total = 0
    # Track parameter tensors by storage identity so modules that share
    # parameters (e.g. the shared CodecEmbedding between AudioEncoder and
    # DecoderExecutor) are counted exactly once in the pipeline total. Note
    # that the per-module breakdown lines above still include the shared
    # params under each owning module.
    seen: set[int] = set()
    for _, model in pipeline:
        for p in model.parameters():
            ptr = p.data_ptr()
            if ptr in seen:
                continue
            seen.add(ptr)
            grand_total += p.numel()

    print()
    print("-" * 60)
    print(f"{'PIPELINE TOTAL (unique)':<28} {_fmt(grand_total)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
