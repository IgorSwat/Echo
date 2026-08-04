#!/usr/bin/env python3
"""Detailed parameter breakdown of the Echo models.

For EchoFM, reports parameter counts for the TimeEncoder, TextEncoder, and each
block in the main processing stack, grouped by block type. For EchoAR, reports
the token embeddings, TextEncoder, decoder blocks and the per-token-layer heads.

Usage:
    python scripts/count_params.py
    python scripts/count_params.py --model ar
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
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
from echo.ar_model import EchoAR
from echo.fm_model import EchoFM
from echo.shortcut_model import EchoShortcut


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _fmt(n: int) -> str:
    return f"{n:,}"


def _print_components(components: list[tuple[str, int]], total: int) -> None:
    """Component table with a share-of-total bar."""
    for label, n in components:
        pct = 100.0 * n / total if total else 0.0
        bar_len = int(round(pct / 100.0 * 30))
        bar = Colors.OKGREEN + "█" * bar_len + Colors.ENDC + "░" * (30 - bar_len)
        print(f"  {Colors.BOLD}{label:<22}{Colors.ENDC} {_fmt(n):>12}  {pct:5.1f}%  {bar}")


def _print_blocks(blocks: torch.nn.Module, total: int, label: str) -> None:
    """Per-block listing followed by an aggregation over block classes."""
    print_section(f"{label} — per block")

    rows = []
    type_totals: OrderedDict[str, list[int]] = OrderedDict()

    for i, block in enumerate(blocks):
        n = _count_params(block)
        cls_name = type(block).__name__
        rows.append((i, cls_name, n))
        type_totals.setdefault(cls_name, []).append(n)

    # Individual blocks
    for i, cls_name, n in rows:
        pct = 100.0 * n / total if total else 0.0
        print(f"  [{i}] {Colors.BOLD}{cls_name:<20}{Colors.ENDC} {_fmt(n):>12}  {pct:5.1f}%")

    # Aggregated per type
    print_section(f"{label} — aggregated by type")

    for cls_name, counts in type_totals.items():
        subtotal = sum(counts)
        pct = 100.0 * subtotal / total if total else 0.0
        avg = subtotal / len(counts)
        print_info(
            f"{cls_name} (x{len(counts)})",
            f"{_fmt(subtotal)}  (avg {_fmt(int(round(avg)))})  {pct:5.1f}%",
        )


def report_fm() -> None:
    model = EchoFM()
    model.eval()

    total = _count_params(model)

    print_header("EchoFM — flow-matching backbone")

    # --- Top-level components ---
    print_section("Top-level components")

    _print_components([
        ("TimeEncoder", _count_params(model.time_encoder)),
        ("TextEncoder", _count_params(model.text_encoder)),
        ("Main blocks", _count_params(model.blocks)),
        ("Final norm + out_proj",
         _count_params(model.final_norm) + _count_params(model.out_proj)),
    ], total)

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    # --- Per-block breakdown ---
    _print_blocks(model.blocks, total, "Main blocks")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


def report_ar() -> None:
    model = EchoAR()
    model.eval()

    total = _count_params(model)

    print_header("EchoAR — autoregressive prosody model")

    # --- Top-level components ---
    print_section("Top-level components")

    components = [
        (f"Token embeddings (x{len(model.embed)})", _count_params(model.embed)),
        ("TextEncoder", _count_params(model.text_encoder)),
        ("Decoder", _count_params(model.decoder)),
        (f"Heads (x{len(model.heads)})", _count_params(model.heads)),
    ]
    if model.in_proj is not None:
        components.insert(3, ("Input projection", _count_params(model.in_proj)))

    _print_components(components, total)

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    # --- Dimensions ---
    print_section("Dimensions")
    print_info("Token layers", model.NUM_TOKEN_LAYERS)
    print_info("Prosody vocab", _fmt(model.vocab_size))
    print_info("Embedding dim", f"{model.emb_dim} (x{model.NUM_TOKEN_LAYERS} -> "
                                f"{model.NUM_TOKEN_LAYERS * model.emb_dim})")
    print_info("Hidden dim", model.hidden_dim)
    print_info("Text context dim", config.ar_model.text_encoder_d_model)

    # --- Per-block breakdown ---
    _print_blocks(model.decoder.blocks, total, "Decoder blocks")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


def report_shortcut() -> None:
    model = EchoShortcut()
    model.eval()

    total = _count_params(model)

    print_header("EchoShortcut — Mimi tokens -> Blue latent")

    # --- Top-level components ---
    print_section("Top-level components")

    components = [
        (f"Token embeddings (x{len(model.embed)})", _count_params(model.embed)),
    ]
    for i, (blocks, (k, factor)) in enumerate(zip(model.stages, model.stage_specs)):
        suffix = f", up x{factor:g}" if factor != 1.0 else ""
        components.append((f"Stage {i} (k={k}{suffix})", _count_params(blocks)))
    components.append(("Output projection", _count_params(model.out_proj)))

    _print_components(components, total)

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    # --- Dimensions ---
    print_section("Dimensions")
    print_info("Token layers", model.NUM_TOKEN_LAYERS)
    print_info("Prosody vocab", _fmt(model.vocab_size))
    print_info("Embedding dim", f"{model.emb_dim} (x{model.NUM_TOKEN_LAYERS} -> "
                                f"{model.hidden_dim})")
    print_info("Output dim", model.d_out)
    print_info("Blocks per stage", config.shortcut_model.blocks_per_stage)

    upsample = 1.0
    for _, factor in model.stage_specs:
        upsample *= factor
    print_info("Total upsample", f"x{upsample:g}")

    # --- Per-block breakdown ---
    flat = torch.nn.ModuleList([b for blocks in model.stages for b in blocks])
    _print_blocks(flat, total, "ConvNeXt blocks")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo parameter breakdown.")
    parser.add_argument("--model", type=str, default="all",
                        choices=["all", "fm", "ar", "shortcut"],
                        help="which model to report on (default: all)")
    args = parser.parse_args()

    print_test_title("Echo — Parameter Breakdown")

    if args.model in ("all", "fm"):
        report_fm()
    if args.model in ("all", "ar"):
        if args.model == "all":
            print()
        report_ar()
    if args.model in ("all", "shortcut"):
        if args.model == "all":
            print()
        report_shortcut()


if __name__ == "__main__":
    main()
