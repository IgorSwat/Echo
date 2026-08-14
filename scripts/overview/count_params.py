#!/usr/bin/env python3
"""Detailed parameter breakdown of the Echo models, in pipeline order.

For EchoAR: the token embeddings, TextEncoder, decoder blocks and the
per-token-layer heads. For EchoFM: the TimeEncoder, TextEncoder and each block
of the main processing stack, grouped by block type.

Usage:
    python scripts/overview/count_params.py
    python scripts/overview/count_params.py --model ar
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from collections import OrderedDict

import torch

import __common__  # noqa: F401 — imported for its side effect: puts the repo root on sys.path
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

    type_totals: OrderedDict[str, list[int]] = OrderedDict()
    for i, block in enumerate(blocks):
        n = _count_params(block)
        cls_name = type(block).__name__
        type_totals.setdefault(cls_name, []).append(n)

        pct = 100.0 * n / total if total else 0.0
        print(f"  [{i}] {Colors.BOLD}{cls_name:<20}{Colors.ENDC} {_fmt(n):>12}  {pct:5.1f}%")

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

    print_section("Top-level components")
    _print_components([
        ("TimeEncoder", _count_params(model.time_encoder)),
        ("TextEncoder", _count_params(model.text_encoder)),
        ("Stem", _count_params(model.stem)),
        ("Main blocks", _count_params(model.blocks)),
        ("Final norm + out_proj",
         _count_params(model.final_norm) + _count_params(model.out_proj)),
    ], total)

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    _print_blocks(model.blocks, total, "Main blocks")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


def report_ar() -> None:
    model = EchoAR()
    model.eval()
    total = _count_params(model)

    print_header("EchoAR — autoregressive prosody model")

    print_section("Top-level components")
    components = [
        ("Token embeddings", _count_params(model.embed)),
        ("TextEncoder", _count_params(model.text_encoder)),
        ("Decoder", _count_params(model.decoder)),
        ("Head", _count_params(model.head)),
    ]
    if model.in_proj is not None:
        components.insert(3, ("Input projection", _count_params(model.in_proj)))

    _print_components(components, total)

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    print_section("Dimensions")
    print_info("Prosody vocab", _fmt(model.vocab_size))
    print_info("Embedding dim", str(model.emb_dim))
    print_info("Hidden dim", model.hidden_dim)
    print_info("Text context dim", config.ar_model.text_encoder_d_model)

    _print_blocks(model.decoder.blocks, total, "Decoder blocks")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


def main() -> None:
    parser = argparse.ArgumentParser(description="Echo parameter breakdown.")
    parser.add_argument("--model", type=str, default="all", choices=["all", "ar", "fm"],
                        help="which model to report on (default: all)")
    args = parser.parse_args()

    print_test_title("Echo — Parameter Breakdown")

    # Pipeline order: EchoAR writes the prosody tokens EchoFM then renders.
    if args.model in ("all", "ar"):
        report_ar()
    if args.model in ("all", "fm"):
        if args.model == "all":
            print()
        report_fm()


if __name__ == "__main__":
    main()
