#!/usr/bin/env python3
"""Detailed parameter breakdown of the Echo model.

Reports parameter counts for the TimeEncoder, TextEncoder, and each block in
the main processing stack, grouped by block type.

Usage:
    python scripts/count_params.py
"""

from __future__ import annotations

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
from echo.fm_model import EchoFM


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _fmt(n: int) -> str:
    return f"{n:,}"


def main() -> None:
    model = EchoFM()
    model.eval()

    total = _count_params(model)

    print_test_title("Echo — Parameter Breakdown")

    # --- Top-level components ---
    print_section("Top-level components")

    time_params = _count_params(model.time_encoder)
    text_params = _count_params(model.text_encoder)
    blocks_params = _count_params(model.blocks)
    final_params = _count_params(model.final_norm) + _count_params(model.out_proj)

    components = [
        ("TimeEncoder", time_params),
        ("TextEncoder", text_params),
        ("Main blocks", blocks_params),
        ("Final norm + out_proj", final_params),
    ]

    for label, n in components:
        pct = 100.0 * n / total if total else 0.0
        bar_len = int(round(pct / 100.0 * 30))
        bar = Colors.OKGREEN + "█" * bar_len + Colors.ENDC + "░" * (30 - bar_len)
        print(f"  {Colors.BOLD}{label:<22}{Colors.ENDC} {_fmt(n):>12}  {pct:5.1f}%  {bar}")

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)

    # --- Per-block breakdown ---
    print_section("Main blocks — per block")

    rows = []
    type_totals: OrderedDict[str, list[int]] = OrderedDict()

    for i, block in enumerate(model.blocks):
        n = _count_params(block)
        cls_name = type(block).__name__
        rows.append((i, cls_name, n))
        type_totals.setdefault(cls_name, []).append(n)

    # Individual blocks
    for i, cls_name, n in rows:
        pct = 100.0 * n / total if total else 0.0
        print(f"  [{i}] {Colors.BOLD}{cls_name:<20}{Colors.ENDC} {_fmt(n):>12}  {pct:5.1f}%")

    # Aggregated per type
    print_section("Main blocks — aggregated by type")

    for cls_name, counts in type_totals.items():
        subtotal = sum(counts)
        pct = 100.0 * subtotal / total if total else 0.0
        avg = subtotal / len(counts)
        print_info(
            f"{cls_name} (x{len(counts)})",
            f"{_fmt(subtotal)}  (avg {_fmt(int(round(avg)))})  {pct:5.1f}%",
        )

    print_separator()
    print_info("Total", _fmt(total), Colors.OKCYAN)


if __name__ == "__main__":
    main()
