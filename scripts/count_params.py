#!/usr/bin/env python3
"""Print a formatted parameter count report for the Echo model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from echo.model import Echo
from scripts.__style__ import Colors, print_header, print_section, print_separator, print_info


def _fmt(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} M"
    if n >= 1_000:
        return f"{n / 1_000:.1f} K"
    return str(n)


def _pct(part: int, total: int) -> str:
    return f"{100 * part / total:.1f}%"


def main() -> None:
    model = Echo()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print_header("Echo model — parameter count")
    print()
    print_info("Total parameters", _fmt(total), Colors.BOLD)
    print_info("Trainable", _fmt(trainable), Colors.OKGREEN)
    print()

    # --- Embeddings ---
    print_section("Embeddings")
    text_token = model.text_embed.token_embed.weight.numel()
    codec_emb = sum(p.numel() for p in model.codec_embed.parameters())
    special_emb = (
        model.bos_embed.numel()
        + model.ref_text_eos_embed.numel()
        + model.ref_codec_eos_embed.numel()
        + model.text_eos_embed.numel()
    )
    pos_emb = model.pos_embed.weight.numel()
    emb_total = text_token + codec_emb + special_emb + pos_emb

    print_info("Text token embedding", _fmt(text_token))
    print_info("Codec embedding", _fmt(codec_emb))
    print_info("Special token embeddings", _fmt(special_emb))
    print_info("Joint positional embedding", _fmt(pos_emb))
    print_info("Embeddings subtotal", f"{_fmt(emb_total)} ({_pct(emb_total, total)})", Colors.OKCYAN)

    # --- Projections ---
    print_section("Dimension adapters")
    in_proj = 0
    out_proj = 0
    if not isinstance(model.input_proj, torch.nn.Identity):
        in_proj = sum(p.numel() for p in model.input_proj.parameters())
    if not isinstance(model.output_proj, torch.nn.Identity):
        out_proj = sum(p.numel() for p in model.output_proj.parameters())
    proj_total = in_proj + out_proj

    print_info("Input projection (emb → model)", _fmt(in_proj))
    print_info("Output projection (model → repr)", _fmt(out_proj))
    print_info("Projections subtotal", f"{_fmt(proj_total)} ({_pct(proj_total, total)})", Colors.OKCYAN)

    # --- Transformer decoder ---
    print_section("Transformer decoder")
    num_layers = len(model.transformer.blocks)
    attn_total = 0
    ffn_total = 0
    norm_total = 0
    for i, block in enumerate(model.transformer.blocks):
        attn_params = sum(p.numel() for p in block.attn.parameters())
        ffn_params = sum(p.numel() for p in block.ffn.parameters())
        norm_params = sum(p.numel() for p in block.norm1.parameters()) + sum(p.numel() for p in block.norm2.parameters())
        attn_total += attn_params
        ffn_total += ffn_params
        norm_total += norm_params
        block_total = attn_params + ffn_params + norm_params
        print_info(f"Block {i:2d}", f"{_fmt(block_total)}  (attn: {_fmt(attn_params)}, FFN: {_fmt(ffn_params)})")

    final_norm_params = sum(p.numel() for p in model.transformer.norm.parameters())
    norm_total += final_norm_params
    xfmr_total = attn_total + ffn_total + norm_total
    print_info("Final layer norm", _fmt(final_norm_params))
    print_separator()
    print_info("Attention total", f"{_fmt(attn_total)} ({_pct(attn_total, total)})")
    print_info("FFN total", f"{_fmt(ffn_total)} ({_pct(ffn_total, total)})")
    print_info("LayerNorm total", f"{_fmt(norm_total)} ({_pct(norm_total, total)})")
    print_info("Transformer subtotal", f"{_fmt(xfmr_total)} ({_pct(xfmr_total, total)})", Colors.OKCYAN)

    # --- Prediction heads ---
    print_section("Prediction heads")
    heads_total = sum(p.numel() for p in model.heads.parameters())
    num_heads = model.heads.num_heads
    per_head = heads_total // num_heads
    print_info("Per head", _fmt(per_head))
    print_info(f"All {num_heads} heads", f"{_fmt(heads_total)} ({_pct(heads_total, total)})", Colors.OKCYAN)

    print()
    print_header(f"Grand total: {_fmt(total)}")
    if trainable < total:
        print_info("Frozen parameters", _fmt(total - trainable), Colors.WARNING)


if __name__ == "__main__":
    main()
