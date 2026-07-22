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
    codec_table = sum(p.numel() for p in model.codec_embed.embedding.parameters())
    codec_mlp = sum(p.numel() for p in model.codec_embed.fuse_mlp.parameters())
    codec_emb = codec_table + codec_mlp
    special_emb = (
        model.bos_embed.numel()
        + model.ref_text_eos_embed.numel()
        + model.ref_codec_eos_embed.numel()
        + model.text_eos_embed.numel()
    )
    emb_total = text_token + codec_emb + special_emb

    print_info("Text token embedding", _fmt(text_token))
    print_info("Codec embedding", _fmt(codec_emb))
    print(f"    ├─ embedding table:  {_fmt(codec_table)}")
    print(f"    └─ fuse MLP:        {_fmt(codec_mlp)}")
    print_info("Special token embeddings", _fmt(special_emb))
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

    # --- Codebook projector ---
    print_section("Codebook projector")
    proj_total = sum(p.numel() for p in model.projector.parameters())
    proj_num_layers = len(model.projector.transformer.blocks)
    proj_attn_total = 0
    proj_ffn_total = 0
    proj_norm_total = 0
    for i, block in enumerate(model.projector.transformer.blocks):
        a = sum(p.numel() for p in block.attn.parameters())
        f = sum(p.numel() for p in block.ffn.parameters())
        n = sum(p.numel() for p in block.norm1.parameters()) + sum(p.numel() for p in block.norm2.parameters())
        proj_attn_total += a
        proj_ffn_total += f
        proj_norm_total += n
        print_info(f"Block {i:2d}", f"{_fmt(a + f + n)}  (attn: {_fmt(a)}, FFN: {_fmt(f)})")
    proj_other = proj_total - proj_attn_total - proj_ffn_total - proj_norm_total
    print_info("Projections + embeddings", _fmt(proj_other))
    print_separator()
    print_info("Attention total", f"{_fmt(proj_attn_total)} ({_pct(proj_attn_total, total)})")
    print_info("FFN total", f"{_fmt(proj_ffn_total)} ({_pct(proj_ffn_total, total)})")
    print_info("LayerNorm total", f"{_fmt(proj_norm_total)} ({_pct(proj_norm_total, total)})")
    print_info("Projector subtotal", f"{_fmt(proj_total)} ({_pct(proj_total, total)})", Colors.OKCYAN)

    print()
    print_header(f"Grand total: {_fmt(total)}")
    if trainable < total:
        print_info("Frozen parameters", _fmt(total - trainable), Colors.WARNING)


if __name__ == "__main__":
    main()
