#!/usr/bin/env python3
"""Train the two-rate EchoFMDual flow-matching model on precomputed latents.

The same data and objective as ``scripts/train/fm.py`` -- only the trunk
differs, so a run here is comparable to one there at matching settings.

Two things this trainer does that the single-stream one does not:

Validation runs on a FIXED timestep grid with fixed noise, not on t ~ U(0,1).
The per-t loss spans roughly 0.40 to 0.89, so a random-t validation number
carries enough variance of its own to hide a several-percent difference between
two runs. The random-t figure is still reported, since that is what older
checkpoints recorded.

The run is measured in optimizer steps rather than epochs, and the cosine
schedule anneals over exactly that many. Comparing two models means giving them
the same step budget, and a budget that anneals early flatters whichever model
converges fastest rather than whichever ends up best.

The trunk shape is not a flag. It comes from ``fm_model.dual`` in
``models/config.json``, the way the baseline's comes from ``fm_model.blocks``,
so a checkpoint here is described by the config rather than by whichever command
line produced it. Trying a different layout means editing that section.

Usage:
    python scripts/train/fm_dual.py --steps 4000
    python scripts/train/fm_dual.py --steps 20000 --tag dual_long --val-every 500
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from typing import Optional

import torch

from __common__ import (
    REPO_ROOT,
    build_optimizer,
    load_tokenizer,
    save_checkpoint,
    seed_everything,
    select_device,
    split_loaders,
)
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.fm_model_dual import EchoFMDual
from echo.training.dataset import EchoDataset

# Timesteps the fixed-grid validation averages over.
VAL_TIMESTEPS = (0.1, 0.3, 0.5, 0.7, 0.9)


def _flow_matching_loss(
    model: EchoFMDual,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    prosody_dropout_p: float = 0.0,
    t_fixed: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Interpolate between Gaussian noise x0 and data x1; predict x1 - x0.

    ``t_fixed`` and ``generator`` are what make validation repeatable: with both
    set, the same batch produces the same number every time it is scored.

    The loss is a mean squared error over valid (non-padded) positions only.
    """
    text = batch["text"].to(device)
    x1 = batch["latent"].to(device)
    text_mask = batch["text_key_padding_mask"].to(device)
    latent_mask = batch["latent_key_padding_mask"].to(device)

    # Layer 0 of the Mimi codec -- what the AR stage predicts.
    prosody = batch["codec"][..., 0].to(device)                      # (B, K)
    prosody_mask = batch["codec_key_padding_mask"].to(device)        # (B, K)

    x0 = torch.randn(x1.shape, device=device, generator=generator)   # (B, T, C)

    if t_fixed is None:
        t = torch.rand(x1.shape[0], device=device)
    else:
        t = torch.full((x1.shape[0],), float(t_fixed), device=device)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    prosody_drop_mask = None
    if prosody_dropout_p > 0.0:
        prosody_drop_mask = torch.rand(x1.shape[0], device=device) < prosody_dropout_p

    pred = model(text, xt, prosody, t,
                 text_mask, latent_mask, prosody_mask, prosody_drop_mask)

    mask = latent_mask.unsqueeze(-1).float()                         # (B, T, 1)
    sq_err = (pred - target).pow(2) * mask                           # (B, T, C)

    return sq_err.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)


@torch.no_grad()
def _validate(model, loader, device) -> tuple[float, float]:
    """(fixed-grid loss, random-t loss) over the whole validation split."""
    model.eval()
    grid_total, grid_n = 0.0, 0
    rand_total, rand_n = 0.0, 0

    for i, batch in enumerate(loader):
        for t in VAL_TIMESTEPS:
            gen = torch.Generator(device=device).manual_seed(9000 + i)
            grid_total += _flow_matching_loss(model, batch, device,
                                              t_fixed=t, generator=gen).item()
            grid_n += 1
        rand_total += _flow_matching_loss(model, batch, device).item()
        rand_n += 1

    model.train()

    return grid_total / max(grid_n, 1), rand_total / max(rand_n, 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the two-rate EchoFMDual model.")
    p.add_argument("--steps", type=int, default=None,
                   help="Optimizer steps to run; the cosine schedule anneals over "
                        "exactly this many (default: num_epochs x steps-per-epoch).")
    p.add_argument("--val-every", type=int, default=250,
                   help="Validate and maybe checkpoint every N steps (default: 250).")
    p.add_argument("--tag", type=str, default="echo_fm_dual",
                   help="Checkpoint name prefix (default: echo_fm_dual).")
    p.add_argument("--lr", type=float, default=None, help="Override the config LR.")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override the config batch size.")
    p.add_argument("--seed", type=int, default=None, help="Override the config seed.")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Where checkpoints go (default: the config's output_dir).")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = config.training.fm

    seed = args.seed if args.seed is not None else cfg.seed
    lr = args.lr if args.lr is not None else cfg.learning_rate
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    seed_everything(seed)
    device = select_device()
    data_dir = REPO_ROOT / cfg.data_dir
    output_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("Echo - Two-Rate Flow Matching Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = load_tokenizer()
    latent_dir = data_dir / "latents"
    latent_stats = latent_dir / "latent_stats.npz"

    # Refused rather than warned about: without the statistics the latents keep
    # their native per-channel std of ~0.04 against unit-variance noise, which
    # leaves the target almost entirely noise and the speech content at a
    # fraction of a percent of the loss. A run like that trains and converges
    # and means nothing, which is exactly why it has to fail loudly here.
    if config.latent_norm == "dataset" and not latent_stats.is_file():
        raise SystemExit(
            f"latent_norm='dataset' but {latent_stats} is missing. The latents "
            f"would be fed unnormalized and the run would be meaningless.\n"
            f"Build it with:\n"
            f"  python scripts/preprocess/latent_stats.py --latent-dir {latent_dir}"
        )

    dataset = EchoDataset(
        data_dir / "phonemes.csv", latent_dir, None, tokenizer,
        latent_stats=latent_stats,
        load_distil=False,                         # the transport starts from noise
        codec_dir=data_dir / "codecs",
        codec_layers=1,                            # layer 0 only: the prosody stream
        norm_mode=config.latent_norm,
    )

    loader_cfg = type(cfg)(**{**vars(cfg), "batch_size": batch_size, "seed": seed})
    train_loader, val_loader, train_set, val_set = split_loaders(dataset, loader_cfg)

    # --- Model / optimizer --------------------------------------------------
    model = EchoFMDual().to(device)

    total_steps = args.steps if args.steps else cfg.num_epochs * len(train_loader)
    sched_cfg = type(cfg)(**{**vars(cfg), "learning_rate": lr})
    optimizer, scheduler = build_optimizer(model, sched_cfg, total_steps)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
    dual = config.fm_model.dual
    print_info("Trunk", f"{dual.num_blocks} blocks, conv {dual.dim_a} / attn "
                        f"{dual.dim_b}, ffn x{dual.ffn_mult:g}, {dual.num_conv} "
                        f"conv per block, head {dual.head_upsample}")
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(batch_size))
    print_info("Learning rate", f"{lr:.2e}")
    print_info("Total steps", f"{total_steps} ({total_steps / max(len(train_loader), 1):.1f} epochs)")
    print_info("Validation", f"every {args.val_every} steps, on a fixed t-grid "
                             f"{VAL_TIMESTEPS}", Colors.OKCYAN)
    print_separator()

    log_file = open(output_dir / f"{args.tag}_loss_log.csv", "w", encoding="utf-8")
    log_file.write("step,train_loss,val_grid,val_random,lr\n")

    # --- Training loop ------------------------------------------------------
    model.train()
    step = 0
    running = None
    best_val = float("inf")
    val_grid = val_rand = float("nan")
    t_start = time.perf_counter()

    done = False
    while not done:
        for batch in train_loader:
            loss = _flow_matching_loss(model, batch, device, cfg.prosody_dropout)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            l = loss.item()
            running = l if running is None else 0.98 * running + 0.02 * l

            if step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t_start
                print_info(
                    f"step {step}/{total_steps}",
                    f"loss {running:.4f} | lr {scheduler.get_last_lr()[0]:.2e} | "
                    f"{elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )

            if step % args.val_every == 0 or step >= total_steps:
                if len(val_set):
                    val_grid, val_rand = _validate(model, val_loader, device)
                    print_info(f"step {step} val",
                               f"grid {val_grid:.4f} | random-t {val_rand:.4f} "
                               f"(train {running:.4f})", Colors.WARNING)
                    log_file.write(f"{step},{running:.6f},{val_grid:.6f},"
                                   f"{val_rand:.6f},{scheduler.get_last_lr()[0]:.6e}\n")
                    log_file.flush()

                    if val_grid < best_val:
                        best_val = val_grid
                        ckpt = output_dir / f"{args.tag}_best.pt"
                        save_checkpoint(ckpt, model, optimizer, step, step, val_grid)
                        print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)

            if step >= total_steps:
                done = True
                break

    # --- Final save ---------------------------------------------------------
    ckpt = output_dir / f"{args.tag}_final.pt"
    save_checkpoint(ckpt, model, optimizer, step, step, val_grid)
    print_separator()
    print_info("Best val (fixed grid)", f"{best_val:.6f}", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
