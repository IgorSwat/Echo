#!/usr/bin/env python3
"""Train the EchoFM flow-matching model on precomputed latents."""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time

import torch

from __common__ import (
    REPO_ROOT,
    build_optimizer,
    load_tokenizer,
    phonemes_csv,
    save_checkpoint,
    seed_everything,
    select_device,
    split_loaders,
)
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.fm_model import EchoFM
from echo.training.dataset import EchoDataset


def _flow_matching_loss(
    model: EchoFM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    text_dropout_p: float = 0.0,
) -> torch.Tensor:
    """Interpolate between a dithered distil x0 and data x1; predict x1 - x0.

    The distil is the starting state of the transport, not a side input: the
    model is handed the AR stage's own latent and has to carry it to the data
    distribution.

    ``config.fm_model.source_noise`` dithers that starting state, with fresh
    noise every step. Without it the source is a deterministic function of the
    target, and the L2-optimal velocity is then the mean of every rendering
    consistent with that distil: measurably blurry output that gets worse the
    longer the model trains. The dither is the seed that lets detail be *sampled*
    instead of averaged.

    With probability ``text_dropout_p`` (per sample) the text conditioning is
    replaced by the model's learned null-text condition, which is what enables
    classifier-free guidance at inference.

    The loss is a mean squared error over valid (non-padded) positions only.
    """
    text = batch["text"].to(device)
    x1 = batch["latent"].to(device)
    x0 = batch["distil"].to(device)
    text_mask = batch["text_key_padding_mask"].to(device)
    latent_mask = batch["latent_key_padding_mask"].to(device)

    sigma = config.fm_model.source_noise
    if sigma > 0.0:
        x0 = x0 + sigma * torch.randn_like(x0)

    t = torch.rand(x1.shape[0], device=device)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    text_drop_mask = None
    if text_dropout_p > 0.0:
        text_drop_mask = torch.rand(x1.shape[0], device=device) < text_dropout_p

    pred = model(text, xt, t, text_mask, latent_mask, text_drop_mask)

    mask = latent_mask.unsqueeze(-1).float()                         # (B, T, 1)
    sq_err = (pred - target).pow(2) * mask                           # (B, T, C)

    return sq_err.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)


def main() -> None:
    cfg = config.training.fm

    seed_everything(cfg.seed)
    device = select_device()
    data_dir = REPO_ROOT / cfg.data_dir
    output_dir = REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / "loss_log.csv", "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,val_loss,lr\n")

    print_header("Echo - Flow Matching Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = load_tokenizer()
    latent_dir = data_dir / "latents"
    latent_stats = latent_dir / "latent_stats.npz"
    manifest = phonemes_csv(data_dir)
    dataset = EchoDataset(
        manifest, latent_dir, data_dir / "distils", tokenizer,
        latent_stats=latent_stats,
        load_codec=False,                          # flow matching runs on latents only
        norm_mode=config.latent_norm,
    )
    train_loader, val_loader, train_set, val_set = split_loaders(dataset, cfg)

    # --- Model / optimizer --------------------------------------------------
    model = EchoFM().to(device)
    total_steps = cfg.num_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Manifest", manifest.name, Colors.OKCYAN if manifest.name != "phonemes.csv"
               else Colors.WARNING)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
    print_info("Source noise",
               f"sigma {config.fm_model.source_noise:g}  (x0 = distil + sigma * eps)"
               if config.fm_model.source_noise > 0 else
               "0 — deterministic pairing; the L2 optimum is the conditional mean (blur)",
               Colors.OKCYAN if config.fm_model.source_noise > 0 else Colors.WARNING)
    if config.latent_norm == "instance":
        print_info("Latent normalization",
                   "per instance (each utterance by its own distil's channel stats; "
                   "inference must invert with the same numbers)", Colors.OKCYAN)
    else:
        print_info(
            "Latent normalization",
            f"per dataset ({latent_stats})" if latent_stats.is_file()
            else "disabled (stats file missing)",
            Colors.OKCYAN if latent_stats.is_file() else Colors.WARNING,
        )
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(cfg.batch_size))
    print_info("Total steps", str(total_steps))
    print_separator()

    # --- Training loop ------------------------------------------------------
    model.train()
    step = 0
    t_start = time.perf_counter()
    best_val_loss = float("inf")
    val_loss = float("nan")
    epochs_no_improve = 0

    for epoch in range(cfg.num_epochs):
        epoch_train_loss = 0.0
        epoch_train_batches = 0

        for batch in train_loader:
            loss = _flow_matching_loss(model, batch, device, cfg.text_dropout)
            epoch_train_loss += loss.item()
            epoch_train_batches += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t_start
                lr = scheduler.get_last_lr()[0]
                print_info(
                    f"epoch {epoch + 1}/{cfg.num_epochs} step {step}/{total_steps}",
                    f"loss {loss.item():.4f} | lr {lr:.2e} | {elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )
                log_file.write(f"{step},{epoch + 1},{loss.item():.6f},,{lr:.6e}\n")
                log_file.flush()

        # --- Validation -----------------------------------------------------
        if len(val_set) == 0:
            continue

        model.eval()
        train_loss_avg = epoch_train_loss / epoch_train_batches
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += _flow_matching_loss(model, batch, device).item()
        val_loss /= len(val_loader)
        model.train()

        lr = scheduler.get_last_lr()[0]
        print_info(f"epoch {epoch + 1}/{cfg.num_epochs} val",
                   f"loss {val_loss:.4f} (train: {train_loss_avg:.4f})", Colors.WARNING)
        log_file.write(
            f"{step},{epoch + 1},{train_loss_avg:.6f},{val_loss:.6f},{lr:.6e}\n"
        )
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = output_dir / "echo_best.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
        else:
            epochs_no_improve += 1

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = output_dir / f"echo_epoch{epoch + 1}.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

        if epochs_no_improve >= cfg.early_stop:
            print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                       Colors.WARNING)
            break

    # --- Final save ---------------------------------------------------------
    ckpt = output_dir / "echo_final.pt"
    save_checkpoint(ckpt, model, optimizer, step, cfg.num_epochs, val_loss)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f}", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
