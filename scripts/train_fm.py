from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.fm_model import EchoFM
from echo.tokenizer import Tokenizer
from echo.training.collate import collate_fn
from echo.training.dataset import EchoDataset


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup followed by cosine decay to zero."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over valid (non-padded) latent positions only."""
    mask = mask.unsqueeze(-1).float()                            # (B, T, 1)
    sq_err = (pred - target).pow(2) * mask                       # (B, T, C)
    return sq_err.sum() / (mask.sum() * pred.shape[-1]).clamp_min(1.0)


def _flow_matching_loss(
    model: EchoFM,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    text_dropout_p: float = 0.0,
) -> torch.Tensor:
    """Interpolate between noise x0 and data x1; predict the velocity x1 - x0.

    The transport runs from a unit Gaussian, and the distil latent enters as
    *conditioning* rather than as the starting point. That distinction is what
    keeps the model generative: pairing each target with its own distil makes
    the source a deterministic function of the target, and against a
    deterministic pairing the L2-optimal velocity is the mean over every
    rendering consistent with that distil — the blur that shows up as a washed
    out, over-smoothed output. Sampling x0 fresh each time restores the seed the
    model needs to produce detail instead of averaging it away.

    With probability ``text_dropout_p`` (per sample), the text conditioning is
    replaced by the model's learned null-text condition, enabling
    classifier-free guidance at inference time.
    """
    text = batch["text"].to(device)
    x1 = batch["latent"].to(device)
    distil = batch["distil"].to(device)
    text_mask = batch["text_key_padding_mask"].to(device)
    latent_mask = batch["latent_key_padding_mask"].to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    text_drop_mask = None
    if text_dropout_p > 0.0:
        text_drop_mask = torch.rand(x1.shape[0], device=device) < text_dropout_p

    pred = model(text, xt, t, distil, text_mask, latent_mask, text_drop_mask)
    return _masked_mse(pred, target, latent_mask)


def main() -> None:
    cfg = config.training.fm

    # --- Reproducibility ----------------------------------------------------
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = _select_device()
    data_dir = _REPO_ROOT / cfg.data_dir
    output_dir = _REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Loss log -----------------------------------------------------------
    loss_log = output_dir / "loss_log.csv"
    log_file = open(loss_log, "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,val_loss,lr\n")

    print_header("Echo - Flow Matching Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    latent_dir = data_dir / "latents"
    distils_dir = data_dir / "distils"
    latent_stats = latent_dir / "latent_stats.npz"
    dataset = EchoDataset(
        data_dir / "phonemes.csv", latent_dir, distils_dir, tokenizer,
        latent_stats=latent_stats,
        load_codec=False,                          # flow matching runs on latents only
        norm_mode=config.latent_norm,
    )

    val_len = int(len(dataset) * cfg.val_ratio)
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_len, val_len],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
    )

    # --- Model / optimizer --------------------------------------------------
    model = EchoFM().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, cfg.warmup_steps, total_steps)
    )

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
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
                    f"loss {loss.item():.4f} | lr {lr:.2e}"
                    f" | {elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )
                log_file.write(f"{step},{epoch + 1},{loss.item():.6f},,{lr:.6e}\n")
                log_file.flush()

        # --- Validation -------------------------------------------------------
        if len(val_set) > 0:
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
            log_file.write(f"{step},{epoch + 1},{train_loss_avg:.6f},{val_loss:.6f},{lr:.6e}\n")
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ckpt = output_dir / "echo_best.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % cfg.save_every == 0:
                ckpt = output_dir / f"echo_epoch{epoch + 1}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

            if epochs_no_improve >= cfg.early_stop:
                print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                           Colors.WARNING)
                break

    # --- Final save -----------------------------------------------------------
    ckpt = output_dir / "echo_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": cfg.num_epochs, "val_loss": val_loss}, ckpt)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f}", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
