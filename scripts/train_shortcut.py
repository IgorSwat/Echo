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
from torch.utils.data import DataLoader, random_split

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.shortcut_model import EchoShortcut
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


def _shortcut_loss(
    model: EchoShortcut,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Masked MSE between the upsampled prediction and the distil latents.

    The trunk's upsampling is a fixed ratio applied to the padded codec length,
    so the predicted length only approximately matches the distil length (the
    real ratio wobbles around 6.88 per utterance, and the collate pads the
    distil grid up to a multiple of 8). Both streams are therefore truncated to
    the shorter of the two before the loss, and only frames that are valid in
    *both* — a real distil frame backed by a real codec frame — are supervised.
    """
    codec = batch["codec"].to(device)                            # (B, T, 2)
    distil = batch["distil"].to(device)                          # (B, T_d, C)
    codec_mask = batch["codec_key_padding_mask"].to(device)      # (B, T)
    distil_mask = batch["distil_key_padding_mask"].to(device)    # (B, T_d)

    # Padded codec frames carry the pad id, which is a valid embedding row, so
    # they produce arbitrary output. They are excluded by the mask below rather
    # than zeroed: the depthwise convs would bleed them into neighbours anyway,
    # which is why the per-sample valid region is what the loss keys on.
    pred = model(codec)                                          # (B, L, C)

    T = min(pred.shape[1], distil.shape[1])
    pred, distil = pred[:, :T], distil[:, :T]                    # (B, T, C)

    # A predicted frame is only meaningful where its source codec frame was
    # real. Map the codec-rate mask onto the output rate by the same ratio the
    # trunk upsamples at, then intersect with the distil's own mask.
    codec_lengths = codec_mask.sum(dim=1)                        # (B,)
    ratio = pred.shape[1] / codec_mask.shape[1]
    pred_lengths = (codec_lengths.float() * ratio).floor().long()
    valid = torch.arange(T, device=device)[None] < pred_lengths[:, None]
    valid = valid & distil_mask[:, :T]                           # (B, T)

    diff = (pred - distil) ** 2                                  # (B, T, C)
    diff = diff * valid.unsqueeze(-1).to(diff.dtype)

    denom = valid.sum() * diff.shape[-1]

    return diff.sum() / denom.clamp(min=1)


def main() -> None:
    cfg = config.training.shortcut

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
    loss_log = output_dir / "shortcut_loss_log.csv"
    log_file = open(loss_log, "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,val_loss,lr\n")

    print_header("Echo - Shortcut (Mimi codec -> Blue latent) Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    # Distils are normalized with the same per-channel stats the flow-matching
    # run uses for its latents, so both models live on one scale.
    latent_stats = data_dir / "latents" / "latent_stats.npz"
    dataset = EchoDataset(
        data_dir / "phonemes.csv", None, data_dir / "distils", tokenizer,
        latent_stats=latent_stats,
        codec_dir=data_dir / "codecs",
        codec_layers=EchoShortcut.NUM_TOKEN_LAYERS,
        load_latent=False,                         # the shortcut maps codec -> distil
        load_distil=True,
        load_codec=True,
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
    model = EchoShortcut().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.num_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, cfg.warmup_steps, total_steps)
    )

    upsample = 1.0
    for _, factor in model.stage_specs:
        upsample *= factor

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
    print_info("Codec layers", str(EchoShortcut.NUM_TOKEN_LAYERS))
    print_info("Output dim", str(model.d_out))
    print_info("Upsample", f"x{upsample:g}")
    print_info(
        "Distil normalization",
        f"enabled ({latent_stats})" if latent_stats.is_file()
        else "disabled (stats file missing)",
        Colors.OKCYAN if latent_stats.is_file() else Colors.WARNING,
    )
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(cfg.batch_size))
    print_info("Total steps", str(total_steps))
    print_info(
        "Note",
        "predicted and distil lengths differ by a few frames per batch; both are "
        "truncated to the shorter one and the loss is masked",
        Colors.WARNING,
    )
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
            loss = _shortcut_loss(model, batch, device)
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
                    f"mse {loss.item():.4f} | lr {lr:.2e} | {elapsed / step:.2f}s/it",
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
                    val_loss += _shortcut_loss(model, batch, device).item()
            val_loss /= len(val_loader)
            model.train()
            lr = scheduler.get_last_lr()[0]
            print_info(f"epoch {epoch + 1}/{cfg.num_epochs} val",
                       f"mse {val_loss:.4f} (train: {train_loss_avg:.4f})", Colors.WARNING)
            log_file.write(f"{step},{epoch + 1},{train_loss_avg:.6f},{val_loss:.6f},{lr:.6e}\n")
            log_file.flush()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                ckpt = output_dir / "echo_shortcut_best.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % cfg.save_every == 0:
                ckpt = output_dir / f"echo_shortcut_epoch{epoch + 1}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

            if epochs_no_improve >= cfg.early_stop:
                print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                           Colors.WARNING)
                break

    # --- Final save -----------------------------------------------------------
    ckpt = output_dir / "echo_shortcut_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": cfg.num_epochs, "val_loss": val_loss}, ckpt)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f}", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
