#!/usr/bin/env python3
"""Train the EchoNAR acoustic model on precomputed codec tokens."""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import random
import time

import torch
import torch.nn.functional as F

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
from echo.nar_model import EchoNAR
from echo.training.dataset import EchoDataset


# The cross-entropy skips these positions: padded frames, which carry the pad id
# on every layer and no target worth scoring.
IGNORE = -100


def _corrupt(codes: torch.Tensor, valid: torch.Tensor, rate: float) -> torch.Tensor:
    """
    Replace a fraction of the conditioning codes with random codebook entries.

    At inference the stack this model reads is never clean: layer 0 comes from
    EchoAR and every layer above it from this model's own draws, so an error
    anywhere below the layer being written is the normal case. Training on a
    ground-truth stack alone teaches it to trust that stack completely.

    NOTE: corruption is per (frame, layer), not per frame — an RVQ error is one
    codebook disagreeing, not the whole column going wrong at once.
    """

    if rate <= 0.0:
        return codes

    hit = (torch.rand(codes.shape, device=codes.device) < rate) & valid[..., None]
    random_codes = torch.randint(
        0, EchoNAR.CODEBOOK_SIZE, codes.shape, device=codes.device
    )

    return torch.where(hit, random_codes, codes)


def _nar_loss(
    model: EchoNAR,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    layer: int,
    token_noise: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cross-entropy for one codec layer, given the ground-truth layers below it.

    Returns `(loss, accuracy)`. Accuracy is worth watching next to the loss: on a
    residual codebook the entropy stays high even when the model is doing its
    job, so the absolute cross-entropy says much less than it does upstream.
    """

    codec = batch["codec"].to(device)                                # (B, T, K)
    text = batch["text"].to(device)                                  # (B, S)
    codec_mask = batch["codec_key_padding_mask"].to(device)          # (B, T)
    text_mask = batch["text_key_padding_mask"].to(device)            # (B, S)

    inputs = _corrupt(codec[..., :layer], codec_mask, token_noise)   # (B, T, layer)
    logits = model(inputs, layer, text, codec_mask, text_mask)       # (B, T, vocab)

    # Padded frames hold the pad id, which is outside the head's range; they drop
    # out through ignore_index rather than being scored against a code that does
    # not exist.
    targets = codec[..., layer].masked_fill(~codec_mask, IGNORE)     # (B, T)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE,
    )

    with torch.no_grad():
        correct = (logits.argmax(dim=-1) == targets) & codec_mask
        accuracy = correct.sum() / codec_mask.sum().clamp_min(1)

    return loss, accuracy.detach()


@torch.no_grad()
def _validate(
    model: EchoNAR, loader, device: torch.device
) -> tuple[list[float], list[float]]:
    """
    Per-layer validation loss and accuracy, on a clean stack.

    Every layer is scored on every batch rather than one sampled layer per step:
    the training draw is random, so a val number taken the same way would move
    with the draw instead of with the model. It also shows the shape that
    matters — how fast the layers get harder as the residual gets finer.
    """

    model.eval()
    losses = [0.0] * model.num_written
    accuracies = [0.0] * model.num_written
    batches = 0

    for batch in loader:
        for layer in range(1, model.num_layers):
            loss, accuracy = _nar_loss(model, batch, device, layer)
            losses[layer - 1] += loss.item()
            accuracies[layer - 1] += accuracy.item()
        batches += 1

    model.train()
    batches = max(batches, 1)

    return [l / batches for l in losses], [a / batches for a in accuracies]


def main() -> None:
    cfg = config.training.nar

    seed_everything(cfg.seed)
    device = select_device()
    data_dir = REPO_ROOT / cfg.data_dir
    output_dir = REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / "nar_loss_log.csv", "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    print_header("Echo - Non-Autoregressive Acoustic Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    dataset = EchoDataset(
        data_dir / "phonemes.csv", None, None, load_tokenizer(),
        codec_dir=data_dir / "codecs",
        codec_layers=EchoNAR.NUM_TOKEN_LAYERS,
        load_latent=False,                        # the NAR model runs on codec tokens only
        load_distil=False,
    )
    train_loader, val_loader, train_set, val_set = split_loaders(dataset, cfg)

    # --- Model / optimizer --------------------------------------------------
    model = EchoNAR().to(device)
    total_steps = cfg.num_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)

    # One layer per step, drawn per batch rather than per sample: the layer index
    # is what the AdaLN conditioning carries, and every row of a batch has to
    # agree on it for one head to score the whole batch.
    layer_rng = random.Random(cfg.seed)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
    print_info("Codec layers", f"{EchoNAR.NUM_TOKEN_LAYERS} "
                               f"(layer 0 from EchoAR, writing 1-{model.num_written})")
    print_info("Codebook size", f"{EchoNAR.CODEBOOK_SIZE:,}")
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(cfg.batch_size))
    print_info("Total steps", str(total_steps))
    if cfg.token_noise > 0.0:
        print_info("Token noise", f"{cfg.token_noise:.1%} of conditioning codes", Colors.OKCYAN)
    else:
        print_info("Token noise", "disabled (token_noise is 0)", Colors.WARNING)
    print_separator()

    # --- Training loop ------------------------------------------------------
    model.train()
    step = 0
    t_start = time.perf_counter()
    best_val_loss = float("inf")
    val_loss = float("nan")
    epochs_no_improve = 0

    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_batches = 0

        for batch in train_loader:
            layer = layer_rng.randint(1, model.num_written)
            loss, accuracy = _nar_loss(model, batch, device, layer, cfg.token_noise)
            epoch_loss += loss.item()
            epoch_acc += accuracy.item()
            epoch_batches += 1

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
                    f"layer {layer} | loss {loss.item():.4f}"
                    f" | ppl {math.exp(min(loss.item(), 20)):.1f}"
                    f" | acc {accuracy.item():.3f}"
                    f" | lr {lr:.2e} | {elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )
                log_file.write(f"{step},{epoch + 1},{loss.item():.6f},"
                               f"{accuracy.item():.6f},,,{lr:.6e}\n")
                log_file.flush()

        # --- Validation -----------------------------------------------------
        if len(val_set) == 0:
            continue

        layer_losses, layer_accs = _validate(model, val_loader, device)
        val_loss = sum(layer_losses) / len(layer_losses)
        val_acc = sum(layer_accs) / len(layer_accs)

        train_loss_avg = epoch_loss / epoch_batches
        train_acc_avg = epoch_acc / epoch_batches
        lr = scheduler.get_last_lr()[0]

        print_info(f"epoch {epoch + 1}/{cfg.num_epochs} val",
                   f"loss {val_loss:.4f} (train: {train_loss_avg:.4f})"
                   f" | acc {val_acc:.3f} (train: {train_acc_avg:.3f})",
                   Colors.WARNING)
        print_info("per layer",
                   "  ".join(f"{i + 1}: {l:.3f}/{a:.3f}"
                             for i, (l, a) in enumerate(zip(layer_losses, layer_accs))),
                   Colors.OKCYAN)
        log_file.write(f"{step},{epoch + 1},{train_loss_avg:.6f},{train_acc_avg:.6f},"
                       f"{val_loss:.6f},{val_acc:.6f},{lr:.6e}\n")
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = output_dir / "echo_nar_best.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
        else:
            epochs_no_improve += 1

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = output_dir / f"echo_nar_epoch{epoch + 1}.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

        if epochs_no_improve >= cfg.early_stop:
            print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                       Colors.WARNING)
            break

    # --- Final save ---------------------------------------------------------
    ckpt = output_dir / "echo_nar_final.pt"
    save_checkpoint(ckpt, model, optimizer, step, cfg.num_epochs, val_loss)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f} (mean over layers)", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
