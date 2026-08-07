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
from echo.ar_model import EchoAR
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


def _add_bos_eos(
    codec: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrap each sequence as ``[BOS, frame_0 ... frame_{L-1}, EOS]``, right-padded.

    Padding sits at the end of every sequence, so BOS goes at index 0, the real
    frames shift one step right, and EOS lands at index ``length + 1``. The
    tensor grows by two along time; positions past the EOS stay ``prosody_pad``.

    Args:
        codec: ``(B, T, L)`` token ids, right-padded with ``config.prosody_pad``.
        mask:  ``(B, T)`` bool, True on real frames.

    Returns ``(B, T + 2, L)`` and its ``(B, T + 2)`` validity mask.
    """
    B, T, L = codec.shape
    lengths = mask.sum(dim=1)                                    # (B,)

    seq = torch.full(
        (B, T + 2, L), config.prosody_pad, dtype=codec.dtype, device=codec.device
    )
    seq[:, 0] = config.prosody_bos
    # Copy real frames only: whatever sits in the padded region of `codec` must
    # not survive into the targets, where it would be supervised instead of
    # skipped by the loss's ignore_index.
    seq[:, 1 : T + 1] = codec.masked_fill(~mask.unsqueeze(-1), config.prosody_pad)
    seq[torch.arange(B, device=codec.device), lengths + 1] = config.prosody_eos

    # BOS and EOS are both real positions, hence lengths + 2.
    seq_mask = torch.arange(T + 2, device=codec.device)[None] < (lengths + 2)[:, None]

    return seq, seq_mask                                         # (B, T + 2, L), (B, T + 2)


def _ar_loss(
    model: EchoAR,
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forced next-token cross-entropy over both codebook layers.

    The BOS/EOS-wrapped sequence is split into input and target halves shifted
    by one frame, so the model reads frame i and predicts frame i+1: BOS gives
    the first real frame, and the last real frame gives the EOS. The shift is
    what keeps BOS out of the targets — it is only ever read, never predicted.
    Padded targets are dropped via ``ignore_index``; a real token can never
    collide with the pad id because the codec alphabet stops below
    ``prosody_pad``.

    Under intra-frame conditioning the lower token layers of the target frame
    are handed back to the model as ``cond_tokens``, so head ``k`` predicts
    layer ``k`` knowing layers ``< k`` of the very frame it is producing. That
    is teacher forcing along the codebook axis, exactly as the frame history is
    teacher-forced along time; the top layer is never fed to itself, so no head
    ever sees its own label.
    """
    codec = batch["codec"].to(device)                            # (B, T, 2)
    text = batch["text"].to(device)                              # (B, S)
    codec_mask = batch["codec_key_padding_mask"].to(device)      # (B, T)
    text_mask = batch["text_key_padding_mask"].to(device)        # (B, S)

    seq, seq_mask = _add_bos_eos(codec, codec_mask)              # (B, T + 2, 2), (B, T + 2)
    inputs = seq[:, :-1]                                         # (B, T + 1, 2)
    targets = seq[:, 1:]                                         # (B, T + 1, 2)
    # Shifting the mask the same way as the targets leaves exactly the positions
    # that carry a supervised prediction: BOS plus every real frame, stopping
    # short of the EOS itself (which is only ever a target, never read).
    input_mask = seq_mask[:, 1:]                                 # (B, T + 1)

    cond_tokens = targets[..., :-1] if config.ar_model.head_intra_frame_cond else None

    logits = model(inputs, text, input_mask, text_mask,
                   cond_tokens=cond_tokens)                      # (B, T + 1, 2, V)

    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=config.prosody_pad,
    )


def main() -> None:
    cfg = config.training.ar

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
    loss_log = output_dir / "ar_loss_log.csv"
    log_file = open(loss_log, "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,val_loss,lr\n")

    print_header("Echo - Autoregressive Prosody Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    codec_dir = data_dir / "codecs"
    dataset = EchoDataset(
        data_dir / "phonemes.csv", None, None, tokenizer,
        codec_dir=codec_dir,
        codec_layers=EchoAR.NUM_TOKEN_LAYERS,
        load_latent=False,                         # the AR model runs on codec tokens only
        load_distil=False,
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
    model = EchoAR().to(device)
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
    print_info("Codec layers", str(EchoAR.NUM_TOKEN_LAYERS))
    print_info("Prosody vocab", f"{config.prosody_vocab_size:,} (pad {config.prosody_pad}, "
                                f"bos {config.prosody_bos}, eos {config.prosody_eos})")
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
            loss = _ar_loss(model, batch, device)
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
                    f"loss {loss.item():.4f} | ppl {math.exp(min(loss.item(), 20)):.1f}"
                    f" | lr {lr:.2e} | {elapsed / step:.2f}s/it",
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
                    val_loss += _ar_loss(model, batch, device).item()
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
                ckpt = output_dir / "echo_ar_best.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
            else:
                epochs_no_improve += 1

            if (epoch + 1) % cfg.save_every == 0:
                ckpt = output_dir / f"echo_ar_epoch{epoch + 1}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "epoch": epoch + 1, "val_loss": val_loss}, ckpt)
                print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

            if epochs_no_improve >= cfg.early_stop:
                print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                           Colors.WARNING)
                break

    # --- Final save -----------------------------------------------------------
    ckpt = output_dir / "echo_ar_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "epoch": cfg.num_epochs, "val_loss": val_loss}, ckpt)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f}", Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
