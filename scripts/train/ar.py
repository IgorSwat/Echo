#!/usr/bin/env python3
"""Train the EchoAR autoregressive prosody model on precomputed codec tokens."""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
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
from echo.ar_model import EchoAR
from echo.config import TrainingConfig
from echo.training.dataset import EchoDataset


# -------------------
# History corruption
# -------------------

def _mask_rate(epoch: int, cfg: TrainingConfig) -> float:
    """Fraction of history frames to corrupt in ``epoch`` (1-based).

    Zero until ``history_mask_start_epoch``, then one ``history_mask_step`` more
    per epoch up to ``history_mask_max``. Ramped rather than switched on because
    early in training the model cannot predict a clean history yet, and
    corrupting one then is noise added to an already-hard task.
    """
    if cfg.history_mask_max <= 0.0 or epoch < cfg.history_mask_start_epoch:
        return 0.0

    steps = epoch - cfg.history_mask_start_epoch + 1

    return min(cfg.history_mask_max, steps * cfg.history_mask_step)


def _corrupt_history(
    inputs: torch.Tensor,                                            # (B, T, 2) long
    valid: torch.Tensor,                                             # (B, T) bool
    rate: float,
    cfg: TrainingConfig,
) -> torch.Tensor:
    """
    Replace a fraction of the *input* frames, leaving the targets untouched.

    Teacher forcing hands the model a history it never has at inference — its own
    frames, errors included — so corrupting the input teaches it to recover
    rather than compound the error.

    NOTE: corruption arrives in contiguous spans and is mostly plausible wrong
    tokens rather than the mask symbol, since that is what a real error looks
    like. BOS and anything outside `valid` are never touched.
    """

    if rate <= 0.0:
        return inputs

    B, T, L = inputs.shape
    device = inputs.device
    span_min, span_max = cfg.history_mask_span_min, cfg.history_mask_span_max
    mean_span = (span_min + span_max) / 2.0

    # Spans are drawn by their start, so the per-start probability has to be
    # divided by the mean span length for the *covered* fraction to come out at
    # `rate`.
    starts = torch.rand(B, T, device=device) < (rate / mean_span)
    starts &= valid
    starts[:, 0] = False                                             # never the BOS frame

    lengths = torch.randint(span_min, span_max + 1, (B, T), device=device)
    corrupt = torch.zeros(B, T, dtype=torch.bool, device=device)
    # Grow each start forward by its own length: at most span_max cheap shifts,
    # which keeps per-span random lengths without a Python loop over spans.
    for offset in range(span_max):
        active = starts & (lengths > offset)
        if offset:
            active = torch.cat(
                [torch.zeros(B, offset, dtype=torch.bool, device=device), active[:, :-offset]],
                dim=1,
            )
        corrupt |= active
    corrupt &= valid                                                 # never past the sequence

    # Most corrupted frames get a random real token; a minority get the mask
    # symbol, which the model may read but `generate` can never emit.
    use_mask = torch.rand(B, T, device=device) < cfg.history_mask_token_frac
    random_tokens = torch.randint(0, EchoAR.CODEBOOK_SIZE, (B, T, L), device=device)
    replacement = torch.where(
        use_mask.unsqueeze(-1), torch.full_like(random_tokens, config.prosody_mask), random_tokens
    )

    return torch.where(corrupt.unsqueeze(-1), replacement, inputs)


# --------
# Losses
# --------

def _add_bos_eos(
    codec: torch.Tensor,                                             # (B, T, L) right-padded
    mask: torch.Tensor,                                              # (B, T) bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrap each sequence as `[BOS, frame_0 ... frame_{L-1}, EOS]`, right-padded.

    NOTE: the tensor grows by two along time; positions past the EOS stay pad.
    """

    B, T, L = codec.shape
    lengths = mask.sum(dim=1)                                        # (B,)

    seq = torch.full(
        (B, T + 2, L), config.prosody_pad, dtype=codec.dtype, device=codec.device
    )
    seq[:, 0] = config.prosody_bos
    # Copy real frames only: whatever sits in the padded region of `codec` must
    # not survive into the targets, where it would be supervised instead of
    # skipped by the loss's ignore_index.
    seq[:, 1:T + 1] = codec.masked_fill(~mask.unsqueeze(-1), config.prosody_pad)
    seq[torch.arange(B, device=codec.device), lengths + 1] = config.prosody_eos

    # BOS and EOS are both real positions, hence lengths + 2.
    seq_mask = torch.arange(T + 2, device=codec.device)[None] < (lengths + 2)[:, None]

    return seq, seq_mask                                             # (B, T+2, L), (B, T+2)


def _ctc_loss(
    model: EchoAR,
    hidden: torch.Tensor,                                            # (B, T, hidden_dim)
    text: torch.Tensor,                                              # (B, S)
    input_mask: torch.Tensor,                                        # (B, T)
    text_mask: torch.Tensor,                                         # (B, S)
) -> torch.Tensor:
    """
    Auxiliary CTC: can the phoneme string be read back off the frame states?

    Teacher-forced cross-entropy cannot penalize a skipped word; CTC scores the
    whole frame sequence against the whole phoneme string, so a phoneme with
    nowhere to go collapses the probability of the entire sequence.

    NOTE: `reduction="mean"` puts the result on a per-phoneme scale comparable to
    the token cross-entropy, which is what makes one fixed weight meaningful.
    """

    log_probs = model.ctc_log_probs(hidden)                          # (B, T*u, V + 1)
    upsample = max(model.ctc_upsample, 1)

    # MPS has no `aten::_ctc_loss`, so the loss itself runs on CPU there. The
    # copy is differentiable, so gradients still reach the trunk, and only the
    # log-probs cross the boundary — a few MB per batch, once per step.
    device = log_probs.device
    host = torch.device("cpu") if device.type == "mps" else device

    loss = F.ctc_loss(
        log_probs.transpose(0, 1).to(host),                          # (T*u, B, V + 1)
        text.to(host),                                               # padded, read per length
        input_lengths=(input_mask.sum(1) * upsample).to(host),
        target_lengths=text_mask.sum(1).to(host),
        blank=model.ctc_blank,
        reduction="mean",
        # An utterance whose phoneme string still outruns its frames has no valid
        # alignment and scores inf. Dropping it beats poisoning the batch.
        zero_infinity=True,
    )

    return loss.to(device)


def _ar_loss(
    model: EchoAR,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    ctc_weight: float = 0.0,
    mask_rate: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Teacher-forced next-token cross-entropy, plus the auxiliary CTC term.

    Returns `(total, cross_entropy, ctc)` with the components detached, so the log
    can tell an improving model from an improving auxiliary head.

    The BOS/EOS-wrapped sequence is split into input and target halves shifted by
    one frame: the model reads frame i and predicts frame i+1, which also keeps
    BOS out of the targets. Padded targets drop out via `ignore_index`.

    NOTE: under intra-frame conditioning the target frame's lower layers come
    back as `cond_tokens` — teacher forcing along the codebook axis. The top
    layer is never fed to itself, so no head sees its own label.
    """

    codec = batch["codec"].to(device)                                # (B, T, 2)
    text = batch["text"].to(device)                                  # (B, S)
    codec_mask = batch["codec_key_padding_mask"].to(device)          # (B, T)
    text_mask = batch["text_key_padding_mask"].to(device)            # (B, S)

    seq, seq_mask = _add_bos_eos(codec, codec_mask)                  # (B, T+2, 2), (B, T+2)
    inputs = seq[:, :-1]                                             # (B, T + 1, 2)
    targets = seq[:, 1:]                                             # (B, T + 1, 2)
    # Shifting the mask the same way as the targets leaves exactly the positions
    # that carry a supervised prediction: BOS plus every real frame, stopping
    # short of the EOS itself (which is only ever a target, never read).
    input_mask = seq_mask[:, 1:]                                     # (B, T + 1)

    # Only the history the model *reads* is corrupted. The targets stay clean —
    # the model is asked to predict the true next frame despite a damaged
    # history — and so do `cond_tokens`, which are intra-frame teacher forcing
    # along the codebook axis, not history.
    inputs = _corrupt_history(inputs, input_mask, mask_rate, config.training.ar)

    cond_tokens = targets[..., :-1] if config.ar_model.head_intra_frame_cond else None

    want_ctc = ctc_weight > 0.0 and config.ar_model.ctc_enabled
    out = model(inputs, text, input_mask, text_mask,
                cond_tokens=cond_tokens, return_hidden=want_ctc)
    logits, hidden = out if want_ctc else (out, None)                # (B, T + 1, 2, V)

    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=config.prosody_pad,
    )
    if not want_ctc:
        return ce, ce.detach(), torch.zeros((), device=ce.device)

    ctc = _ctc_loss(model, hidden, text, input_mask, text_mask)

    return ce + ctc_weight * ctc, ce.detach(), ctc.detach()


def main() -> None:
    cfg = config.training.ar

    seed_everything(cfg.seed)
    device = select_device()
    data_dir = REPO_ROOT / cfg.data_dir
    output_dir = REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = open(output_dir / "ar_loss_log.csv", "w", encoding="utf-8")
    log_file.write("step,epoch,train_loss,val_loss,train_ctc,val_ctc,"
                   "train_total,val_total,lr\n")

    print_header("Echo - Autoregressive Prosody Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    dataset = EchoDataset(
        data_dir / "phonemes.csv", None, None, load_tokenizer(),
        codec_dir=data_dir / "codecs",
        codec_layers=EchoAR.NUM_TOKEN_LAYERS,
        load_latent=False,                         # the AR model runs on codec tokens only
        load_distil=False,
    )
    train_loader, val_loader, train_set, val_set = split_loaders(dataset, cfg)

    # --- Model / optimizer --------------------------------------------------
    model = EchoAR().to(device)
    total_steps = cfg.num_epochs * len(train_loader)
    optimizer, scheduler = build_optimizer(model, cfg, total_steps)

    use_ctc = cfg.ctc_weight > 0.0 and config.ar_model.ctc_enabled

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Train / val samples", f"{len(train_set)} / {len(val_set)}")
    print_info("Codec layers", str(EchoAR.NUM_TOKEN_LAYERS))
    print_info("Prosody vocab", f"{config.prosody_vocab_size:,} (pad {config.prosody_pad}, "
                                f"bos {config.prosody_bos}, eos {config.prosody_eos})")
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(cfg.batch_size))
    print_info("Total steps", str(total_steps))

    if use_ctc:
        print_info("CTC auxiliary", f"weight {cfg.ctc_weight:g}, "
                                    f"{config.ar_model.ctc_upsample}x upsampled grid "
                                    f"({12.5 * config.ar_model.ctc_upsample:g} Hz)",
                   Colors.OKCYAN)
    elif cfg.ctc_weight > 0.0:
        print_info("CTC auxiliary", "disabled (ar_model.ctc.enabled is false)", Colors.WARNING)
    else:
        print_info("CTC auxiliary", "disabled (ctc_weight is 0)", Colors.WARNING)

    if cfg.history_mask_max > 0.0:
        peak = cfg.history_mask_start_epoch + int(
            math.ceil(cfg.history_mask_max / cfg.history_mask_step)
        ) - 1
        print_info("History masking",
                   f"from epoch {cfg.history_mask_start_epoch}, "
                   f"+{cfg.history_mask_step:.1%}/epoch up to {cfg.history_mask_max:.1%} "
                   f"(reached at epoch {peak})", Colors.OKCYAN)
    else:
        print_info("History masking", "disabled (history_mask_max is 0)", Colors.WARNING)
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
        epoch_train_ctc = 0.0
        epoch_train_total = 0.0
        epoch_train_batches = 0

        mask_rate = _mask_rate(epoch + 1, cfg)
        if mask_rate > 0.0:
            print_info(f"epoch {epoch + 1}/{cfg.num_epochs} history masking",
                       f"{mask_rate:.1%} of frames "
                       f"(spans of {cfg.history_mask_span_min}-{cfg.history_mask_span_max})",
                       Colors.OKCYAN)

        for batch in train_loader:
            loss, ce, ctc = _ar_loss(model, batch, device, cfg.ctc_weight, mask_rate)
            epoch_train_loss += ce.item()
            epoch_train_ctc += ctc.item()
            epoch_train_total += loss.item()
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
                # `total` is what the optimizer descends. Perplexity stays on the
                # cross-entropy alone; folding the auxiliary term into it would
                # make the number meaningless.
                print_info(
                    f"epoch {epoch + 1}/{cfg.num_epochs} step {step}/{total_steps}",
                    (f"total {loss.item():.4f} | ce {ce.item():.4f}" if use_ctc
                     else f"loss {ce.item():.4f}")
                    + f" | ppl {math.exp(min(ce.item(), 20)):.1f}"
                    + (f" | ctc {ctc.item():.4f}" if use_ctc else "")
                    + f" | lr {lr:.2e} | {elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )
                log_file.write(f"{step},{epoch + 1},{ce.item():.6f},,{ctc.item():.6f},,"
                               f"{loss.item():.6f},,{lr:.6e}\n")
                log_file.flush()

        # --- Validation -----------------------------------------------------
        # Checkpoints are selected on the cross-entropy alone, not the total: the
        # CTC term is a means, not the goal. Validation also runs on a clean
        # history whatever the training schedule is doing, so val CE keeps
        # meaning one fixed thing — comparable across mask rates, across
        # ctc_weight values, and against checkpoints from before either existed.
        if len(val_set) == 0:
            continue

        model.eval()
        val_loss, val_ctc, val_total = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                total, ce, ctc = _ar_loss(model, batch, device, cfg.ctc_weight)
                val_loss += ce.item()
                val_ctc += ctc.item()
                val_total += total.item()
        val_loss /= len(val_loader)
        val_ctc /= len(val_loader)
        val_total /= len(val_loader)
        model.train()

        train_loss_avg = epoch_train_loss / epoch_train_batches
        train_ctc_avg = epoch_train_ctc / epoch_train_batches
        train_total_avg = epoch_train_total / epoch_train_batches
        lr = scheduler.get_last_lr()[0]

        print_info(f"epoch {epoch + 1}/{cfg.num_epochs} val",
                   (f"total {val_total:.4f} (train: {train_total_avg:.4f})"
                    f" | ce {val_loss:.4f} (train: {train_loss_avg:.4f})"
                    f" | ctc {val_ctc:.4f} (train: {train_ctc_avg:.4f})") if use_ctc
                   else f"loss {val_loss:.4f} (train: {train_loss_avg:.4f})",
                   Colors.WARNING)
        log_file.write(f"{step},{epoch + 1},{train_loss_avg:.6f},{val_loss:.6f},"
                       f"{train_ctc_avg:.6f},{val_ctc:.6f},"
                       f"{train_total_avg:.6f},{val_total:.6f},{lr:.6e}\n")
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            ckpt = output_dir / "echo_ar_best.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Best checkpoint", str(ckpt), Colors.OKCYAN)
        else:
            epochs_no_improve += 1

        if (epoch + 1) % cfg.save_every == 0:
            ckpt = output_dir / f"echo_ar_epoch{epoch + 1}.pt"
            save_checkpoint(ckpt, model, optimizer, step, epoch + 1, val_loss)
            print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

        if epochs_no_improve >= cfg.early_stop:
            print_info("Early stopping", f"no improvement for {epochs_no_improve} epochs",
                       Colors.WARNING)
            break

    # --- Final save ---------------------------------------------------------
    ckpt = output_dir / "echo_ar_final.pt"
    save_checkpoint(ckpt, model, optimizer, step, cfg.num_epochs, val_loss)
    print_separator()
    print_info("Best val loss", f"{best_val_loss:.6f}"
                               + (" (cross-entropy; checkpoints are selected on it)"
                                  if use_ctc else ""), Colors.OKCYAN)
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)
    log_file.close()


if __name__ == "__main__":
    main()
