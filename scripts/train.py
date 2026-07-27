from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from echo.model import Echo
from echo.config import EchoConfig
from echo.tokenizer import Tokenizer
from echo.training.dataset import EchoDataset
from echo.training.collate import collate_fn


IGNORE_INDEX = -100


def _build_dataloaders(
    cfg: EchoConfig,
    root: Path,
) -> tuple[DataLoader, DataLoader | None]:
    tokenizer = Tokenizer(root / "models" / "phoneme_vocab.json")
    ds = EchoDataset(
        Path(cfg.training_phonemes_csv),
        Path(cfg.training_codec_dir),
        tokenizer,
    )

    if cfg.training_val_fraction > 0:
        val_size = max(1, int(len(ds) * cfg.training_val_fraction))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        train_collate = partial(collate_fn, dataset=train_ds)
        # A fixed held-out reference makes validation loss comparable between epochs.
        val_collate = partial(collate_fn, dataset=val_ds, reference_index=0)
        val_dl = DataLoader(val_ds, batch_size=cfg.training_batch_size, shuffle=False, collate_fn=val_collate, drop_last=False)
        train_dl = DataLoader(train_ds, batch_size=cfg.training_batch_size, shuffle=True, collate_fn=train_collate, drop_last=True)
        return train_dl, val_dl

    train_collate = partial(collate_fn, dataset=ds)
    train_dl = DataLoader(ds, batch_size=cfg.training_batch_size, shuffle=True, collate_fn=train_collate, drop_last=True)
    return train_dl, None


# IMPORTANT
def _compute_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    weighted: bool = False, decay: float = 0.9,
) -> torch.Tensor:
    # logits: (B, T, C, V), targets: (B, T, C)
    B, T, C, V = logits.shape

    mask = targets != IGNORE_INDEX
    logits_flat = logits.reshape(B * T * C, V)
    targets_flat = targets.reshape(B * T * C)

    ce = F.cross_entropy(logits_flat, targets_flat, reduction="none", ignore_index=IGNORE_INDEX)
    ce = ce.view(B, T, C)

    masked_ce = ce * mask

    if weighted:
        # Codebook layers are hierarchical — lower layers carry more information.
        # Layer c gets weight decay^c (layer 0 = 1.0, layer 1 = decay, ...).
        weights = decay ** torch.arange(C, device=logits.device)       # (C,)
        total_loss = (masked_ce * weights).sum()                       # scale each layer's CE by its weight
        num_valid = (mask * weights).sum().clamp(min=1)                # scale token counts by the same weights
    else:
        total_loss = masked_ce.sum()                                   # all layers weighted equally
        num_valid = mask.sum().clamp(min=1)                            # raw count of supervised tokens

    return total_loss / num_valid


def _build_targets(audio_codec: torch.Tensor, audio_lengths: torch.Tensor, eos_id: int) -> torch.Tensor:
    """
    Build real codec targets plus head-0 EOS; all other slots are ignored.
    """

    B, T, C = audio_codec.shape

    # IGNORE_INDEX = -100 is a default ignore index in F.cross_entropy
    targets = torch.full((B, T + 1, C), IGNORE_INDEX, dtype=audio_codec.dtype, device=audio_codec.device)

    for i in range(B):
        length = int(audio_lengths[i].item())
        targets[i, :length] = audio_codec[i, :length]
        targets[i, length, 0] = eos_id

    return targets


def _scheduled_sampling_probability(
    epoch: int,
    start_epoch: int,
    probability_increment: float,
    max_probability: float,
) -> float:
    """Return the scheduled-sampling probability for a zero-based epoch index."""
    epoch_number = epoch + 1
    if epoch_number < start_epoch:
        return 0.0
    return min(max_probability, (epoch_number - start_epoch + 1) * probability_increment)


@torch.no_grad()
def _mix_scheduled_audio(
    logits: torch.Tensor,
    audio_codec: torch.Tensor,
    audio_lengths: torch.Tensor,
    probability: float,
    first_special_token_id: int,
) -> tuple[torch.Tensor, float]:
    """Replace complete valid frames with greedy model predictions."""
    if probability <= 0:
        return audio_codec, 0.0

    B, T, _C = audio_codec.shape
    # Only real codec IDs are legal model inputs. PAD and EOS are output classes
    # but must never be selected as replacement frame tokens.
    predictions = logits[:, :T, :, :first_special_token_id].argmax(dim=-1)
    valid_frames = torch.arange(T, device=audio_codec.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
    replace_frames = (torch.rand((B, T), device=audio_codec.device) < probability) & valid_frames
    mixed_audio = torch.where(replace_frames.unsqueeze(-1), predictions, audio_codec)
    replacement_fraction = float(replace_frames.sum() / valid_frames.sum().clamp_min(1))
    return mixed_audio, replacement_fraction


@torch.no_grad()
def _validate(model: Echo, dl: DataLoader, eos_id: int, device: torch.device, weighted: bool, decay: float) -> float:
    model.eval()
    total_loss = 0.0
    for ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths in dl:
        ref_text = ref_text.to(device)
        ref_audio = ref_audio.to(device)
        texts = texts.to(device)
        audio_codec = audio_codec.to(device)
        text_lengths = text_lengths.to(device)
        audio_lengths = audio_lengths.to(device)

        logits = model(ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths)
        targets = _build_targets(audio_codec, audio_lengths, eos_id)

        total_loss += _compute_loss(logits, targets, weighted=weighted, decay=decay).item()

    model.train()
    return total_loss / len(dl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Echo model")
    parser.add_argument("--config", type=str, default="models/config.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    root = _REPO_ROOT
    cfg = EchoConfig.from_json(str(root / args.config))
    if cfg.training_scheduled_sampling_start_epoch < 1:
        raise ValueError("scheduled_sampling_start_epoch must be at least 1")
    if not 0 <= cfg.training_scheduled_sampling_probability_increment <= 1:
        raise ValueError("scheduled_sampling_probability_increment must be in [0, 1]")
    if not 0 <= cfg.training_scheduled_sampling_max_probability <= 1:
        raise ValueError("scheduled_sampling_max_probability must be in [0, 1]")

    # Select the best available device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    print(f"device: {device}")
    print(f"config: {args.config}")

    # Build the model from provided config
    model = Echo(cfg)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"loaded checkpoint: {args.checkpoint}")
    model = model.to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params / 1e6:.1f}M")

    # Build the dataset from provided path and config setup
    train_dl, val_dl = _build_dataloaders(cfg, root)
    print(f"batches per epoch: {len(train_dl)}")
    if val_dl:
        print(f"val batches: {len(val_dl)}")

    # Build optimizer
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.training_learning_rate, 
        weight_decay=cfg.training_weight_decay
    )

    total_steps = cfg.training_num_epochs * len(train_dl)
    warmup_steps = int(cfg.training_warmup_fraction * total_steps)

    # Build scheduler
    def _lr_schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_schedule)

    start_epoch = 0
    global_step = 0
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        if "optimizer_state_dict" in checkpoint:
            opt.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("global_step", start_epoch * len(train_dl)))
        print(f"resuming at epoch {start_epoch + 1}, step {global_step}")

    output_dir = root / cfg.training_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main training loop
    for epoch in range(start_epoch, cfg.training_num_epochs):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        epoch_replacement_fraction = 0.0
        scheduled_sampling_probability = _scheduled_sampling_probability(
            epoch,
            cfg.training_scheduled_sampling_start_epoch,
            cfg.training_scheduled_sampling_probability_increment,
            cfg.training_scheduled_sampling_max_probability,
        )

        for ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths in train_dl:
            ref_text = ref_text.to(device)
            ref_audio = ref_audio.to(device)
            texts = texts.to(device)
            audio_codec = audio_codec.to(device)
            text_lengths = text_lengths.to(device)
            audio_lengths = audio_lengths.to(device)

            eos_id = cfg.eos_id
            replacement_fraction = 0.0
            model_audio = audio_codec
            if scheduled_sampling_probability > 0:
                model.eval()
                with torch.no_grad():
                    teacher_logits = model(
                        ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths
                    )
                model.train()
                model_audio, replacement_fraction = _mix_scheduled_audio(
                    teacher_logits,
                    audio_codec,
                    audio_lengths,
                    scheduled_sampling_probability,
                    cfg.audio_pad_id,
                )

            logits = model(
                ref_text,
                ref_audio,
                texts,
                model_audio,
                text_lengths,
                audio_lengths,
                projector_audio_codec=audio_codec,
            )
            targets = _build_targets(audio_codec, audio_lengths, eos_id)

            loss = _compute_loss(
                logits, targets,
                weighted=cfg.training_weighted_loss,
                decay=cfg.training_loss_decay,
            )

            opt.zero_grad()
            loss.backward()

            grad_clip = cfg.training_grad_clip
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_replacement_fraction += replacement_fraction
            global_step += 1

            if global_step % cfg.training_log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr:.2e}"
                    f" | ss {scheduled_sampling_probability:.3f}"
                )

        avg_loss = epoch_loss / len(train_dl)
        elapsed = time.perf_counter() - epoch_start
        status = f"epoch {epoch + 1:3d} | train loss {avg_loss:.4f}"
        if scheduled_sampling_probability > 0:
            average_replacement = epoch_replacement_fraction / len(train_dl)
            status += (
                f" | ss probability {scheduled_sampling_probability:.3f}"
                f" | replaced {average_replacement:.3f}"
            )

        if val_dl:
            val_loss = _validate(model, val_dl, eos_id, device,
                                 weighted=cfg.training_weighted_loss, decay=cfg.training_loss_decay)
            status += f" | val loss {val_loss:.4f}"

        print(f"{status} | time {elapsed:.1f}s")

        save_every = cfg.training_save_interval
        is_last = epoch == cfg.training_num_epochs - 1
        if is_last or (save_every > 0 and (epoch + 1) % save_every == 0):
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(ckpt, output_dir / f"checkpoint_epoch{epoch + 1:03d}.pt")

    print("done")


if __name__ == "__main__":
    main()
