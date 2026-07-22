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
from echo.tokenizer import Tokenizer
from echo.training.dataset import EchoDataset
from echo.training.collate import collate_fn


IGNORE_INDEX = -100


def _load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _build_model(cfg: dict) -> Echo:
    return Echo(
        text_vocab_size=cfg["vocab_size"]["text"],
        text_emb_dim=cfg["embedding_dim"],
        d_emb=cfg["embedding_dim"],
        d_model=cfg["decoder"]["hidden_dim"],
        d_repr=cfg["intermediate_dim"],
        num_layers=cfg["decoder"]["no_layers"],
        num_heads=cfg["decoder"]["no_heads"],
        ffn_dim=cfg["decoder"]["ffn_dim"],
        dropout=cfg["decoder"]["dropout"],
        num_pred_heads=cfg["heads"]["no_heads"],
        pred_hidden_dim=cfg["heads"]["hidden_dim"],
        codec_logit_dim=cfg["heads"]["logit_dim"],
        pred_num_layers=cfg["heads"]["no_layers"],
        pred_dropout=cfg["heads"]["dropout"],
    )


def _build_dataloaders(
    cfg: dict,
    root: Path,
    batch_size: int,
    val_fraction: float,
) -> tuple[DataLoader, DataLoader | None]:
    tokenizer = Tokenizer(root / "models" / "phoneme_vocab.json")
    tc = cfg["training"]
    ds = EchoDataset(
        Path(tc["phonemes_csv"]),
        Path(tc["codec_dir"]),
        tokenizer,
    )

    # Bind the full dataset to the collate function so each batch can draw a
    # shared (ref_text, ref_audio_codec) reference pair from anywhere in the
    # training set, not just from the current batch.
    collate = partial(collate_fn, dataset=ds)

    if val_fraction > 0:
        val_size = max(1, int(len(ds) * val_fraction))
        train_size = len(ds) - val_size
        train_ds, val_ds = random_split(ds, [train_size, val_size])
        val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
        return train_dl, val_dl

    train_dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
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
    cfg = _load_config(str(root / args.config))
    train_cfg = cfg["training"]

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
    model = _build_model(cfg)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
        print(f"loaded checkpoint: {args.checkpoint}")
    model = model.to(device).train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {total_params / 1e6:.1f}M")

    # Build the dataset from provided path and config setup
    train_dl, val_dl = _build_dataloaders(cfg, root, train_cfg["batch_size"], train_cfg["val_fraction"])
    print(f"batches per epoch: {len(train_dl)}")
    if val_dl:
        print(f"val batches: {len(val_dl)}")

    # Build optimizer
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=train_cfg["learning_rate"], 
        weight_decay=train_cfg["weight_decay"]
    )

    total_steps = train_cfg["num_epochs"] * len(train_dl)
    warmup_fraction = train_cfg["warmup_fraction"]
    warmup_steps = int(warmup_fraction * total_steps)

    # Build scheduler
    def _lr_schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_schedule)

    output_dir = root / train_cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0

    # Main training loop
    for epoch in range(train_cfg["num_epochs"]):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0

        for ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths in train_dl:
            ref_text = ref_text.to(device)
            ref_audio = ref_audio.to(device)
            texts = texts.to(device)
            audio_codec = audio_codec.to(device)
            text_lengths = text_lengths.to(device)
            audio_lengths = audio_lengths.to(device)

            eos_id = cfg["special_tokens"]["eos"]
            logits = model(ref_text, ref_audio, texts, audio_codec, text_lengths, audio_lengths)
            targets = _build_targets(audio_codec, audio_lengths, eos_id)

            loss = _compute_loss(
                logits, targets,
                weighted=train_cfg["weighted_loss"],
                decay=train_cfg["loss_decay"],
            )

            opt.zero_grad()
            loss.backward()

            grad_clip = train_cfg["grad_clip"]
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            opt.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

            if global_step % train_cfg["log_interval"] == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"  step {global_step:6d} | loss {loss.item():.4f} | lr {lr:.2e}")

        avg_loss = epoch_loss / len(train_dl)
        elapsed = time.perf_counter() - epoch_start
        status = f"epoch {epoch + 1:3d} | train loss {avg_loss:.4f}"

        if val_dl:
            val_loss = _validate(model, val_dl, eos_id, device,
                                 weighted=train_cfg["weighted_loss"], decay=train_cfg["loss_decay"])
            status += f" | val loss {val_loss:.4f}"

        print(f"{status} | time {elapsed:.1f}s")

        save_every = train_cfg["save_interval"]
        is_last = epoch == train_cfg["num_epochs"] - 1
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
