from __future__ import annotations

import math
import sys
import time
from pathlib import Path

# Make the ``supertonic``/``echo`` packages and ``__style__`` importable when
# running this script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo.config import config as echo_config
from echo.tokenizer import Tokenizer
from echo.training.collate import collate_fn
from echo.training.dataset import EchoDataset

from supertonic import config
from supertonic.fold import fold_latent
from supertonic.model import Supertonic

# Training hyperparameters and the data pipeline are shared with Echo (same
# data, same optimizer settings) so the two architectures are compared under
# identical conditions.
train_cfg = echo_config.training


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
    model: Supertonic,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    text_dropout_p: float = 0.0,
) -> torch.Tensor:
    """Interpolate between noise x0 and data x1; predict the velocity x1 - x0.

    The codec latents are folded (24ch x 6 sub-frames -> 144ch, 6x coarser
    time grid) before the model sees them, exactly as in the original
    Supertonic pipeline. With probability ``text_dropout_p`` (per sample),
    the text/style conditioning is replaced by the model's learned null
    conditions, enabling classifier-free guidance at inference time.
    """
    text = batch["text"].to(device)
    text_mask = batch["text_key_padding_mask"].to(device)

    # Fold latents onto the model's grid: (B, T, 24) -> (B, T/6, 144).
    x1, latent_mask = fold_latent(
        batch["latent"].to(device),
        batch["latent_key_padding_mask"].to(device),
        config.chunk_compress_factor,
    )

    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=device)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
    target = x1 - x0

    text_drop_mask = None
    if text_dropout_p > 0.0:
        text_drop_mask = torch.rand(x1.shape[0], device=device) < text_dropout_p

    pred = model(text, xt, t, text_mask, latent_mask, text_drop_mask)
    return _masked_mse(pred, target, latent_mask)


def main() -> None:
    cfg = train_cfg
    torch.manual_seed(cfg.seed)

    device = _select_device()
    data_dir = _REPO_ROOT / cfg.data_dir
    output_dir = _REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("Supertonic - Flow Matching Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    latent_dir = data_dir / "latents"
    latent_stats = latent_dir / "latent_stats.npz"
    dataset = EchoDataset(
        data_dir / "phonemes.csv", latent_dir, tokenizer,
        latent_stats=latent_stats,
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
    model = Supertonic().to(device)
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
    print_info(
        "Latent normalization",
        f"enabled ({latent_stats})" if latent_stats.is_file() else "disabled (stats file missing)",
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

    for epoch in range(cfg.num_epochs):
        for batch in train_loader:
            loss = _flow_matching_loss(model, batch, device, cfg.text_dropout)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0:
                elapsed = time.perf_counter() - t_start
                print_info(
                    f"epoch {epoch + 1}/{cfg.num_epochs} step {step}/{total_steps}",
                    f"loss {loss.item():.4f} | lr {scheduler.get_last_lr()[0]:.2e}"
                    f" | {elapsed / step:.2f}s/it",
                    Colors.OKGREEN,
                )

            if step % cfg.save_every == 0:
                ckpt = output_dir / f"supertonic_step{step}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step}, ckpt)
                print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

        # --- Validation -------------------------------------------------------
        if len(val_set) > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    val_loss += _flow_matching_loss(model, batch, device).item()
            val_loss /= len(val_loader)
            model.train()
            print_info(f"epoch {epoch + 1}/{cfg.num_epochs} val",
                       f"loss {val_loss:.4f}", Colors.WARNING)

    # --- Final save -----------------------------------------------------------
    ckpt = output_dir / "supertonic_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step}, ckpt)
    print_separator()
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)


if __name__ == "__main__":
    main()
