from __future__ import annotations

import math
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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.model import Echo
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


def main() -> None:
    cfg = config.training
    torch.manual_seed(cfg.seed)

    device = _select_device()
    data_dir = _REPO_ROOT / cfg.data_dir
    output_dir = _REPO_ROOT / cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print_header("Echo - Flow Matching Training")
    print_separator()

    # --- Data ---------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    dataset = EchoDataset(data_dir / "phonemes.csv", data_dir / "latents", tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # --- Model / optimizer --------------------------------------------------
    model = Echo().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    total_steps = cfg.num_epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, cfg.warmup_steps, total_steps)
    )

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Samples", str(len(dataset)))
    print_info("Parameters", f"{sum(p.numel() for p in model.parameters()):,}")
    print_info("Batch size", str(cfg.batch_size))
    print_info("Total steps", str(total_steps))
    print_separator()

    # --- Training loop ------------------------------------------------------
    model.train()
    step = 0
    t_start = time.perf_counter()

    for epoch in range(cfg.num_epochs):
        for batch in loader:
            text = batch["text"].to(device)
            x1 = batch["latent"].to(device)
            text_mask = batch["text_key_padding_mask"].to(device)
            latent_mask = batch["latent_key_padding_mask"].to(device)

            # Flow matching: interpolate between noise x0 and data x1.
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.shape[0], device=device)
            xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
            target = x1 - x0

            pred = model(text, xt, t, text_mask, latent_mask)
            loss = _masked_mse(pred, target, latent_mask)

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
                ckpt = output_dir / f"echo_step{step}.pt"
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step}, ckpt)
                print_info("Checkpoint", str(ckpt), Colors.OKCYAN)

    # --- Final save -----------------------------------------------------------
    ckpt = output_dir / "echo_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step}, ckpt)
    print_separator()
    print_info("Final checkpoint", str(ckpt), Colors.OKCYAN)
    print_info("Total time", f"{time.perf_counter() - t_start:.1f}s", Colors.OKCYAN)


if __name__ == "__main__":
    main()
