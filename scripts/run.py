#!/usr/bin/env python3
"""Generate audio with a trained Echo flow-matching model.

The model is trained with conditional flow matching
(``x_t = (1 - t) * noise + t * data``, velocity target ``data - noise``), so
sampling starts from Gaussian noise at ``t = 0`` and solves the ODE
``dx/dt = v(x, t)`` up to ``t = 1`` with a proper numerical integrator
(``--solver euler`` or the second-order ``--solver midpoint``/RK2).
The resulting audio latent is decoded to a waveform with BlueCodec.

Usage:
    python scripts/run.py --model checkpoints/echo_final.pt --text "həlˈOʊ wˈɜːld" \
        --duration 3.0 --steps 8 --cfg 3.0 --output out.wav
"""

from __future__ import annotations

import argparse
import math
import sys
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
import torchaudio
from bluecodec import BlueCodec

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.model import Echo
from echo.tokenizer import Tokenizer

# BlueCodec operates at 44.1 kHz with a hop of 512 samples per latent frame.
_SAMPLE_RATE = 44100
_HOP = 512


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _velocity(
    model: Echo,
    text_ids: torch.Tensor,
    x: torch.Tensor,
    t: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    """One velocity evaluation at (x, t), with classifier-free guidance.

    With ``cfg_scale != 1`` the model runs batch-doubled (conditioned +
    null-text) and extrapolates: v = v_uncond + cfg * (v_cond - v_uncond).
    """
    if cfg_scale == 1.0:
        return model(text_ids, x, t)
    drop = torch.tensor([False, True], device=x.device)
    v2 = model(text_ids.repeat(2, 1), x.repeat(2, 1, 1), t.repeat(2),
               None, None, drop)
    return v2[1:2] + cfg_scale * (v2[0:1] - v2[1:2])


@torch.no_grad()
def _generate(
    model: Echo,
    text_ids: torch.Tensor,
    num_frames: int,
    steps: int,
    cfg_scale: float,
    solver: str,
) -> torch.Tensor:
    """Integrate the velocity field from t=0 (noise) to t=1 (data).

    ``euler``    -- x <- x + v(x, t_i) * dt with t_i = i / steps; the canonical
                    flow-matching sampler (1 model evaluation per step).
    ``midpoint`` -- explicit midpoint (RK2): predict the state at t_i + dt/2
                    with an Euler half-step, then advance using the velocity
                    evaluated there (2 model evaluations per step).
    """
    dt = 1.0 / steps
    x = torch.randn(1, num_frames, config.latent_dim, device=text_ids.device)
    for i in range(steps):
        t0 = torch.full((1,), i / steps, device=text_ids.device)
        if solver == "euler":
            x = x + dt * _velocity(model, text_ids, x, t0, cfg_scale)
        else:                                                             # midpoint (RK2)
            v1 = _velocity(model, text_ids, x, t0, cfg_scale)
            x_mid = x + 0.5 * dt * v1
            t_mid = torch.full((1,), (i + 0.5) / steps, device=text_ids.device)
            x = x + dt * _velocity(model, text_ids, x_mid, t_mid, cfg_scale)
    return x                                                              # (1, T, C)


def _load_latent_stats(stats_path: Path, device: torch.device) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load per-channel mean/std (shape ``(latent_dim,)``) for denormalization."""
    if not stats_path.is_file():
        return None
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)   # (C,)
    std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)     # (C,)
    return mean, std


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with Echo.")
    parser.add_argument("--model", type=str, required=True, help="Path to a training checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True, help="Phoneme string to synthesize.")
    parser.add_argument("--duration", type=float, required=True, help="Audio duration in seconds.")
    parser.add_argument("--steps", type=int, default=8, help="Flow-matching integration steps (default: 8).")
    parser.add_argument("--solver", choices=("euler", "midpoint"), default="euler",
                        help="ODE integrator: 'euler' (1 model eval/step, default) or "
                             "'midpoint' (RK2, 2 evals/step, better at few steps).")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0; 1.0 disables guidance).")
    parser.add_argument(
        "--stats", type=str, default=None,
        help="Path to latent_stats.npz (mean/std) used to denormalize latents before codec decode. "
             "Defaults to <data_dir>/latents/latent_stats.npz from the training config.",
    )
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path.")
    args = parser.parse_args()

    device = _select_device()

    print_header("Echo - Generation")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Checkpoint", args.model)
    print_info("Steps", str(args.steps))
    print_info("Solver", args.solver)
    print_info("CFG scale", str(args.cfg))
    print_info("Duration", f"{args.duration:.2f}s")

    # --- Latent normalization stats -----------------------------------------
    stats_path = Path(args.stats) if args.stats else _REPO_ROOT / config.training.data_dir / "latents" / "latent_stats.npz"
    stats = _load_latent_stats(stats_path, device)
    if stats is not None:
        print_info("Latent denorm", f"enabled ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent denorm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    # --- Model --------------------------------------------------------------
    model = Echo().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    # --- Text -----------------------------------------------------------------
    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    text_ids = torch.tensor([tokenizer.tokenize(args.text)], dtype=torch.long, device=device)

    # --- Sample ---------------------------------------------------------------
    num_frames = math.ceil(args.duration * _SAMPLE_RATE / _HOP)
    print_info("Latent frames", str(num_frames))

    latent = _generate(model, text_ids, num_frames, args.steps, args.cfg, args.solver)  # (1, T, C)

    # --- Denormalize ----------------------------------------------------------
    # The model operates in channel-normalized latent space; BlueCodec expects
    # its native (raw) latent scale, so undo the z-scoring before decoding.
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean

    # --- Decode ---------------------------------------------------------------
    codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    with torch.no_grad():
        audio = codec.decode(latent.transpose(1, 2))                      # (B, C, T) -> audio
    audio = audio.squeeze(0).cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio[..., : int(args.duration * _SAMPLE_RATE)]

    torchaudio.save(args.output, audio, _SAMPLE_RATE)
    print_separator()
    print_info("Saved", args.output, Colors.OKGREEN)


if __name__ == "__main__":
    main()
