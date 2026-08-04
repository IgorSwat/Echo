#!/usr/bin/env python3
"""Decode Mimi codec tokens to audio through the trained EchoShortcut model.

Pipeline: 2-layer Mimi codec tokens -> EchoShortcut -> Blue latents -> BlueCodec
decode -> waveform. This is the cheap replacement for the reference path
(Mimi decode -> audio -> BlueCodec encode -> BlueCodec decode), so comparing the
two outputs on the same input is what tells you how much the shortcut costs.

Usage:
    # Single sample
    python scripts/run_shortcut.py --model checkpoints/echo_shortcut_final.pt \\
        --codec data/norbi/codecs/clone_0000.npz --output out.wav

    # Full test suite (reads <csv dir>/codecs/<npz> for every CSV entry)
    python scripts/run_shortcut.py --model checkpoints/echo_shortcut_final.pt \\
        --test-suite data/norbi/phonemes_test.csv
"""

from __future__ import annotations

import argparse
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
from echo.shortcut_model import EchoShortcut

# BlueCodec operates at 44.1 kHz with a hop of 512 samples per latent frame.
_SAMPLE_RATE = 44100
_HOP = 512

# Mimi's token grid, used only to report the input duration.
_MIMI_FPS = 12.5


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_latent_stats(
    stats_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Load per-channel mean/std (shape ``(latent_dim,)``) for denormalization."""
    if not stats_path.is_file():
        return None
    stats = np.load(stats_path)
    mean = torch.from_numpy(stats["mean"].astype(np.float32)).to(device)   # (C,)
    std = torch.from_numpy(stats["std"].astype(np.float32)).to(device)     # (C,)
    return mean, std


def _load_codec(codec_path: Path, device: torch.device) -> torch.Tensor:
    """Load a ``(num_layers, T)`` codec file as a ``(1, T, NUM_TOKEN_LAYERS)`` tensor."""
    codes = np.load(codec_path)["codes"]                             # (num_layers, T)
    # Tolerate a leading singleton (batch) dim, as the dataset loader does.
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.ndim != 2:
        raise ValueError(
            f"{codec_path.name}: expected codec of shape (num_layers, T), got {codes.shape}"
        )
    layers = EchoShortcut.NUM_TOKEN_LAYERS
    if codes.shape[0] < layers:
        raise ValueError(
            f"{codec_path.name}: codec has {codes.shape[0]} layers, but the model "
            f"needs {layers}"
        )

    codes = codes[:layers]                                           # (layers, T)

    return torch.from_numpy(codes.T.copy()).long().unsqueeze(0).to(device)


@torch.no_grad()
def _generate_one(
    model: EchoShortcut,
    codec: BlueCodec,
    codes: torch.Tensor,                                             # (1, T, layers)
    stats: tuple[torch.Tensor, torch.Tensor] | None,
    output_path: Path,
) -> int:
    """Run the shortcut and write the decoded waveform. Returns the latent length."""
    latent = model(codes)                                            # (1, L, C)

    # The model is trained on channel-normalized targets, so the prediction has
    # to be put back on BlueCodec's own scale before decoding.
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean

    audio = codec.decode(latent.transpose(1, 2))                     # (1, L, C) -> audio
    audio = audio.squeeze(0).float().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio, _SAMPLE_RATE)

    return latent.shape[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode Mimi codec tokens to audio with EchoShortcut."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a training checkpoint (.pt).")
    parser.add_argument("--codec", type=str, default=None,
                        help="Path to a .npz codec file (key 'codes', shape (layers, T)).")
    parser.add_argument("--test-suite", type=str, default=None,
                        help="Path to a phonemes CSV file for batch generation; codec files "
                             "are read from <csv dir>/codecs/.")
    parser.add_argument(
        "--stats", type=str, default=None,
        help="Path to latent_stats.npz (mean/std) used to denormalize the prediction. "
             "Defaults to <data_dir>/latents/latent_stats.npz from the training config.",
    )
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio path (single-sample mode).")
    args = parser.parse_args()

    if args.test_suite is None and args.codec is None:
        parser.error("--codec is required when --test-suite is not provided")

    device = _select_device()

    # --- Latent normalization stats -----------------------------------------
    stats_path = (
        Path(args.stats) if args.stats
        else _REPO_ROOT / config.training.shortcut.data_dir / "latents" / "latent_stats.npz"
    )
    stats = _load_latent_stats(stats_path, device)

    # --- Model --------------------------------------------------------------
    model = EchoShortcut().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    # --- Codec (loaded once) ------------------------------------------------
    codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    def _print_common_setup() -> None:
        print_info("Device", str(device), Colors.OKCYAN)
        print_info("Checkpoint", args.model)
        print_info("Codec layers", str(EchoShortcut.NUM_TOKEN_LAYERS))
        if stats is not None:
            print_info("Latent norm", f"enabled ({stats_path})", Colors.OKCYAN)
        else:
            print_info("Latent norm", f"disabled (stats not found: {stats_path})",
                       Colors.WARNING)

    if args.test_suite is not None:
        # --- Test suite mode ------------------------------------------------
        test_csv = Path(args.test_suite)
        codecs_dir = test_csv.parent / "codecs"
        names: list[str] = []
        with open(test_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                names.append(line.split("|", 1)[0].strip())

        print_header("Echo - Shortcut Test Suite")
        print_separator()
        print_section("Setup")
        _print_common_setup()
        print_info("Test suite", str(test_csv))
        print_info("Entries", str(len(names)))

        for i, npz_name in enumerate(names):
            codes = _load_codec(codecs_dir / npz_name, device)
            output_path = test_csv.parent / "test_outputs" / (Path(npz_name).stem + "_shortcut.wav")
            frames = _generate_one(model, codec, codes, stats, output_path)

            print_info(
                f"[{i + 1}/{len(names)}]",
                f"{npz_name}: {codes.shape[1]} -> {frames} frames "
                f"({frames * _HOP / _SAMPLE_RATE:.2f}s) -> {output_path.name}",
                Colors.OKGREEN,
            )

        print_separator()
        print_info("Done", f"{len(names)} files saved", Colors.OKGREEN)
    else:
        # --- Single-sample mode ---------------------------------------------
        codec_path = Path(args.codec)
        codes = _load_codec(codec_path, device)

        print_header("Echo - Shortcut Generation")
        print_separator()
        print_section("Setup")
        _print_common_setup()
        print_info("Codec", str(codec_path))
        print_info("Codec frames", f"{codes.shape[1]} ({codes.shape[1] / _MIMI_FPS:.2f}s)")

        output_path = Path(args.output)
        frames = _generate_one(model, codec, codes, stats, output_path)

        print_section("Output")
        print_info("Latent frames", f"{frames} ({frames * _HOP / _SAMPLE_RATE:.2f}s)")
        print_separator()
        print_info("Saved", str(output_path), Colors.OKGREEN)


if __name__ == "__main__":
    main()
