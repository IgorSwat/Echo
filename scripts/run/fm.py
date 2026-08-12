#!/usr/bin/env python3
"""Generate audio with a trained Echo flow-matching model.

Integrates `dx/dt = v(x, t)` from Gaussian noise at t=0 up to t=1 and decodes the
result with BlueCodec. Nothing seeds the transport: the layer-0 prosody tokens
are conditioning the model reads at every step, and they also set the output
length, since the latent grid runs at a fixed multiple of the token grid.

Usage:
    python scripts/run/fm.py --model checkpoints/echo_final.pt --text "hello world" \\
        --codec data/codecs/clone_0000.npz --steps 8 --cfg 3.0 --output out.wav

    python scripts/run/fm.py --model checkpoints/echo_final.pt \\
        --test-suite data/norbi/phonemes_test.csv --steps 8 --cfg 3.0
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch
import torchaudio
from bluecodec import BlueCodec

from __common__ import (
    BLUE_HOP,
    BLUE_SR,
    REPO_ROOT,
    Stats,
    load_checkpoint,
    load_latent_stats,
    load_pairs_csv,
    load_tokenizer,
    select_device,
)
from __phonemize__ import add_phonemize_args, phonemize_args
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.fm_model import EchoFM


def _load_prosody(path: Path, device: torch.device) -> torch.Tensor:
    """
    Layer 0 of a codec file as a `(1, K)` batch -- the stream EchoFM renders.
    """

    codes = np.load(path)["codes"]                                   # (num_layers, K)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]

    return torch.from_numpy(codes[0].astype(np.int64))[None].to(device)


@torch.no_grad()
def _generate(
    model: EchoFM,
    codec: BlueCodec,
    text_ids: torch.Tensor,
    prosody: torch.Tensor,
    duration: float,
    stats: Stats | None,
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Integrate from noise, denormalize, decode, and write the waveform."""
    latent = model.sample(text_ids, prosody, args.steps, args.cfg, args.solver)
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean

    audio = codec.decode(latent.transpose(1, 2)).squeeze(0).cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(output_path), audio[..., : int(duration * BLUE_SR)], BLUE_SR)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with Echo.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a training checkpoint (.pt).")
    parser.add_argument("--text", type=str, default=None,
                        help="Raw text to synthesize (phonemized with eSpeak).")
    parser.add_argument("--codec", type=str, default=None,
                        help="Path to the matching codec .npz; its layer 0 is the prosody "
                             "conditioning. In --test-suite mode these are read from "
                             "<suite dir>/codecs/ automatically.")
    parser.add_argument("--test-suite", type=str, default=None,
                        help="Path to a phonemes CSV file for batch generation.")
    parser.add_argument("--steps", type=int, default=8,
                        help="Flow-matching integration steps (default: 8).")
    parser.add_argument("--solver", choices=EchoFM.SOLVERS, default="euler",
                        help="ODE integrator: 'euler' (1 model eval/step, default) or "
                             "'midpoint' (RK2, 2 evals/step, better at few steps).")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0; 1.0 disables it).")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to latent_stats.npz. Defaults to "
                             "<data_dir>/latents/latent_stats.npz from the training config.")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio path (single-sample mode).")
    add_phonemize_args(parser)
    args = parser.parse_args()

    if args.test_suite is None and (args.text is None or args.codec is None):
        parser.error("--text and --codec are required when --test-suite is not provided")

    return args


def main() -> None:
    args = _parse_args()
    device = select_device()

    stats_path = (Path(args.stats) if args.stats
                  else REPO_ROOT / config.training.fm.data_dir / "latents" / "latent_stats.npz")
    stats = load_latent_stats(stats_path, device)

    model = EchoFM().to(device)
    load_checkpoint(model, args.model, device, "EchoFM")
    model.eval()
    tokenizer = load_tokenizer()
    codec = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    # Both modes reduce to a list of (codec, phonemes, output path).
    if args.test_suite is not None:
        suite = Path(args.test_suite)
        jobs = [
            (suite.parent / "codecs" / name, phonemes,
             suite.parent / "test_outputs" / f"{Path(name).stem}.wav")
            for name, phonemes in load_pairs_csv(suite)
        ]
        title, source = "Echo - Test Suite", f"{suite} ({len(jobs)} entries)"
    else:
        phonemes = phonemize_args(args.text, args)
        jobs = [(Path(args.codec), phonemes, Path(args.output))]
        title, source = "Echo - Generation", str(args.codec)

    print_header(title)
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Checkpoint", args.model)
    print_info("Source", source)
    print_info("Sampler", f"{args.steps} {args.solver} steps, cfg {args.cfg:g}")
    if args.test_suite is None:
        print_info("Language", args.language)
        if args.print_phonemes:
            print_info("Phonemes", jobs[0][1], Colors.OKCYAN)
    if config.latent_norm == "instance":
        raise SystemExit(
            "latent_norm='instance' took its statistics from the distil, and the "
            "transport no longer has one. Set latent_norm to 'dataset' in "
            "models/config.json and train against latent_stats.npz."
        )
    if stats is not None:
        print_info("Latent norm", f"per dataset ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent norm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    print_section("Generating")
    for i, (codec_path, phonemes, output_path) in enumerate(jobs, start=1):
        prosody = _load_prosody(codec_path, device)
        # The token count sets the length, so it also sets the duration.
        duration = model.latent_frames(prosody.shape[1]) * BLUE_HOP / BLUE_SR
        text_ids = torch.tensor(
            [tokenizer.tokenize(phonemes)], dtype=torch.long, device=device
        )
        _generate(model, codec, text_ids, prosody, duration, stats, args, output_path)
        print_info(f"[{i}/{len(jobs)}]",
                   f"{codec_path.name} -> {output_path.name}  ({duration:.2f}s)",
                   Colors.OKGREEN)

    print_separator()
    print_info("Saved", str(jobs[0][2]) if len(jobs) == 1 else f"{len(jobs)} files",
               Colors.OKGREEN)


if __name__ == "__main__":
    main()
