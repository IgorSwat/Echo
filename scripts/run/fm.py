#!/usr/bin/env python3
"""Generate audio with a trained Echo flow-matching model.

Integrates `dx/dt = v(x, t)` from the dithered distil latent at t=0 up to t=1 —
the transport the model was trained on — and decodes the result with BlueCodec.

Usage:
    python scripts/run/fm.py --model checkpoints/echo_final.pt --text "hello world" \\
        --distil data/distils/clone_0000.npz --steps 8 --cfg 3.0 --output out.wav

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
    norm_stats,
    select_device,
)
from __phonemize__ import add_phonemize_args, phonemize_args
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.fm_model import EchoFM


def _load_distil(
    path: Path, device: torch.device, stats: Stats | None
) -> tuple[torch.Tensor, float, Stats | None]:
    """
    A normalized distil latent, its duration, and the stats that normalized it.
    """

    arr = np.load(path)["latents"]                                   # (C, T)
    distil = torch.from_numpy(arr.T.copy()).float().to(device)       # (T, C)
    duration = distil.shape[0] * BLUE_HOP / BLUE_SR

    used = norm_stats(distil, stats)
    if used is not None:
        mean, std = used
        distil = (distil - mean) / std

    return distil, duration, used


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
    distil: torch.Tensor,
    prosody: torch.Tensor,
    duration: float,
    stats: Stats | None,                          # the ones that normalized `distil`
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Integrate, denormalize, decode, and write the waveform."""
    latent = model.sample(text_ids, distil.unsqueeze(0), prosody,
                          args.steps, args.cfg, args.solver)
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
    parser.add_argument("--distil", type=str, default=None,
                        help="Path to a .npz file with the starting distil latent (C, T).")
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

    if args.test_suite is None and (args.text is None or args.distil is None
                                    or args.codec is None):
        parser.error("--text, --distil and --codec are required when --test-suite "
                     "is not provided")

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

    # Both modes reduce to a list of (distil, codec, phonemes, output path). The
    # distil and the codec share a basename: they describe the same utterance.
    if args.test_suite is not None:
        suite = Path(args.test_suite)
        jobs = [
            (suite.parent / "distils" / name, suite.parent / "codecs" / name, phonemes,
             suite.parent / "test_outputs" / f"{Path(name).stem}.wav")
            for name, phonemes in load_pairs_csv(suite)
        ]
        title, source = "Echo - Test Suite", f"{suite} ({len(jobs)} entries)"
    else:
        phonemes = phonemize_args(args.text, args)
        jobs = [(Path(args.distil), Path(args.codec), phonemes, Path(args.output))]
        title, source = "Echo - Generation", str(args.distil)

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
            print_info("Phonemes", jobs[0][2], Colors.OKCYAN)
    if config.latent_norm == "instance":
        print_info("Latent norm", "per instance (from each distil's own channel stats)",
                   Colors.OKCYAN)
    elif stats is not None:
        print_info("Latent norm", f"per dataset ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent norm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    print_section("Generating")
    for i, (distil_path, codec_path, phonemes, output_path) in enumerate(jobs, start=1):
        distil, duration, used = _load_distil(distil_path, device, stats)
        prosody = _load_prosody(codec_path, device)
        text_ids = torch.tensor(
            [tokenizer.tokenize(phonemes)], dtype=torch.long, device=device
        )
        _generate(model, codec, text_ids, distil, prosody, duration, used,
                  args, output_path)
        print_info(f"[{i}/{len(jobs)}]",
                   f"{distil_path.name} -> {output_path.name}  ({duration:.2f}s)",
                   Colors.OKGREEN)

    print_separator()
    print_info("Saved", str(jobs[0][3]) if len(jobs) == 1 else f"{len(jobs)} files",
               Colors.OKGREEN)


if __name__ == "__main__":
    main()
