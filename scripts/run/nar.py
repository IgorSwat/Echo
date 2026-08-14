#!/usr/bin/env python3
"""Text -> audio through the two token stages: EchoAR, then EchoNAR, then Mimi.

    text -> EchoAR -> layer 0 -> EchoNAR -> layers 1..K-1 -> Mimi.decode

The AR stage writes Mimi's semantic layer one frame at a time and decides the
duration by emitting EOS; the NAR stage fills the acoustic layers above it in
``K - 1`` forward passes, whatever the length. This is the VALL-E split, and it
is an alternative to `run/full.py`, which hands the same layer-0 tokens to the
flow-matching stage and BlueCodec instead.

Pass ``--codec`` to read layer 0 from a precomputed .npz instead of running the
AR stage. That measures the NAR stage on its own — the ceiling the AR stage's
tokens are being judged against — and needs no AR checkpoint.

Usage:
    python scripts/run/nar.py --ar-model checkpoints/ljspeech/echo_ar_best.pt \\
        --nar-model checkpoints/ljspeech/echo_nar_best.pt \\
        --text "hello world" --output out.wav

    python scripts/run/nar.py --nar-model checkpoints/ljspeech/echo_nar_best.pt \\
        --codec data/ljspeech/codecs/LJ001-0001.npz \\
        --text "printing in the only sense" --output out.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
from contextlib import contextmanager

import numpy as np
import torch
import torchaudio
from transformers import MimiModel

from __common__ import (
    MIMI_FPS,
    MIMI_SR,
    decode_mimi,
    load_checkpoint,
    load_tokenizer,
    select_device,
    sync_device,
)
from __phonemize__ import add_phonemize_args, phonemize_args
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo.ar_model import EchoAR
from echo.nar_model import EchoNAR


@contextmanager
def _timed(name: str, device: torch.device, into: dict[str, float]):
    """Time a stage, synchronizing on both ends — CUDA/MPS ops are asynchronous,
    so without this the measurement would only record queue submission."""
    sync_device(device)
    t0 = time.perf_counter()
    yield
    sync_device(device)
    into[name] = time.perf_counter() - t0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AR + NAR token pipeline from text to audio."
    )
    parser.add_argument("--nar-model", type=str, required=True,
                        help="Path to an EchoNAR checkpoint (.pt).")
    parser.add_argument("--ar-model", type=str, default=None,
                        help="Path to an EchoAR checkpoint (.pt). Required unless "
                             "--codec supplies layer 0 instead.")
    parser.add_argument("--codec", type=str, default=None,
                        help="Read layer 0 from this precomputed codec .npz rather than "
                             "generating it, isolating the NAR stage.")
    parser.add_argument("--text", type=str, required=True,
                        help="Raw text to synthesize (phonemized with eSpeak). Still needed "
                             "with --codec: the NAR stage is text-conditioned too.")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on AR frames (default: 1000, i.e. 80s).")
    parser.add_argument("--temperature", type=float, default=0.0, metavar="T",
                        help="AR sampling temperature (default: 0.0 = greedy).")
    parser.add_argument("--top-k", type=int, default=0, metavar="K",
                        help="Restrict each AR draw to the K most likely ids (0 = off).")
    parser.add_argument("--nar-temperature", type=float, default=0.0, metavar="T",
                        help="NAR sampling temperature (default: 0.0 = greedy, which is "
                             "what VALL-E's NAR stage uses: the acoustic residual is nearly "
                             "determined by the layers below it).")
    parser.add_argument("--nar-top-k", type=int, default=0, metavar="K",
                        help="Restrict each NAR draw to the K most likely ids (0 = off).")
    parser.add_argument("--layers", type=int, default=None, metavar="N",
                        help=f"Decode with the first N codec layers "
                             f"(default: all {EchoNAR.NUM_TOKEN_LAYERS}). Useful for hearing "
                             f"what each layer adds.")
    parser.add_argument("--save-semantic", action="store_true",
                        help="Also write the layer-0-only decode as <output stem>_l0.wav, "
                             "which is the AR stage's audio before this stage touched it.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path.")
    add_phonemize_args(parser)
    args = parser.parse_args()

    if args.ar_model is None and args.codec is None:
        parser.error("pass --ar-model, or --codec to supply layer 0 from a file")
    if args.temperature < 0 or args.nar_temperature < 0:
        parser.error("temperatures must be >= 0 (0 selects greedy decoding)")
    if args.top_k < 0 or args.nar_top_k < 0:
        parser.error("top-k must be >= 0 (0 disables top-k)")
    if args.layers is not None and not 2 <= args.layers <= EchoNAR.NUM_TOKEN_LAYERS:
        parser.error(f"--layers must be in [2, {EchoNAR.NUM_TOKEN_LAYERS}]")

    return args


def _layer0_from_file(path: Path, device: torch.device) -> torch.Tensor:
    """Layer 0 of a precomputed codec file, as a `(1, T)` tensor."""
    codes = np.load(path)["codes"]                                   # (layers, T)
    if codes.ndim == 3 and codes.shape[0] == 1:
        codes = codes[0]

    return torch.from_numpy(codes[0].astype(np.int64))[None].to(device)


def main() -> None:
    args = _parse_args()

    device = select_device()
    output_path = Path(args.output)
    target_layers = args.layers or EchoNAR.NUM_TOKEN_LAYERS

    tokenizer = load_tokenizer()
    phonemes = phonemize_args(args.text, args)
    text_ids = torch.tensor([tokenizer.tokenize(phonemes)], dtype=torch.long, device=device)

    timings: dict[str, float] = {}

    print_header("Echo - AR + NAR Pipeline")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("NAR checkpoint", args.nar_model)
    print_info("Layer 0", args.codec if args.codec else f"EchoAR ({args.ar_model})")
    print_info("Language", args.language)
    if args.print_phonemes:
        print_info("Phonemes", phonemes, Colors.OKCYAN)
    print_info("Text tokens", str(text_ids.shape[1]))
    print_info("Codec layers", f"{target_layers} of {EchoNAR.NUM_TOKEN_LAYERS}")
    if args.nar_temperature > 0:
        print_info("NAR decoding", f"sampling (temperature {args.nar_temperature:g}"
                                   + (f", top-k {args.nar_top_k}" if args.nar_top_k > 0 else "")
                                   + ")", Colors.OKCYAN)
    else:
        print_info("NAR decoding", "greedy (argmax)")

    # --- Stage 1: text -> layer 0 -------------------------------------------
    print_section("Stage 1 — layer 0")
    if args.codec:
        codes0 = _layer0_from_file(Path(args.codec), device)         # (1, T)
        timings["ar"] = 0.0
        print_info("Source", f"ground truth, {codes0.shape[1]} frames", Colors.OKCYAN)
    else:
        ar_model = EchoAR().to(device)
        load_checkpoint(ar_model, args.ar_model, device, "EchoAR")
        ar_model.eval()

        if args.temperature > 0:
            print_info("AR decoding", f"sampling (temperature {args.temperature:g}"
                                      + (f", top-k {args.top_k}" if args.top_k > 0 else "") + ")",
                       Colors.OKCYAN)
        else:
            print_info("AR decoding", "greedy (argmax)")

        with _timed("ar", device, timings):
            codes0 = ar_model.generate(text_ids, max_frames=args.max_frames,
                                       temperature=args.temperature,
                                       top_k=args.top_k)             # (1, T)

        frames = codes0.shape[1]
        if frames == 0:
            print_info("Frames", "0 — the model emitted EOS immediately", Colors.FAIL)
            raise SystemExit(1)
        if frames >= args.max_frames:
            print_info("Frames", f"{frames} — hit --max-frames without emitting EOS",
                       Colors.WARNING)
        else:
            print_info("Frames", str(frames), Colors.OKGREEN)
        print_info("Time", f"{timings['ar']:.3f}s  "
                           f"({1000 * timings['ar'] / frames:.1f} ms/frame)")

    frames = codes0.shape[1]
    print_info("Duration", f"{frames / MIMI_FPS:.2f}s")

    # --- Stage 2: layer 0 -> the acoustic layers ----------------------------
    print_section("Stage 2 — EchoNAR")
    nar_model = EchoNAR().to(device)
    load_checkpoint(nar_model, args.nar_model, device, "EchoNAR")
    nar_model.eval()

    with _timed("nar", device, timings):
        codes = nar_model.generate(
            codes0, text_ids,
            temperature=args.nar_temperature, top_k=args.nar_top_k,
            max_layer=target_layers,
        )                                                            # (1, T, target_layers)

    passes = target_layers - 1
    print_info("Passes", f"{passes} (one per written layer)")
    print_info("Time", f"{timings['nar']:.3f}s  ({timings['nar'] / passes:.3f}s per layer)")

    # --- Mimi decode --------------------------------------------------------
    print_section("Decoding")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    with _timed("mimi", device, timings), torch.no_grad():
        audio = decode_mimi(mimi, codes.transpose(1, 2))             # (1, T_audio)
    torchaudio.save(str(output_path), audio.float().cpu(), MIMI_SR)
    print_info("Time", f"{timings['mimi']:.3f}s")

    if args.save_semantic:
        with torch.no_grad():
            audio0 = decode_mimi(mimi, codes0[:, None, :])           # (1, T_audio)
        semantic_path = output_path.with_name(output_path.stem + "_l0" + output_path.suffix)
        torchaudio.save(str(semantic_path), audio0.float().cpu(), MIMI_SR)
        print_info("Layer 0 only", str(semantic_path), Colors.OKCYAN)

    # --- Summary ------------------------------------------------------------
    total = sum(timings.values())
    audio_seconds = audio.shape[-1] / MIMI_SR

    print_separator()
    print_section("Timing")
    for name in ("ar", "nar", "mimi"):
        share = 100 * timings[name] / total if total else 0.0
        print_info(name.upper(), f"{timings[name]:.3f}s  ({share:.0f}%)")
    print_info("Total", f"{total:.3f}s for {audio_seconds:.2f}s of audio "
                        f"({audio_seconds / total:.2f}x real time)", Colors.OKCYAN)
    print_info("Saved", str(output_path), Colors.OKGREEN)


if __name__ == "__main__":
    main()
