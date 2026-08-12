#!/usr/bin/env python3
"""Full Echo pipeline: text -> prosody tokens -> audio.

    text -> EchoAR -> layer-0 tokens -> EchoFM (ODE from noise) -> BlueCodec.decode

EchoFM starts from Gaussian noise and reads the AR stage's layer-0 tokens as
conditioning, which is the transport it was trained on. The AR stage decides the
duration by emitting EOS, and the token count then fixes the latent length, so
nothing needs to be supplied.

Usage:
    python scripts/run/full.py --ar-model checkpoints/echo_ar_best.pt \\
        --fm-model checkpoints/echo_final.pt --text "hello world" \\
        --steps 8 --cfg 3.0 --output out.wav

    python scripts/run/full.py --ar-model checkpoints/echo_ar_best.pt \\
        --fm-model checkpoints/echo_final.pt --text "hello world" \\
        --temperature 1.0 --output out.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import warnings
from contextlib import contextmanager

import torch
import torchaudio
from bluecodec import BlueCodec
from transformers import MimiModel

from __common__ import (
    BLUE_HOP,
    BLUE_SR,
    MIMI_FPS,
    MIMI_SR,
    REPO_ROOT,
    decode_mimi,
    load_checkpoint,
    load_latent_stats,
    load_tokenizer,
    lowpass,
    select_device,
    sync_device,
)
from __phonemize__ import add_phonemize_args, phonemize_args
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR
from echo.fm_model import EchoFM


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
    parser = argparse.ArgumentParser(description="Run the full Echo pipeline from text to audio.")
    parser.add_argument("--ar-model", type=str, required=True,
                        help="Path to an EchoAR checkpoint (.pt).")
    parser.add_argument("--fm-model", type=str, required=True,
                        help="Path to an EchoFM checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True,
                        help="Raw text to synthesize (phonemized with eSpeak).")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on AR frames (default: 1000, i.e. 80s).")
    parser.add_argument("--temperature", type=float, default=0.0, metavar="T",
                        help="AR sampling temperature (default: 0.0 = greedy). Around 1.0 the "
                             "emitted token statistics track the training data much more "
                             "closely than greedy does; greedy is deterministic and blander.")
    parser.add_argument("--top-k", type=int, default=0, metavar="K",
                        help="Restrict each AR draw to the K most likely ids (0 = off). "
                             "Only applies when --temperature > 0.")
    parser.add_argument("--steps", type=int, default=8,
                        help="Flow-matching integration steps (default: 8).")
    parser.add_argument("--solver", choices=EchoFM.SOLVERS, default="euler",
                        help="ODE integrator: 'euler' (1 model eval/step, default) or "
                             "'midpoint' (RK2, 2 evals/step, better at few steps).")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Seed the noise the transport starts from, making a rendering "
                             "reproducible. Omitted, each run differs.")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0; 1.0 disables it).")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to latent_stats.npz (mean/std) used to normalize and "
                             "denormalize latents. Defaults to "
                             "<data_dir>/latents/latent_stats.npz from the training config.")
    parser.add_argument("--cut_last", "--cut-last", type=float, default=0.0, metavar="MS",
                        help="Trim this many milliseconds off the end of the output audio "
                             "(default: 0). The last frames often carry a codec edge artifact.")
    parser.add_argument("--lowpass", type=float, default=0.0, metavar="HZ",
                        help="Low-pass the output at this frequency (default: 0 = off). The "
                             "sample rate is unchanged — this only removes content above HZ. "
                             "For a 24 kHz source corpus try 11500: nothing above 12 kHz is "
                             "real, and BlueCodec (a 44.1 kHz model) fabricates noise there.")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Also write the AR/Mimi stage as <output stem>_ar.wav.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path.")
    add_phonemize_args(parser)
    args = parser.parse_args()

    if args.cut_last < 0:
        parser.error("--cut_last must be >= 0")
    if args.temperature < 0:
        parser.error("--temperature must be >= 0 (0 selects greedy decoding)")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 disables top-k)")
    if args.lowpass < 0:
        parser.error("--lowpass must be >= 0 (0 disables the filter)")
    if args.lowpass >= BLUE_SR / 2:
        parser.error(f"--lowpass must be below the {BLUE_SR // 2} Hz Nyquist frequency, "
                     f"got {args.lowpass:g}")

    return args


def main() -> None:
    args = _parse_args()

    # BlueCodec's STFT reuses an `out` tensor that torch resizes on the first
    # call; the deprecation notice is internal to the codec and says nothing
    # about this script's inputs.
    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    device = select_device()
    output_path = Path(args.output)

    stats_path = (Path(args.stats) if args.stats
                  else REPO_ROOT / config.training.fm.data_dir / "latents" / "latent_stats.npz")
    stats = load_latent_stats(stats_path, device)

    # --- Models -------------------------------------------------------------
    ar_model = EchoAR().to(device)
    load_checkpoint(ar_model, args.ar_model, device, "EchoAR")
    ar_model.eval()

    fm_model = EchoFM().to(device)
    load_checkpoint(fm_model, args.fm_model, device, "EchoFM")
    fm_model.eval()

    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    tokenizer = load_tokenizer()
    phonemes = phonemize_args(args.text, args)
    text_ids = torch.tensor([tokenizer.tokenize(phonemes)], dtype=torch.long, device=device)

    timings: dict[str, float] = {}

    print_header("Echo - Full Pipeline")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("AR checkpoint", args.ar_model)
    print_info("FM checkpoint", args.fm_model)
    print_info("Language", args.language)
    if args.print_phonemes:
        print_info("Phonemes", phonemes, Colors.OKCYAN)
    print_info("Text tokens", str(text_ids.shape[1]))
    if args.temperature > 0:
        print_info("AR decoding", f"sampling (temperature {args.temperature:g}"
                                  + (f", top-k {args.top_k}" if args.top_k > 0 else "") + ")",
                   Colors.OKCYAN)
    else:
        print_info("AR decoding", "greedy (argmax)")
    print_info("Steps", str(args.steps))
    print_info("Solver", args.solver)
    print_info("CFG scale", str(args.cfg))
    print_info("Source", "Gaussian noise"
               + (f", seed {args.seed}" if args.seed is not None
                  else ", unseeded (varies per run)"))
    print_info("Low-pass", f"{args.lowpass:g} Hz" if args.lowpass > 0 else "off")
    if config.latent_norm == "instance":
        raise SystemExit(
            "latent_norm='instance' took its statistics from the distil, and the "
            "pipeline no longer builds one. Set latent_norm to 'dataset' in "
            "models/config.json and train against latent_stats.npz."
        )
    if stats is not None:
        print_info("Latent norm", f"per dataset ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent norm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    # --- Stage 1: text -> prosody tokens ------------------------------------
    print_section("Stage 1 — EchoAR")
    with _timed("ar", device, timings):
        codes = ar_model.generate(text_ids, max_frames=args.max_frames,
                                  temperature=args.temperature,
                                  top_k=args.top_k)                  # (1, T_ar, layers)
    frames = codes.shape[1]
    if frames == 0:
        print_info("Frames", "0 — the model emitted EOS immediately", Colors.FAIL)
        raise SystemExit(1)
    if frames >= args.max_frames:
        print_info("Frames", f"{frames} — hit --max-frames without emitting EOS", Colors.WARNING)
    else:
        print_info("Frames", str(frames), Colors.OKGREEN)

    duration = frames / MIMI_FPS
    print_info("Duration", f"{duration:.2f}s")
    print_info("Time", f"{timings['ar']:.3f}s  ({1000 * timings['ar'] / frames:.1f} ms/frame)")

    # --- Stage 2: prosody tokens -> data latent -----------------------------
    # The Mimi decode / BlueCodec re-encode that used to sit here is gone: it
    # existed only to build a distil for the transport to start from, and the
    # transport now starts from noise with the tokens as conditioning. That also
    # takes a lossy 2-codebook audio round trip out of the pipeline.
    print_section("Stage 2 — EchoFM")
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    prosody = codes                                                  # (1, T_ar)
    with _timed("fm", device, timings):
        latent = fm_model.sample(text_ids, prosody, args.steps, args.cfg,
                                 args.solver, generator=generator)
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean

    if args.save_intermediate:
        with torch.no_grad():
            # Mimi wants (B, layers, T); the AR model produces the semantic layer only.
            audio_mimi = decode_mimi(mimi, codes[:, None, :])        # (1, T) @ 24 kHz
        inter = output_path.with_name(output_path.stem + "_ar" + output_path.suffix)
        torchaudio.save(str(inter), audio_mimi.float().cpu(), MIMI_SR)
        print_info("Intermediate", str(inter), Colors.OKCYAN)

    print_info("Latent", f"{tuple(latent.shape)}  "
                         f"({latent.shape[1] / (BLUE_SR / BLUE_HOP):.2f}s)")
    print_info("Time", f"{timings['fm']:.3f}s  ({1000 * timings['fm'] / args.steps:.1f} ms/step)")

    with _timed("decode", device, timings):
        with torch.no_grad():
            audio = blue.decode(latent.transpose(1, 2))              # (1, C, T) -> audio
    audio = audio.squeeze(0).float().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio[..., : int(duration * BLUE_SR)]

    # Optional tail trim, applied last so it acts on the finished waveform.
    cut_samples = int(round(args.cut_last * BLUE_SR / 1000))
    if cut_samples > 0:
        if cut_samples >= audio.shape[-1]:
            raise SystemExit(
                f"--cut_last {args.cut_last:g} ms would remove the whole output "
                f"({1000 * audio.shape[-1] / BLUE_SR:.0f} ms)"
            )
        audio = audio[..., :-cut_samples]

    # Band-limit last, on the finished waveform: the trims above are the only
    # steps that care about sample positions, and this one preserves them.
    if args.lowpass > 0:
        before = float(audio.pow(2).sum())
        audio = lowpass(audio, args.lowpass, BLUE_SR)
        removed = 1.0 - float(audio.pow(2).sum()) / max(before, 1e-12)
        print_info("Low-pass", f"{args.lowpass:g} Hz  ({100 * removed:.2f}% of energy removed, "
                               f"sample rate unchanged)", Colors.OKCYAN)

    saved_duration = audio.shape[-1] / BLUE_SR
    torchaudio.save(str(output_path), audio, BLUE_SR)

    # --- Timing summary -----------------------------------------------------
    print_section("Inference time")
    rows = [
        (f"EchoAR ({frames} frames)", timings["ar"]),
        (f"EchoFM ({args.steps} {args.solver} steps)", timings["fm"]),
        ("BlueCodec decode (latent -> audio)", timings["decode"]),
    ]
    total = sum(t for _, t in rows)
    for label, t in rows:
        print(f"  {Colors.BOLD}{label:<36}{Colors.ENDC} {t:8.3f}s  {100 * t / total:5.1f}%")
    print("  " + "─" * 52)
    print_info("Total", f"{total:.3f}s", Colors.OKCYAN)
    print_info("Audio duration", f"{duration:.2f}s generated")
    if cut_samples > 0:
        print_info("Saved duration",
                   f"{saved_duration:.2f}s  (--cut_last {args.cut_last:g} ms trimmed)",
                   Colors.OKCYAN)
    # RTF stays against what the pipeline synthesized, so a cosmetic tail trim
    # does not make the model look slower than it is.
    print_info("Real-time factor", f"{total / duration:.2f}x  "
                                   f"({duration / total:.2f}x realtime; lower RTF is faster)",
               Colors.OKGREEN if total < duration else Colors.WARNING)

    print_separator()
    print_info("Saved", str(output_path), Colors.OKGREEN)


if __name__ == "__main__":
    main()
