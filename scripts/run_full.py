#!/usr/bin/env python3
"""Full Echo pipeline: text -> prosody tokens -> audio.

Chains the two models. EchoAR writes a Mimi token grid from the phoneme string
alone, Mimi decodes it to a coarse 24 kHz waveform, BlueCodec re-encodes that
into a distil latent, and EchoFM flows from the distil latent to the data
distribution — exactly the ``distil -> data`` transport it was trained on, only
with the distil side synthesised instead of read from disk.

    text -> EchoAR -> Mimi.decode -> resample -> BlueCodec.encode
         -> EchoFM (ODE) -> BlueCodec.decode -> audio

The AR stage decides the duration (it stops when it emits EOS), so nothing about
the length needs to be supplied.

Usage:
    python scripts/run_full.py --ar-model checkpoints/echo_ar_best.pt \\
        --fm-model checkpoints/echo_final.pt --text "həlˈOʊ wˈɜːld" \\
        --steps 8 --cfg 3.0 --output out.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import contextmanager
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
import torchaudio
from bluecodec import BlueCodec
from transformers import MimiModel

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR
from echo.fm_model import EchoFM
from echo.tokenizer import Tokenizer

# Reuse the samplers rather than restate them: the flow-matching integrator and
# the Mimi decode helper are exactly the ones the single-model scripts run.
from run_ar import _decode_codes
from run_fm import _generate, _load_latent_stats

# Mimi: 24 kHz audio on a 12.5 Hz token grid. BlueCodec: 44.1 kHz, 512-sample hop.
_MIMI_SR = 24000
_MIMI_FPS = 12.5
_BLUE_SR = 44100
_BLUE_HOP = 512


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _sync(device: torch.device) -> None:
    """Drain the queue so wall-clock time reflects work actually finished."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@contextmanager
def _timed(name: str, device: torch.device, into: dict[str, float]):
    """Time a stage, synchronizing on both ends — CUDA/MPS ops are asynchronous,
    so without this the measurement would just record queue submission."""
    _sync(device)
    t0 = time.perf_counter()
    yield
    _sync(device)
    into[name] = time.perf_counter() - t0


def _load_checkpoint(model: torch.nn.Module, path: str, device: torch.device, what: str) -> None:
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get("model", ckpt)
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        hint = ""
        if any("theta" in k for k in list(state) + [n for n, _ in model.named_parameters()]):
            hint = ("\nHint: cross-attention rope_norm in models/config.json must match what the "
                    "checkpoint was trained with ('query' stores `theta`, 'absolute' stores "
                    "`theta_q`/`theta_k`).")
        raise SystemExit(f"Could not load the {what} checkpoint {path}:\n{e}{hint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Echo pipeline from text to audio.")
    parser.add_argument("--ar-model", type=str, required=True,
                        help="Path to an EchoAR checkpoint (.pt).")
    parser.add_argument("--fm-model", type=str, required=True,
                        help="Path to an EchoFM checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True,
                        help="Phoneme string to synthesize.")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on AR frames (default: 1000, i.e. 80s).")
    parser.add_argument("--steps", type=int, default=8,
                        help="Flow-matching integration steps (default: 8).")
    parser.add_argument("--solver", choices=("euler", "midpoint"), default="euler",
                        help="ODE integrator: 'euler' (1 model eval/step, default) or "
                             "'midpoint' (RK2, 2 evals/step, better at few steps).")
    parser.add_argument("--cfg", type=float, default=3.0,
                        help="Classifier-free guidance scale (default: 3.0; 1.0 disables guidance).")
    parser.add_argument(
        "--stats", type=str, default=None,
        help="Path to latent_stats.npz (mean/std) used to normalize/denormalize latents. "
             "Defaults to <data_dir>/latents/latent_stats.npz from the training config.",
    )
    parser.add_argument("--cut_last", "--cut-last", type=float, default=0.0, metavar="MS",
                        help="Trim this many milliseconds off the end of the output audio "
                             "(default: 0). The last frames often carry a codec edge artifact.")
    parser.add_argument("--save-intermediate", action="store_true",
                        help="Also write the AR/Mimi stage as <output stem>_ar.wav.")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio path.")
    args = parser.parse_args()

    if args.cut_last < 0:
        parser.error("--cut_last must be >= 0")

    device = _select_device()
    output_path = Path(args.output)

    # --- Latent normalization stats -----------------------------------------
    stats_path = (Path(args.stats) if args.stats
                  else _REPO_ROOT / config.training.fm.data_dir / "latents" / "latent_stats.npz")
    stats = _load_latent_stats(stats_path, device)

    # --- Models -------------------------------------------------------------
    ar_model = EchoAR().to(device)
    _load_checkpoint(ar_model, args.ar_model, device, "EchoAR")
    ar_model.eval()

    fm_model = EchoFM().to(device)
    _load_checkpoint(fm_model, args.fm_model, device, "EchoFM")
    fm_model.eval()

    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    text_ids = torch.tensor([tokenizer.tokenize(args.text)], dtype=torch.long, device=device)

    timings: dict[str, float] = {}

    print_header("Echo - Full Pipeline")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("AR checkpoint", args.ar_model)
    print_info("FM checkpoint", args.fm_model)
    print_info("Text tokens", str(text_ids.shape[1]))
    print_info("Steps", str(args.steps))
    print_info("Solver", args.solver)
    print_info("CFG scale", str(args.cfg))
    if stats is not None:
        print_info("Latent norm", f"enabled ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent norm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    # --- Stage 1: text -> prosody tokens ------------------------------------
    print_section("Stage 1 — EchoAR")
    with _timed("ar", device, timings):
        codes = ar_model.generate(text_ids, max_frames=args.max_frames)  # (1, T_ar, layers)
    frames = codes.shape[1]
    if frames == 0:
        print_info("Frames", "0 — the model emitted EOS immediately", Colors.FAIL)
        raise SystemExit(1)
    if frames >= args.max_frames:
        print_info("Frames", f"{frames} — hit --max-frames without emitting EOS", Colors.WARNING)
    else:
        print_info("Frames", str(frames), Colors.OKGREEN)

    duration = frames / _MIMI_FPS
    print_info("Duration", f"{duration:.2f}s")
    print_info("Time", f"{timings['ar']:.3f}s  ({1000 * timings['ar'] / frames:.1f} ms/frame)")

    # --- Stage 2: tokens -> coarse audio -> distil latent --------------------
    print_section("Stage 2 — Mimi decode -> BlueCodec encode")
    with _timed("codec", device, timings):
        with torch.no_grad():
            audio_mimi = _decode_codes(mimi, codes.transpose(1, 2))    # (1, T_audio) @ 24 kHz
            audio_blue = torchaudio.functional.resample(audio_mimi, _MIMI_SR, _BLUE_SR)
            distil = blue.encode(audio_blue)                           # (1, C, T_lat)

    if args.save_intermediate:
        inter = output_path.with_name(output_path.stem + "_ar" + output_path.suffix)
        torchaudio.save(str(inter), audio_mimi.float().cpu(), _MIMI_SR)
        print_info("Intermediate", str(inter), Colors.OKCYAN)

    distil = distil.transpose(1, 2).float()                            # (1, T_lat, C)
    print_info("Distil latent", f"{tuple(distil.shape)}  ({distil.shape[1] / (_BLUE_SR / _BLUE_HOP):.2f}s)")
    print_info("Time", f"{timings['codec']:.3f}s")

    if stats is not None:
        mean, std = stats
        distil = (distil - mean) / std

    # --- Stage 3: distil latent -> data latent -> audio ----------------------
    print_section("Stage 3 — EchoFM")
    with _timed("fm", device, timings):
        latent = _generate(fm_model, text_ids, distil, args.steps, args.cfg, args.solver)
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean
    print_info("Time", f"{timings['fm']:.3f}s  ({1000 * timings['fm'] / args.steps:.1f} ms/step)")

    with _timed("decode", device, timings):
        with torch.no_grad():
            audio = blue.decode(latent.transpose(1, 2))                # (1, C, T) -> audio
    audio = audio.squeeze(0).float().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio[..., : int(duration * _BLUE_SR)]

    # Optional tail trim, applied last so it acts on the finished waveform.
    cut_samples = int(round(args.cut_last * _BLUE_SR / 1000))
    if cut_samples > 0:
        if cut_samples >= audio.shape[-1]:
            raise SystemExit(
                f"--cut_last {args.cut_last:g} ms would remove the whole output "
                f"({1000 * audio.shape[-1] / _BLUE_SR:.0f} ms)"
            )
        audio = audio[..., :-cut_samples]

    saved_duration = audio.shape[-1] / _BLUE_SR

    torchaudio.save(str(output_path), audio, _BLUE_SR)

    # --- Timing summary -----------------------------------------------------
    print_section("Inference time")
    rows = [
        (f"EchoAR ({frames} frames)", timings["ar"]),
        ("Mimi + BlueCodec (codec -> latent)", timings["codec"]),
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
        print_info("Saved duration", f"{saved_duration:.2f}s  (--cut_last {args.cut_last:g} ms trimmed)",
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
