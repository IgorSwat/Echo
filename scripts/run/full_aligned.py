#!/usr/bin/env python3
"""Full Echo pipeline for the aligned FM model: text -> prosody tokens -> audio.

    text -> EchoAR -> layer-0 tokens
                   -> CTC forced alignment (same AR, its auxiliary head)
                   -> EchoFMAligned (ODE from noise) -> BlueCodec.decode

The difference from ``scripts/run/full.py`` is the middle stage. The baseline
hands EchoFM the text tokens to attend over; this model reads a frame-level
phoneme path instead, so one is built here: the AR's decoder states over its own
generated tokens are scored by the CTC head and force-aligned against the known
phoneme string. That costs one extra AR forward and a Viterbi pass.

The alignment is what carries the text, so it needs a phoneme per frame and the
AR decides how many frames there are. A short generation can leave fewer frames
than the string needs, which has no valid CTC path; ``--on-align-fail uniform``
falls back to spreading the phonemes evenly rather than giving up.

Usage:
    python scripts/run/full_aligned.py --ar-model checkpoints/ljspeech/echo_ar_best.pt \\
        --fm-model checkpoints/ljspeech/echo_fm_aligned_with_prosody_drop.pt \\
        --text "hello world" --output out.wav

    python scripts/run/full_aligned.py --ar-model ... --fm-model ... \\
        --text "hello world" --steps 32 --cfg 2.0 --print-alignment
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
import torchaudio.functional as AF
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
from echo.fm_model_aligned import EchoFMAligned


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
        description="Run the aligned Echo pipeline from text to audio."
    )
    parser.add_argument("--ar-model", type=str, required=True,
                        help="Path to an EchoAR checkpoint (.pt) trained with the CTC head.")
    parser.add_argument("--fm-model", type=str, required=True,
                        help="Path to an EchoFMAligned checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True,
                        help="Raw text to synthesize (phonemized with eSpeak).")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on AR frames (default: 1000, i.e. 80s).")
    parser.add_argument("--temperature", type=float, default=0.0, metavar="T",
                        help="AR sampling temperature (default: 0.0 = greedy).")
    parser.add_argument("--top-k", type=int, default=0, metavar="K",
                        help="Restrict each AR draw to the K most likely ids (0 = off). "
                             "Only applies when --temperature > 0.")
    parser.add_argument("--steps", type=int, default=8,
                        help="Flow-matching integration steps (default: 8).")
    parser.add_argument("--solver", choices=EchoFMAligned.SOLVERS, default="euler",
                        help="ODE integrator: 'euler' (1 model eval/step, default) or "
                             "'midpoint' (RK2, 2 evals/step, better at few steps).")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Seed the whole render — the AR's sampling as well as the noise "
                             "the transport starts from — so a rendering is reproducible and "
                             "two runs differing only in --cfg are comparable. Omitted, each "
                             "run differs.")
    parser.add_argument("--cfg", type=float, default=2.0,
                        help="Classifier-free guidance scale on the prosody stream "
                             "(default: 2.0; 1.0 disables it). Measured on held-out data, "
                             "1.5-2.0 restores the sample's high-frequency energy to the real "
                             "latent's; past 2.0 the slow bands oversaturate.")
    parser.add_argument("--on-align-fail", choices=("error", "uniform"), default="uniform",
                        help="What to do when the AR emits too few frames to hold the phoneme "
                             "string (default: uniform, i.e. spread them evenly).")
    parser.add_argument("--print-alignment", action="store_true",
                        help="Print the phoneme timeline the FM model reads.")
    parser.add_argument("--stats", type=str, default=None,
                        help="Path to latent_stats.npz (mean/std) used to normalize and "
                             "denormalize latents. Defaults to "
                             "<data_dir>/latents/latent_stats.npz from the training config.")
    parser.add_argument("--cut_last", "--cut-last", type=float, default=0.0, metavar="MS",
                        help="Trim this many milliseconds off the end of the output audio "
                             "(default: 0). The last frames often carry a codec edge artifact.")
    parser.add_argument("--lowpass", type=float, default=0.0, metavar="HZ",
                        help="Low-pass the output at this frequency (default: 0 = off).")
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


# ----------
# Alignment
# ----------

def _ctc_states(ids: torch.Tensor) -> int:
    """Frames a CTC path needs: one per phoneme, plus a blank between repeats."""

    return int(ids.numel() + (ids[1:] == ids[:-1]).sum())


def force_align(
    ar_model: EchoAR,
    codes: torch.Tensor,                                             # (1, T) long
    text_ids: torch.Tensor,                                          # (1, P) long
) -> torch.Tensor:
    """The phoneme path over the AR's own generated tokens: (1, u * T) long.

    Mirrors scripts/preprocess/ctc_align.py, which is what the FM model was
    trained on: the BOS-prefixed input, upsampled decoder states, Viterbi
    against the known string. The trailing `u` frames come from the state that
    predicts EOS and sit past the last audio frame, so they are dropped -- but
    only while they are blank, since a peaky head puts the last phoneme inside
    them and a fixed trim would delete it. Same rule as EchoDataset.
    """

    device = codes.device
    upsample = max(ar_model.ctc_upsample, 1)

    inputs = torch.cat([
        torch.tensor([[config.prosody_bos]], device=device), codes
    ], dim=1)                                                        # (1, T + 1)

    _, hidden = ar_model(inputs, text_ids, return_hidden=True)
    log_probs = ar_model.ctc_log_probs(hidden)                       # (1, u * (T+1), V+1)

    # forced_align has no MPS kernel, and the Viterbi pass is cheap next to the
    # forward, so it always runs on the host.
    path, _ = AF.forced_align(
        log_probs.float().cpu(), text_ids.cpu(), blank=ar_model.ctc_blank,
    )                                                                # (1, u * (T+1))

    row = path[0]
    end = row.numel()
    while end > 0 and row.numel() - end < upsample and row[end - 1] == ar_model.ctc_blank:
        end -= 1

    return row[:end].unsqueeze(0).to(device)


def uniform_align(text_ids: torch.Tensor, frames: int) -> torch.Tensor:
    """Fallback path: every phoneme gets an equal share of the frames."""

    pos = torch.arange(frames, device=text_ids.device)
    idx = (pos * text_ids.shape[1]) // max(frames, 1)

    return text_ids[0, idx.clamp(max=text_ids.shape[1] - 1)].unsqueeze(0)


def print_timeline(align: torch.Tensor, tokenizer, blank: int, rate: float) -> None:
    """The alignment as spans, the way the FM model reads it."""

    from itertools import groupby

    print_section("Alignment")
    t = 0
    for k, g in groupby(align[0].tolist()):
        n = len(list(g))
        symbol = "·  (blank)" if k == blank else repr(tokenizer.detokenize([k]))
        print(f"  {t / rate:6.2f} - {(t + n) / rate:6.2f}s  {n * 1000 / rate:5.0f}ms  {symbol}")
        t += n


def main() -> None:
    args = _parse_args()

    # BlueCodec's STFT reuses an `out` tensor that torch resizes on the first
    # call; the deprecation notice is internal to the codec.
    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    device = select_device()
    output_path = Path(args.output)

    # Seeding the global RNG as well as the transport's own generator is what
    # makes the AR's sampling repeat; without it only stage 3 is reproducible
    # and every run draws different prosody tokens.
    if args.seed is not None:
        torch.manual_seed(args.seed)

    stats_path = (Path(args.stats) if args.stats
                  else REPO_ROOT / config.training.fm.data_dir / "latents" / "latent_stats.npz")
    stats = load_latent_stats(stats_path, device)

    if config.latent_norm == "instance":
        raise SystemExit(
            "latent_norm='instance' took its statistics from the distil, and the "
            "pipeline no longer builds one. Set latent_norm to 'dataset' in "
            "models/config.json and train against latent_stats.npz."
        )

    # --- Models -------------------------------------------------------------
    ar_model = EchoAR().to(device)
    load_checkpoint(ar_model, args.ar_model, device, "EchoAR")
    ar_model.eval()
    if ar_model.ctc_head is None:
        raise SystemExit(
            "the loaded EchoAR has no CTC head, so it cannot align: set "
            "ar_model.ctc.enabled in models/config.json and use a checkpoint "
            "trained with ctc_weight > 0."
        )
    state = torch.load(args.ar_model, map_location="cpu")
    state = state.get("model", state)
    if not any(k.startswith("ctc_") for k in state):
        raise SystemExit(
            f"{args.ar_model} carries no CTC weights (no 'ctc_*' keys), so its head is "
            f"untrained and its alignment would be noise."
        )

    fm_model = EchoFMAligned().to(device)
    load_checkpoint(fm_model, args.fm_model, device, "EchoFMAligned")
    fm_model.eval()

    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval() if args.save_intermediate else None
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    tokenizer = load_tokenizer()
    phonemes = phonemize_args(args.text, args)
    text_ids = torch.tensor([tokenizer.tokenize(phonemes)], dtype=torch.long, device=device)
    if text_ids.numel() == 0:
        raise SystemExit("the text phonemized to nothing this tokenizer recognizes")

    timings: dict[str, float] = {}
    align_rate = MIMI_FPS * max(ar_model.ctc_upsample, 1)

    print_header("Echo - Full Pipeline (aligned)")
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
    print_info("Alignment grid", f"{align_rate:g} Hz "
                                 f"(12.5 Hz x {max(ar_model.ctc_upsample, 1)})", Colors.OKCYAN)
    print_info("Steps", str(args.steps))
    print_info("Solver", args.solver)
    print_info("CFG scale", str(args.cfg))
    print_info("Source", "Gaussian noise"
               + (f", seed {args.seed}" if args.seed is not None
                  else ", unseeded (varies per run)"))
    print_info("Low-pass", f"{args.lowpass:g} Hz" if args.lowpass > 0 else "off")
    if stats is not None:
        print_info("Latent norm", f"per dataset ({stats_path})", Colors.OKCYAN)
    else:
        print_info("Latent norm", f"disabled (stats not found: {stats_path})", Colors.WARNING)

    # --- Stage 1: text -> prosody tokens ------------------------------------
    print_section("Stage 1 — EchoAR")
    with torch.no_grad(), _timed("ar", device, timings):
        codes = ar_model.generate(text_ids, max_frames=args.max_frames,
                                  temperature=args.temperature,
                                  top_k=args.top_k)                  # (1, T_ar)
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

    # --- Stage 2: tokens + text -> frame-level phoneme path -----------------
    print_section("Stage 2 — CTC forced alignment")
    upsample = max(ar_model.ctc_upsample, 1)
    available, needed = upsample * (frames + 1), _ctc_states(text_ids[0])
    fallback = available < needed
    if fallback:
        message = (f"{available} alignment frames cannot hold {needed} CTC states "
                   f"— the AR stopped too early for this text")
        if args.on_align_fail == "error":
            raise SystemExit(message)
        print_info("Forced alignment", message + "; spreading phonemes evenly", Colors.WARNING)

    with torch.no_grad(), _timed("align", device, timings):
        if fallback:
            align = uniform_align(text_ids, upsample * frames)
        else:
            align = force_align(ar_model, codes, text_ids)            # (1, u * T)

    blank = ar_model.ctc_blank
    n_blank = int((align == blank).sum())
    print_info("Frames", f"{align.shape[1]} at {align_rate:g} Hz  "
                         f"({align.shape[1] / text_ids.shape[1]:.2f} per phoneme)")
    print_info("Blanks", f"{n_blank} ({100 * n_blank / align.shape[1]:.1f}%)")
    print_info("Time", f"{timings['align']:.3f}s")
    if args.print_alignment:
        print_timeline(align, tokenizer, blank, align_rate)

    # --- Stage 3: alignment + prosody -> data latent ------------------------
    print_section("Stage 3 — EchoFMAligned")
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    with _timed("fm", device, timings):
        # The alignment takes the place of the text tokens; the prosody stream
        # is unchanged, and it is what the guidance scale acts on.
        latent = fm_model.sample(align, codes, args.steps, args.cfg,
                                 args.solver, generator=generator)
    if stats is not None:
        mean, std = stats
        latent = latent * std + mean

    if args.save_intermediate:
        with torch.no_grad():
            # Mimi wants (B, layers, T); the AR model produces the semantic layer only.
            audio_mimi = decode_mimi(mimi, codes[:, None, :])         # (1, T) @ 24 kHz
        inter = output_path.with_name(output_path.stem + "_ar" + output_path.suffix)
        torchaudio.save(str(inter), audio_mimi.float().cpu(), MIMI_SR)
        print_info("Intermediate", str(inter), Colors.OKCYAN)

    print_info("Latent", f"{tuple(latent.shape)}  "
                         f"({latent.shape[1] / (BLUE_SR / BLUE_HOP):.2f}s)")
    print_info("Time", f"{timings['fm']:.3f}s  ({1000 * timings['fm'] / args.steps:.1f} ms/step)")

    with _timed("decode", device, timings):
        with torch.no_grad():
            audio = blue.decode(latent.transpose(1, 2))               # (1, C, T) -> audio
    audio = audio.squeeze(0).float().cpu()
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    audio = audio[..., : int(duration * BLUE_SR)]

    cut_samples = int(round(args.cut_last * BLUE_SR / 1000))
    if cut_samples > 0:
        if cut_samples >= audio.shape[-1]:
            raise SystemExit(
                f"--cut_last {args.cut_last:g} ms would remove the whole output "
                f"({1000 * audio.shape[-1] / BLUE_SR:.0f} ms)"
            )
        audio = audio[..., :-cut_samples]

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
        ("CTC forced alignment", timings["align"]),
        (f"EchoFMAligned ({args.steps} {args.solver} steps)", timings["fm"]),
        ("BlueCodec decode (latent -> audio)", timings["decode"]),
    ]
    total = sum(t for _, t in rows)
    for label, t in rows:
        print(f"  {Colors.BOLD}{label:<36}{Colors.ENDC} {t:8.3f}s  {100 * t / total:5.1f}%")
    print("  " + "─" * 52)
    print_info("Total", f"{total:.3f}s", Colors.OKCYAN)
    print_info("Audio duration", f"{duration:.2f}s generated")
    print_info("Output", str(output_path), Colors.OKGREEN)
    if cut_samples > 0:
        print_info("Saved duration",
                   f"{saved_duration:.2f}s  (--cut_last {args.cut_last:g} ms trimmed)",
                   Colors.OKCYAN)
    print_info("Real-time factor", f"{total / duration:.2f}x  "
                                   f"({duration / total:.2f}x realtime; lower RTF is faster)",
               Colors.OKGREEN if total < duration else Colors.WARNING)

    print_separator()


if __name__ == "__main__":
    main()
