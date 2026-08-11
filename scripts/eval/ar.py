#!/usr/bin/env python3
"""Evaluate an EchoAR checkpoint: does it say the words, and in time?

Neither failure the AR stage actually has shows up in its cross-entropy — a
skipped word costs a handful of frames out of hundreds, and drift costs nothing
at all — so the decoded audio is measured directly:

    phonemes -> EchoAR -> Mimi.decode -> 24 kHz audio
                                      |-> Whisper -> WER vs the transcript
                                      |-> DTW     -> timing vs the recording

Deletions are the number to watch. The same ASR runs on the original recording
to give a floor, since only the gap between the two is the model's doing.

Usage:
    python scripts/eval/ar.py --model checkpoints/ljspeech/echo_ar_best.pt
    python scripts/eval/ar.py --model ... --limit 20 --temperature 1.0 --top-k 50
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import time
import warnings

import librosa
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from transformers import MimiModel

from __common__ import (
    MIMI_FPS,
    MIMI_SR,
    decode_mimi,
    load_checkpoint,
    load_pairs_csv,
    load_tokenizer,
    select_device,
)
from __eval__ import ASR, edit_ops, normalize_text
from __style__ import (
    Colors,
    print_error,
    print_header,
    print_info,
    print_section,
    print_separator,
    print_success,
)

from echo.ar_model import EchoAR

# Mel front-end for the DTW. 25 ms window / 10 ms hop is the usual speech
# analysis grid and is fine enough to resolve a syllable.
_MEL_N_FFT = 600
_MEL_HOP = 240                                                       # 10 ms at 24 kHz
_MEL_BINS = 40


# ----------
# Alignment
# ----------

def _log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=_MEL_N_FFT, hop_length=_MEL_HOP, n_mels=_MEL_BINS,
    )

    return librosa.power_to_db(mel, ref=np.max)                      # (n_mels, T)


def _alignment_metrics(gen: np.ndarray, ref: np.ndarray, sample_rate: int) -> dict[str, float]:
    """
    DTW the generated audio against the recording and describe the warping path.

    NOTE: cosine distance compares spectral *shape*, so it stays blind to the
    level and timbre shifts a codec round-trip introduces.
    """

    X, Y = _log_mel(gen, sample_rate), _log_mel(ref, sample_rate)
    D, path = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
    path = path[::-1]                                                # start -> end
    sec = _MEL_HOP / sample_rate
    gi, ri = path[:, 0].astype(float), path[:, 1].astype(float)

    # Where the path would run if the generation were the recording at a constant
    # tempo. Deviation from it is *local* drift, with global tempo already
    # divided out — a rendition uniformly 10% slow scores 0.
    diagonal = gi * (Y.shape[1] - 1) / max(X.shape[1] - 1, 1)
    deviation = np.abs(ri - diagonal) * sec

    def longest_stall(idx: np.ndarray) -> float:
        """Longest run where this index does not advance while the other does."""
        best = run = 0
        for stalled in np.diff(idx) == 0:
            run = run + 1 if stalled else 0
            best = max(best, run)
        return best * sec

    return {
        "gen_dur": X.shape[1] * sec,
        "ref_dur": Y.shape[1] * sec,
        "dur_ratio": (X.shape[1] * sec) / max(Y.shape[1] * sec, 1e-9),
        "dtw_cost": float(D[-1, -1] / len(path)),
        "dev_mean": float(deviation.mean()),
        "dev_p95": float(np.percentile(deviation, 95)),
        # Generated stalls while the reference runs on: the recording has content
        # the generation skipped.
        "stall_gen": longest_stall(gi),
        "stall_ref": longest_stall(ri),
    }


def _load_transcripts(path: Path) -> dict[str, str]:
    """``<name>.wav|<text>`` (LJSpeech metadata), keyed on the bare stem."""
    out: dict[str, str] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or "|" not in line:
            continue
        parts = line.split("|")
        # LJSpeech ships raw and normalized transcripts; prefer the last
        # non-empty field, which is the normalized one when present.
        text = next((p for p in reversed(parts[1:]) if p.strip()), "")
        out[Path(parts[0]).stem] = text.strip()

    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate an EchoAR checkpoint: ASR word error rate and DTW alignment."
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to an EchoAR checkpoint (.pt).")
    parser.add_argument("--test-suite", type=str, default="data/ljspeech/phonemes_test.csv",
                        help="CSV of '<name>.npz|<phonemes>' lines.")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Directory of reference recordings "
                             "(default: <test-suite dir>/audio).")
    parser.add_argument("--metadata", type=str, default=None,
                        help="CSV of '<name>.wav|<text>' transcripts for the WER reference "
                             "(default: <test-suite dir>/metadata.csv).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N entries.")

    gen = parser.add_argument_group("AR decoding")
    gen.add_argument("--max-frames", type=int, default=1000,
                     help="Hard cap on generated frames (default: 1000, i.e. 80s).")
    gen.add_argument("--temperature", type=float, default=0.0,
                     help="Sampling temperature (default: 0.0 = greedy).")
    gen.add_argument("--top-k", type=int, default=0,
                     help="Restrict each draw to the K most likely ids (0 = off).")
    gen.add_argument("--seed", type=int, default=42, help="Seeds sampling (default: 42).")

    asr = parser.add_argument_group("ASR")
    asr.add_argument("--asr-backend", choices=("auto", "mlx", "transformers"), default="auto",
                     help="'auto' (default) prefers mlx-whisper when it is installed.")
    asr.add_argument("--asr-model", type=str, default=None,
                     help="Whisper model id (default: mlx-community/whisper-small-mlx on MLX, "
                          "openai/whisper-small otherwise).")
    asr.add_argument("--language", type=str, default="en", help="ASR language (default: en).")
    asr.add_argument("--no-asr", action="store_true", help="Skip the WER stage entirely.")
    asr.add_argument("--no-asr-floor", action="store_true",
                     help="Do not transcribe the original recordings. Halves ASR time, but the "
                          "WER then has no baseline to be read against.")

    out = parser.add_argument_group("Output")
    out.add_argument("--no-align", action="store_true", help="Skip the DTW alignment stage.")
    out.add_argument("--save-dir", type=str, default=None,
                     help="Write each generated waveform here as <name>.wav.")
    out.add_argument("--csv", type=str, default=None,
                     help="Write per-file metrics to this CSV.")
    out.add_argument("--verbose", action="store_true",
                     help="Print the reference and hypothesis text for every entry.")
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be >= 0 (0 selects greedy decoding)")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 disables top-k)")

    return args


def _report_wer(agg) -> None:
    # Corpus WER, not the mean of per-file rates: long utterances should weigh
    # more, and a 3-word file should not swing the number with a single error.
    total_ref = agg("ref_words").sum()
    s, d, i = agg("sub").sum(), agg("dele").sum(), agg("ins").sum()

    print_section("Word error rate")
    print_info("WER", f"{(s + d + i) / max(total_ref, 1):.4f}", Colors.OKGREEN)
    print_info("  substitutions", f"{s / max(total_ref, 1):.4f}  ({int(s)})")
    print_info("  deletions", f"{d / max(total_ref, 1):.4f}  ({int(d)})  <- dropped words",
               Colors.WARNING)
    print_info("  insertions", f"{i / max(total_ref, 1):.4f}  ({int(i)})")

    per_file = agg("wer")
    print_info("Per-file WER", f"median {np.median(per_file):.3f}, "
                               f"p90 {np.percentile(per_file, 90):.3f}, "
                               f"max {per_file.max():.3f}")
    print_info("Files above 50% WER", f"{int((per_file > 0.5).sum())}/{len(per_file)}")

    floor = agg("wer_floor")
    if floor.size:
        print_info("ASR floor (real audio)", f"{floor.mean():.4f}", Colors.OKCYAN)
        print_info("Attributable to the model",
                   f"{(s + d + i) / max(total_ref, 1) - floor.mean():+.4f}", Colors.OKCYAN)


def _report_alignment(agg) -> None:
    ratio, dev = agg("dur_ratio"), agg("dev_mean")
    skipped, added, cost = agg("stall_gen"), agg("stall_ref"), agg("dtw_cost")

    print_section("Timing vs the original recording (lower is better; tempo targets 1.00)")

    def fmt(seconds: float) -> str:
        return f"{seconds:.1f}s" if seconds >= 1.0 else f"{seconds * 1000:.0f}ms"

    def row(label: str, value: str, meaning: str, bad: bool = False) -> None:
        colour = Colors.WARNING if bad else Colors.OKGREEN
        print(f"  {Colors.BOLD}{label:<9}{Colors.ENDC} {colour}{value:<34}{Colors.ENDC}{meaning}")

    med_ratio, med_dev = float(np.median(ratio)), float(np.median(dev))
    med_skip, med_add = float(np.median(skipped)), float(np.median(added))

    row("Tempo", f"x{med_ratio:.2f}   (p10 {np.percentile(ratio, 10):.2f}, "
                 f"p90 {np.percentile(ratio, 90):.2f})",
        "length vs the recording", bad=not 0.9 <= med_ratio <= 1.1)
    row("Drift", f"{fmt(med_dev)} typical, {fmt(float(dev.max()))} worst",
        "timing error after tempo is divided out", bad=med_dev > 0.15)
    row("Skipped", f"{fmt(med_skip)} typical, {fmt(float(skipped.max()))} worst",
        "longest unmatched stretch of the recording: dropped words", bad=med_skip > 0.2)
    row("Added", f"{fmt(med_add)} typical, {fmt(float(added.max()))} worst",
        "longest unmatched stretch generated: babble or repeats", bad=med_add > 0.2)
    row("Spectral", f"{cost.mean():.4f}",
        "0 = identical, ~0.03 = unrelated", bad=float(cost.mean()) > 0.02)

    # Medians hide the failures that matter, so count them explicitly.
    bad_skip, bad_add = int((skipped > 0.3).sum()), int((added > 1.0).sum())
    print()
    print_info("Files skipping >300 ms", f"{bad_skip}/{len(skipped)}",
               Colors.WARNING if bad_skip else Colors.OKGREEN)
    print_info("Files adding >1 s", f"{bad_add}/{len(skipped)}",
               Colors.WARNING if bad_add else Colors.OKGREEN)


def _write_csv(rows: list[dict], csv_path: Path) -> None:
    keys = sorted({k for r in rows for k in r})
    keys = ["name"] + [k for k in keys if k != "name"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(
                f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                for k in keys
            ) + "\n")


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    # --- Inputs -------------------------------------------------------------
    suite_path = Path(args.test_suite)
    if not suite_path.is_file():
        print_error(f"Test suite not found: {suite_path}")
        sys.exit(1)
    data_dir = suite_path.parent
    audio_dir = Path(args.audio_dir) if args.audio_dir else data_dir / "audio"
    meta_path = Path(args.metadata) if args.metadata else data_dir / "metadata.csv"

    items = load_pairs_csv(suite_path)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        print_error(f"No entries in {suite_path}")
        sys.exit(1)

    need_audio = not args.no_align or not args.no_asr_floor
    if need_audio and not audio_dir.is_dir():
        print_error(f"Reference audio directory not found: {audio_dir}")
        sys.exit(1)

    transcripts: dict[str, str] = {}
    if not args.no_asr:
        if not meta_path.is_file():
            print_error(f"Transcript CSV not found: {meta_path} (pass --metadata, or --no-asr)")
            sys.exit(1)
        transcripts = _load_transcripts(meta_path)

    device = select_device()
    torch.manual_seed(args.seed)

    print_header("EchoAR - Evaluation")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Checkpoint", args.model)
    print_info("Test suite", f"{suite_path} ({len(items)} entries)")
    print_info("AR decoding",
               "greedy (argmax)" if args.temperature <= 0
               else f"sampling (temperature {args.temperature:g}"
                    + (f", top-k {args.top_k}" if args.top_k > 0 else "") + ")")

    # --- Models -------------------------------------------------------------
    ar_model = EchoAR().to(device)
    ckpt = load_checkpoint(ar_model, args.model, device, "EchoAR")
    ar_model.eval()
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print_info("Trained", f"epoch {ckpt['epoch']}, "
                              f"val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    tokenizer = load_tokenizer()
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    asr_engine = None
    if not args.no_asr:
        asr_engine = ASR(args.asr_backend, args.asr_model, args.language, device)
        print_info("ASR", f"{asr_engine.name} ({asr_engine.backend})", Colors.OKCYAN)
        print_info("ASR floor", "skipped" if args.no_asr_floor else "original audio transcribed")
    else:
        print_info("ASR", "skipped", Colors.WARNING)
    print_info("Alignment", "skipped" if args.no_align else "DTW over log-mel frames")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print_info("Saving audio to", str(save_dir), Colors.OKCYAN)

    # --- Evaluate -----------------------------------------------------------
    print_section("Evaluating")
    rows: list[dict] = []
    n_fail, n_capped, n_empty = 0, 0, 0
    t_start = time.perf_counter()

    for name, phonemes in tqdm(items, desc="Evaluating", unit="file"):
        stem = Path(name).stem
        try:
            text_ids = torch.tensor(
                [tokenizer.tokenize(phonemes)], dtype=torch.long, device=device
            )
            with torch.no_grad():
                codes = ar_model.generate(text_ids, max_frames=args.max_frames,
                                          temperature=args.temperature, top_k=args.top_k)
            frames = codes.shape[1]
            if frames == 0:
                n_empty += 1
                rows.append({"name": stem, "frames": 0})
                continue
            if frames >= args.max_frames:
                n_capped += 1

            with torch.no_grad():
                audio_t = decode_mimi(mimi, codes.transpose(1, 2))    # (1, T) @ 24 kHz
            gen_audio = audio_t.squeeze(0).float().cpu().numpy()

            if save_dir:
                torchaudio.save(str(save_dir / f"{stem}.wav"),
                                torch.from_numpy(gen_audio)[None], MIMI_SR)

            row: dict = {"name": stem, "frames": frames, "gen_dur": frames / MIMI_FPS}

            ref_audio = None
            if need_audio:
                ref_path = next(
                    (p for p in (audio_dir / f"{stem}{ext}" for ext in (".wav", ".flac", ".mp3"))
                     if p.is_file()), None
                )
                if ref_path is None:
                    raise FileNotFoundError(f"no reference audio for {stem} in {audio_dir}")
                ref_audio, _ = librosa.load(str(ref_path), sr=MIMI_SR, mono=True)

            if not args.no_align:
                row.update(_alignment_metrics(gen_audio, ref_audio, MIMI_SR))

            if asr_engine is not None:
                reference = transcripts.get(stem)
                if reference is None:
                    raise KeyError(f"no transcript for {stem} in {meta_path}")
                ref_words = normalize_text(reference)
                hyp_words = normalize_text(asr_engine.transcribe(gen_audio, MIMI_SR))
                s, d, i = edit_ops(ref_words, hyp_words)
                row.update({"ref_words": len(ref_words), "sub": s, "dele": d, "ins": i,
                            "wer": (s + d + i) / max(len(ref_words), 1)})

                if not args.no_asr_floor:
                    fs, fd, fi = edit_ops(
                        ref_words, normalize_text(asr_engine.transcribe(ref_audio, MIMI_SR))
                    )
                    row["wer_floor"] = (fs + fd + fi) / max(len(ref_words), 1)

                if args.verbose:
                    tqdm.write(f"  {stem}  WER {row['wer']:.3f}  (S{s} D{d} I{i})")
                    tqdm.write(f"    ref: {' '.join(ref_words)}")
                    tqdm.write(f"    hyp: {' '.join(hyp_words)}")

            rows.append(row)
        except Exception as e:                                       # noqa: BLE001
            tqdm.write(f"{Colors.FAIL}Failed on {stem}: {e}{Colors.ENDC}")
            n_fail += 1

    elapsed = time.perf_counter() - t_start
    scored = [r for r in rows if r.get("frames", 0) > 0]
    if not scored:
        print_error("Nothing was evaluated successfully.")
        sys.exit(1)

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_section("Results")
    print_info("Evaluated", f"{len(scored)}/{len(items)}", Colors.OKGREEN)
    if n_fail:
        print_info("Failed", str(n_fail), Colors.FAIL)
    if n_empty:
        print_info("Empty generations", f"{n_empty} (EOS emitted immediately)", Colors.FAIL)
    if n_capped:
        print_info("Hit --max-frames", f"{n_capped} (no EOS emitted)", Colors.WARNING)

    def agg(key: str) -> np.ndarray:
        return np.array([r[key] for r in scored if key in r], dtype=float)

    if asr_engine is not None and any("wer" in r for r in scored):
        _report_wer(agg)
    if not args.no_align and any("dur_ratio" in r for r in scored):
        _report_alignment(agg)

    print_section("Worst entries")
    rank_key = "wer" if any("wer" in r for r in scored) else "dev_mean"
    worst = sorted((r for r in scored if rank_key in r),
                   key=lambda r: r[rank_key], reverse=True)[:5]
    for r in worst:
        bits = [f"{r['name']}"]
        if "wer" in r:
            bits.append(f"WER {r['wer']:.3f} (S{r['sub']} D{r['dele']} I{r['ins']})")
        if "dur_ratio" in r:
            bits.append(f"dur x{r['dur_ratio']:.2f}, stall {r['stall_gen'] * 1000:.0f} ms")
        print(f"  {Colors.BOLD}{bits[0]:<16}{Colors.ENDC} " + "  |  ".join(bits[1:]))

    if args.csv:
        _write_csv(rows, Path(args.csv))
        print_info("Per-file metrics", str(args.csv), Colors.OKCYAN)

    print_separator("═", 60)
    print_info("Total time", f"{elapsed:.1f}s  ({elapsed / len(scored):.2f}s/file)", Colors.OKCYAN)
    print_success("Done.")


if __name__ == "__main__":
    main()
