#!/usr/bin/env python3
"""Evaluate an EchoAR checkpoint on a test suite: does it say the words, and in time?

Two failure modes matter for the AR stage, and neither is visible in its
cross-entropy: it can *drop* content (a skipped word costs a handful of frames
of loss out of hundreds), and it can lose the plot temporally (rushing, stalling,
or running past the real duration). This script measures both directly.

    phonemes -> EchoAR -> Mimi.decode -> 24 kHz audio
                                      |-> Whisper  -> WER vs the reference text
                                      |-> DTW      -> timing vs the real recording

**Word error rate.** The decoded audio is transcribed and scored against the
corpus transcript, broken down into substitutions / deletions / insertions.
Deletions are the number to watch: they are the missing words. The same ASR is
run on the *original* recording to establish a floor — Whisper is not perfect on
LJSpeech either, and only the gap between the two is attributable to the model.

**Alignment.** Generated and original audio are compared by DTW over log-mel
frames, which needs no aligner and no transcript. Three numbers come out:

  - *duration ratio* — global tempo, generated over real.
  - *path deviation* — how far the warping path strays from the straight line a
    perfectly-paced rendition would follow, in seconds. Catches drift that a
    matching total duration would otherwise hide.
  - *stalls* — the longest run where the path advances in one signal but not the
    other. A long stall on the generated side means the recording contains
    something the generation does not: a dropped word, in the time domain rather
    than the text domain. It corroborates the ASR deletions without depending
    on ASR.

Usage:
    python scripts/eval_ar.py --model checkpoints/ljspeech/echo_ar_best.pt
    python scripts/eval_ar.py --model checkpoints/ljspeech/echo_ar_best.pt \\
        --limit 20 --temperature 1.0 --top-k 50 --save-dir out/eval_ar
    python scripts/eval_ar.py --model checkpoints/ljspeech/echo_ar_best.pt \\
        --no-asr-floor --csv out/eval_ar.csv

Whisper runs on MLX when ``mlx-whisper`` is installed (``pip install
mlx-whisper``), which is much faster on Apple silicon, and falls back to the
``transformers`` implementation otherwise.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import librosa  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import MimiModel  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_error,
    print_header,
    print_info,
    print_section,
    print_separator,
    print_success,
)

from echo.ar_model import EchoAR  # noqa: E402
from echo.tokenizer import Tokenizer  # noqa: E402

# Reuse the decode helper rather than restate it.
from run_ar import _decode_codes  # noqa: E402

_MIMI_SR = 24000
_MIMI_FPS = 12.5
_ASR_SR = 16000                      # every Whisper variant wants 16 kHz mono

# Mel front-end for the DTW. 25 ms window / 10 ms hop is the usual speech
# analysis grid and is fine enough to resolve a syllable.
_MEL_N_FFT = 600
_MEL_HOP = 240                       # 10 ms at 24 kHz
_MEL_BINS = 40


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------

def _build_normalizer():
    """Whisper's own English normalizer when available, else a plain fallback.

    WER is only meaningful if both sides are written the same way, and the two
    sides here disagree on more than punctuation: Whisper writes "11" where the
    corpus transcript writes "eleven", and expands contractions differently.
    Whisper ships the normalizer it was evaluated with, which settles all of
    that consistently — scoring against anything else invents errors the model
    did not make.
    """
    try:
        from transformers.models.whisper.english_normalizer import (   # noqa: PLC0415
            EnglishTextNormalizer,
        )
    except ImportError:
        return None
    return EnglishTextNormalizer({})


_NORMALIZER = _build_normalizer()


def _normalize_text(text: str) -> list[str]:
    """Normalize a transcript to a comparable word list."""
    if _NORMALIZER is not None:
        return _NORMALIZER(text).split()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("-", " ").replace("—", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return text.split()


def _edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """Levenshtein alignment of two word sequences -> (subs, deletions, insertions).

    Kept separate rather than collapsed into a single WER because the three
    error types mean different things here: deletions are dropped words (the
    failure this script exists to find), insertions are usually the model
    babbling past the text or the ASR hallucinating in silence.
    """
    n, m = len(ref), len(hyp)
    # d[i][j] = cost, plus a backtrace of which op got us there.
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------

class _ASR:
    """Whisper on MLX where available, otherwise the transformers implementation."""

    _MLX_DEFAULT = "mlx-community/whisper-small-mlx"
    _HF_DEFAULT = "openai/whisper-small"

    def __init__(self, backend: str, model: str | None, language: str, device: torch.device) -> None:
        if backend == "auto":
            backend = "mlx" if self._mlx_available() else "transformers"
        if backend == "mlx" and not self._mlx_available():
            raise SystemExit(
                "--asr-backend mlx needs the mlx-whisper package: pip install mlx-whisper"
            )

        self.backend = backend
        self.language = language
        self.name = model or (self._MLX_DEFAULT if backend == "mlx" else self._HF_DEFAULT)

        if backend == "mlx":
            import mlx_whisper                                        # noqa: PLC0415
            self._mlx = mlx_whisper
        else:
            from transformers import pipeline                          # noqa: PLC0415
            # Whisper's own generation defaults handle >30 s inputs by chunking.
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.name,
                device=device,
                chunk_length_s=30,
            )

    @staticmethod
    def _mlx_available() -> bool:
        from importlib.util import find_spec                           # noqa: PLC0415
        return find_spec("mlx_whisper") is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """``audio`` is mono float32 at ``sample_rate``; resampled to 16 kHz here."""
        if sample_rate != _ASR_SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=_ASR_SR)
        audio = audio.astype(np.float32)

        if self.backend == "mlx":
            out = self._mlx.transcribe(
                audio, path_or_hf_repo=self.name, language=self.language, verbose=None,
            )
            return out["text"]
        return self._pipe(
            {"raw": audio, "sampling_rate": _ASR_SR},
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )["text"]


# --------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------

def _log_mel(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_fft=_MEL_N_FFT, hop_length=_MEL_HOP, n_mels=_MEL_BINS,
    )
    return librosa.power_to_db(mel, ref=np.max)                       # (n_mels, T)


def _alignment_metrics(gen: np.ndarray, ref: np.ndarray, sample_rate: int) -> dict[str, float]:
    """DTW the generated audio against the recording and describe the warping path.

    Cosine distance over log-mel frames: it compares spectral *shape* and so is
    largely blind to the level and timbre differences a codec round-trip
    introduces, which are not what we are measuring here.
    """
    X, Y = _log_mel(gen, sample_rate), _log_mel(ref, sample_rate)
    D, path = librosa.sequence.dtw(X=X, Y=Y, metric="cosine")
    path = path[::-1]                                                  # start -> end
    sec = _MEL_HOP / sample_rate
    gi, ri = path[:, 0].astype(float), path[:, 1].astype(float)

    # Where the path would run if the generation were the recording at a
    # constant tempo. Deviation from it is *local* drift, with global tempo
    # already divided out — a rendition that is uniformly 10% slow scores 0.
    diagonal = gi * (Y.shape[1] - 1) / max(X.shape[1] - 1, 1)
    deviation = np.abs(ri - diagonal) * sec

    # A stall is a run where one index advances and the other does not: content
    # present in one signal and absent in the other.
    def _longest_stall(idx: np.ndarray) -> float:
        stalled = np.diff(idx) == 0
        best = run = 0
        for s in stalled:
            run = run + 1 if s else 0
            best = max(best, run)
        return best * sec

    return {
        "gen_dur": X.shape[1] * sec,
        "ref_dur": Y.shape[1] * sec,
        "dur_ratio": (X.shape[1] * sec) / max(Y.shape[1] * sec, 1e-9),
        "dtw_cost": float(D[-1, -1] / len(path)),
        "dev_mean": float(deviation.mean()),
        "dev_p95": float(np.percentile(deviation, 95)),
        # Generated stalls while the reference runs on: the recording has
        # content the generation skipped.
        "stall_gen": _longest_stall(gi),
        "stall_ref": _longest_stall(ri),
    }


# --------------------------------------------------------------------------

def _load_suite(path: Path) -> list[tuple[str, str]]:
    """``<name>.npz|<phonemes>`` lines, as written by the phonemizer."""
    items: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and "|" in line:
                name, phonemes = line.split("|", 1)
                items.append((name.strip(), phonemes.strip()))
    return items


def _load_transcripts(path: Path) -> dict[str, str]:
    """``<name>.wav|<text>`` (LJSpeech metadata). Keyed on the bare stem."""
    out: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            parts = line.split("|")
            # LJSpeech ships raw and normalized transcripts; prefer the last
            # non-empty field, which is the normalized one when present.
            text = next((p for p in reversed(parts[1:]) if p.strip()), "")
            out[Path(parts[0]).stem] = text.strip()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an EchoAR checkpoint: ASR word error rate and DTW alignment."
    )
    parser.add_argument("--model", type=str, required=True, help="Path to an EchoAR checkpoint (.pt).")
    parser.add_argument("--test-suite", type=str, default="data/ljspeech/phonemes_test.csv",
                        help="CSV of '<name>.npz|<phonemes>' lines "
                             "(default: data/ljspeech/phonemes_test.csv).")
    parser.add_argument("--audio-dir", type=str, default=None,
                        help="Directory of reference recordings "
                             "(default: <test-suite dir>/audio).")
    parser.add_argument("--metadata", type=str, default=None,
                        help="CSV of '<name>.wav|<text>' transcripts for the WER reference "
                             "(default: <test-suite dir>/metadata.csv).")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N entries.")

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
    out.add_argument("--csv", type=str, default=None, help="Write per-file metrics to this CSV.")
    out.add_argument("--verbose", action="store_true",
                     help="Print the reference and hypothesis text for every entry.")
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be >= 0 (0 selects greedy decoding)")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 disables top-k)")

    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    suite_path = Path(args.test_suite)
    if not suite_path.is_file():
        print_error(f"Test suite not found: {suite_path}")
        sys.exit(1)
    data_dir = suite_path.parent
    audio_dir = Path(args.audio_dir) if args.audio_dir else data_dir / "audio"
    meta_path = Path(args.metadata) if args.metadata else data_dir / "metadata.csv"

    items = _load_suite(suite_path)
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

    device = _select_device()
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
    ckpt = torch.load(args.model, map_location=device)
    ar_model = EchoAR().to(device)
    ar_model.load_state_dict(ckpt.get("model", ckpt))
    ar_model.eval()
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print_info("Trained", f"epoch {ckpt['epoch']}, "
                              f"val_loss {ckpt.get('val_loss', float('nan')):.4f}")

    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    asr_engine = None
    if not args.no_asr:
        asr_engine = _ASR(args.asr_backend, args.asr_model, args.language, device)
        print_info("ASR", f"{asr_engine.name} ({asr_engine.backend})", Colors.OKCYAN)
        print_info("ASR floor", "skipped" if args.no_asr_floor else "original audio transcribed")
    else:
        print_info("ASR", "skipped", Colors.WARNING)
    print_info("Alignment", "skipped" if args.no_align else "DTW over log-mel frames")

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        print_info("Saving audio to", str(save_dir), Colors.OKCYAN)

    # --- Evaluate ------------------------------------------------------------
    print_section("Evaluating")
    rows: list[dict] = []
    n_fail, n_capped, n_empty = 0, 0, 0
    t_start = time.perf_counter()

    for name, phonemes in tqdm(items, desc="Evaluating", unit="file"):
        stem = Path(name).stem
        try:
            text_ids = torch.tensor([tokenizer.tokenize(phonemes)], dtype=torch.long, device=device)
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
                audio_t = _decode_codes(mimi, codes.transpose(1, 2))   # (1, T) @ 24 kHz
            gen_audio = audio_t.squeeze(0).float().cpu().numpy()

            if save_dir:
                torchaudio.save(str(save_dir / f"{stem}.wav"),
                                torch.from_numpy(gen_audio)[None], _MIMI_SR)

            row: dict = {"name": stem, "frames": frames,
                         "gen_dur": frames / _MIMI_FPS}

            ref_audio = None
            if need_audio:
                ref_path = next((p for p in (audio_dir / f"{stem}{ext}"
                                             for ext in (".wav", ".flac", ".mp3")) if p.is_file()), None)
                if ref_path is None:
                    raise FileNotFoundError(f"no reference audio for {stem} in {audio_dir}")
                ref_audio, _ = librosa.load(str(ref_path), sr=_MIMI_SR, mono=True)

            if not args.no_align:
                row.update(_alignment_metrics(gen_audio, ref_audio, _MIMI_SR))

            if asr_engine is not None:
                reference = transcripts.get(stem)
                if reference is None:
                    raise KeyError(f"no transcript for {stem} in {meta_path}")
                ref_words = _normalize_text(reference)
                hyp_words = _normalize_text(asr_engine.transcribe(gen_audio, _MIMI_SR))
                s, d, i = _edit_ops(ref_words, hyp_words)
                row.update({"ref_words": len(ref_words), "sub": s, "dele": d, "ins": i,
                            "wer": (s + d + i) / max(len(ref_words), 1)})

                if not args.no_asr_floor:
                    fs, fd, fi = _edit_ops(ref_words,
                                           _normalize_text(asr_engine.transcribe(ref_audio, _MIMI_SR)))
                    row["wer_floor"] = (fs + fd + fi) / max(len(ref_words), 1)

                if args.verbose:
                    tqdm.write(f"  {stem}  WER {row['wer']:.3f}  (S{s} D{d} I{i})")
                    tqdm.write(f"    ref: {' '.join(ref_words)}")
                    tqdm.write(f"    hyp: {' '.join(hyp_words)}")

            rows.append(row)
        except Exception as e:                                         # noqa: BLE001
            tqdm.write(f"{Colors.FAIL}Failed on {stem}: {e}{Colors.ENDC}")
            n_fail += 1

    elapsed = time.perf_counter() - t_start
    scored = [r for r in rows if r.get("frames", 0) > 0]
    if not scored:
        print_error("Nothing was evaluated successfully.")
        sys.exit(1)

    # --- Summary --------------------------------------------------------------
    print_separator("═", 60)
    print_section("Results")
    print_info("Evaluated", f"{len(scored)}/{len(items)}", Colors.OKGREEN)
    if n_fail:
        print_info("Failed", str(n_fail), Colors.FAIL)
    if n_empty:
        print_info("Empty generations", f"{n_empty} (EOS emitted immediately)", Colors.FAIL)
    if n_capped:
        print_info("Hit --max-frames", f"{n_capped} (no EOS emitted)", Colors.WARNING)

    def _agg(key: str) -> np.ndarray:
        return np.array([r[key] for r in scored if key in r], dtype=float)

    if asr_engine is not None and any("wer" in r for r in scored):
        # Corpus WER, not the mean of per-file rates: long utterances should
        # weigh more than short ones, and a 3-word file should not be able to
        # swing the number with a single error.
        total_ref = _agg("ref_words").sum()
        s, d, i = _agg("sub").sum(), _agg("dele").sum(), _agg("ins").sum()
        print_section("Word error rate")
        print_info("WER", f"{(s + d + i) / max(total_ref, 1):.4f}", Colors.OKGREEN)
        print_info("  substitutions", f"{s / max(total_ref, 1):.4f}  ({int(s)})")
        print_info("  deletions", f"{d / max(total_ref, 1):.4f}  ({int(d)})  <- dropped words",
                   Colors.WARNING)
        print_info("  insertions", f"{i / max(total_ref, 1):.4f}  ({int(i)})")
        per_file = _agg("wer")
        print_info("Per-file WER", f"median {np.median(per_file):.3f}, "
                                   f"p90 {np.percentile(per_file, 90):.3f}, "
                                   f"max {per_file.max():.3f}")
        print_info("Files above 50% WER", f"{int((per_file > 0.5).sum())}/{len(per_file)}")
        floor = _agg("wer_floor")
        if floor.size:
            print_info("ASR floor (real audio)", f"{floor.mean():.4f}", Colors.OKCYAN)
            print_info("Attributable to the model",
                       f"{(s + d + i) / max(total_ref, 1) - floor.mean():+.4f}", Colors.OKCYAN)

    if not args.no_align and any("dur_ratio" in r for r in scored):
        ratio, dev = _agg("dur_ratio"), _agg("dev_mean")
        skipped, added, cost = _agg("stall_gen"), _agg("stall_ref"), _agg("dtw_cost")

        print_section("Timing vs the original recording")
        print(f"  {Colors.BOLD}Every line compares the generated audio to the real recording of "
              f"the same sentence.{Colors.ENDC}")
        print(f"  {Colors.BOLD}Lower is better everywhere except Tempo, where 1.00 is "
              f"the target.{Colors.ENDC}\n")

        def _t(seconds: float) -> str:
            return f"{seconds:.1f}s" if seconds >= 1.0 else f"{seconds * 1000:.0f}ms"

        def _row(label: str, value: str, meaning: str, bad: bool = False) -> None:
            colour = Colors.WARNING if bad else Colors.OKGREEN
            print(f"  {Colors.BOLD}{label:<9}{Colors.ENDC} {colour}{value:<34}{Colors.ENDC}{meaning}")

        med_ratio = float(np.median(ratio))
        _row("Tempo", f"x{med_ratio:.2f}   (p10 {np.percentile(ratio, 10):.2f}, "
                      f"p90 {np.percentile(ratio, 90):.2f})",
             "total length vs the recording. >1 slower, <1 faster",
             bad=not 0.9 <= med_ratio <= 1.1)
        _row("Drift", f"{_t(float(np.median(dev)))} typical, {_t(float(dev.max()))} worst",
             "timing error *after* tempo is divided out", bad=float(np.median(dev)) > 0.15)
        _row("Skipped", f"{_t(float(np.median(skipped)))} typical, "
                        f"{_t(float(skipped.max()))} worst",
             "longest bit of the recording with no match: dropped words",
             bad=float(np.median(skipped)) > 0.2)
        _row("Added", f"{_t(float(np.median(added)))} typical, {_t(float(added.max()))} worst",
             "longest bit generated with no match: babble or repeats",
             bad=float(np.median(added)) > 0.2)
        _row("Spectral", f"{cost.mean():.4f}",
             "0 = identical audio, ~0.03 = unrelated sentences", bad=float(cost.mean()) > 0.02)

        # Medians hide the failures that matter, so count them explicitly.
        bad_skip = int((skipped > 0.3).sum())
        bad_add = int((added > 1.0).sum())
        n = len(skipped)
        print()
        print_info("Files skipping >300 ms", f"{bad_skip}/{n}",
                   Colors.WARNING if bad_skip else Colors.OKGREEN)
        print_info("Files adding >1 s", f"{bad_add}/{n}  (often the model not stopping)",
                   Colors.WARNING if bad_add else Colors.OKGREEN)

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
        keys = sorted({k for r in rows for k in r})
        keys = ["name"] + [k for k in keys if k != "name"]
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for r in rows:
                f.write(",".join(
                    f"{r[k]:.6g}" if isinstance(r.get(k), float) else str(r.get(k, ""))
                    for k in keys
                ) + "\n")
        print_info("Per-file metrics", str(csv_path), Colors.OKCYAN)

    print_separator("═", 60)
    print_info("Total time", f"{elapsed:.1f}s  ({elapsed / len(scored):.2f}s/file)", Colors.OKCYAN)
    print_success("Done.")


if __name__ == "__main__":
    main()
