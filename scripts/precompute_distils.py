#!/usr/bin/env python3
"""Precompute distillation latents: Mimi encode → truncate → decode → BlueCodec encode.

Pipeline: audio -> Mimi codec (first --layers layers) -> decoded audio -> BlueCodec latents.

With ``--codec-dir`` the Mimi encode step is skipped: the codecs are read from
``.npz`` files (a ``codes`` array of shape ``(layers, T_codec)``, as written by
``precompute_codecs.py``) and the pipeline starts at the truncate step.

With ``--model-ar`` the codec layer is replaced by the AR model's *own* tokens,
produced under teacher forcing: the model reads the ground-truth history and
predicts every next frame in one pass. The flow-matching model then trains on
the kind of input it actually meets at inference — tokens carrying the AR's
error statistics — instead of on ground truth it will never see. Teacher forcing
is what keeps this usable: because the history is never the model's own, the
prediction for frame ``i`` still lines up with frame ``i`` of the real audio, and
the sequence ends where the real one ends, so the frame-wise flow-matching loss
stays meaningful. Free-running decoding would drift in both content and length
and destroy that correspondence.

A fraction ``--clean-fraction`` of the files keeps the ground-truth codec, so the
resulting dataset covers both input distributions the flow-matching model meets:
resynthesis from real audio, and full text-to-speech through the AR.

Output files are ``.npz`` (zlib compressed) containing a single ``latents`` array
of shape ``(num_channels, T_latent)`` of float32 values.

Usage:
    python scripts/precompute_distils.py --audio-dir data/audio --output-dir data/distils
    python scripts/precompute_distils.py --audio-dir data/audio --output-dir data/distils --layers 4
    python scripts/precompute_distils.py --codec-dir data/kanclerz/codecs --output-dir data/kanclerz/distils
    python scripts/precompute_distils.py --codec-dir data/ljspeech/codecs \\
        --output-dir data/ljspeech/distils_ar \\
        --model-ar checkpoints/ljspeech/echo_ar_best.pt --clean-fraction 0.33
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import torch  # noqa: E402
import torchaudio  # noqa: E402
import librosa  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import MimiModel, AutoFeatureExtractor  # noqa: E402
from bluecodec import BlueCodec  # noqa: E402

from echo import config  # noqa: E402
from echo.ar_model import EchoAR  # noqa: E402
from echo.tokenizer import Tokenizer  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif"}

_MIMI_SR = 24000
# Mimi's 12.5 Hz token grid over 24 kHz audio: 1920 samples per codec frame.
_MIMI_FRAME = 1920
_BLUE_SR = 44100
_NUM_CHANNELS = 24


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _save_latents(latents: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, latents=latents.astype(np.float32))


def _load_phoneme_map(csv_path: Path) -> dict[str, str]:
    """Map ``<name>.npz`` to its phoneme string, as written by the phonemizer."""
    mapping: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "|" not in line:
                continue
            npz_name, phonemes = line.split("|", 1)
            mapping[npz_name.strip()] = phonemes.strip()
    return mapping


@torch.no_grad()
def _teacher_forced_codes(
    model: EchoAR,
    codes: torch.Tensor,                                 # (1, layers, T) ground-truth
    text: torch.Tensor,                                  # (1, S) phoneme ids
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    The AR model's own tokens for a sequence, with the history teacher-forced.

    The model reads ``[BOS, frame_0 ... frame_{T-1}]`` — all ground truth — and
    every position predicts the frame after it, so one pass yields a prediction
    for all ``T`` frames aligned one-to-one with the real ones. The output is
    therefore exactly ``T`` frames long, whatever the model thinks.

    Layers within a frame are produced in sequence, not in parallel: with
    intra-frame conditioning on, head ``k`` reads layer ``k - 1`` of the same
    frame, and it must read the layer *this function just predicted*, not the
    ground-truth one. Mixing the two would build frames the AR could never emit
    — precisely the off-distribution inputs this whole path exists to avoid.
    The decoder trunk runs once and only the heads are re-evaluated.

    Special ids are suppressed before the draw: BOS/EOS/pad/mask do not address
    a Mimi codebook entry, and the sequence length is fixed by the ground truth
    anyway, so an EOS here would be meaningless.
    """
    T = codes.shape[2]
    # (1, T, layers) frame-major, as the AR model reads it.
    frames = codes.squeeze(0).T.unsqueeze(0)                       # (1, T, layers)
    bos = torch.full(
        (1, 1, model.NUM_TOKEN_LAYERS), config.prosody_bos,
        dtype=frames.dtype, device=frames.device,
    )
    inputs = torch.cat([bos, frames], dim=1)[:, :T]                # (1, T, layers)

    h, _ = model._trunk(inputs, text=text)                         # (1, T, hidden)

    out_layers: list[torch.Tensor] = []
    cond: torch.Tensor | None = None                               # (1, T) token ids or None
    for k in range(model.NUM_TOKEN_LAYERS):
        logits = model._head(k, h, cond)                           # (1, T, vocab)
        logits[..., config.prosody_bos] = float("-inf")
        logits[..., config.prosody_eos] = float("-inf")
        logits[..., config.prosody_pad] = float("-inf")
        logits[..., config.prosody_mask] = float("-inf")

        if temperature <= 0.0:
            tok = logits.argmax(dim=-1)                            # (1, T)
        else:
            scaled = logits / temperature
            if top_k > 0:
                kth = scaled.topk(min(top_k, scaled.shape[-1]), dim=-1).values[..., -1:]
                scaled = scaled.masked_fill(scaled < kth, float("-inf"))
            probs = scaled.softmax(dim=-1).reshape(-1, scaled.shape[-1])
            tok = torch.multinomial(probs, num_samples=1).reshape(1, T)

        out_layers.append(tok)
        # Head k + 1 conditions on the layer just drawn, exactly as decoding does.
        # ``_head`` embeds the ids itself, so the raw tokens are what it wants.
        cond = tok if model.film is not None else None

    pred = torch.stack(out_layers, dim=2)                          # (1, T, layers)
    return pred.squeeze(0).T.unsqueeze(0)                          # (1, layers, T)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute distillation latents: Mimi → truncate → decode → BlueCodec."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, help="Directory containing input audio files.")
    source.add_argument("--codec-dir", type=str,
                        help="Directory containing precomputed Mimi codec .npz files (a 'codes' "
                             "array of shape (layers, T)). Skips the Mimi encode step.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write .npz latent files to.")
    parser.add_argument("--layers", type=int, default=2, help="Number of Mimi codec layers to keep (default: 16).")
    parser.add_argument("--ext", action="append", help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N audio files (for testing).")

    ar = parser.add_argument_group("AR teacher forcing")
    ar.add_argument("--model-ar", type=str, default=None,
                    help="Path to an EchoAR checkpoint. When given, the codec tokens are "
                         "replaced by the model's own teacher-forced predictions before "
                         "decoding. Requires --codec-dir.")
    ar.add_argument("--phonemes", type=str, default=None,
                    help="phonemes.csv mapping <name>.npz to a phoneme string "
                         "(default: <codec-dir>/../phonemes.csv). Only used with --model-ar.")
    ar.add_argument("--clean-fraction", type=float, default=1.0 / 3.0,
                    help="Fraction of files that keep their ground-truth codec instead of the "
                         "AR's predictions, so the set still covers the resynthesis path "
                         "(default: 1/3).")
    ar.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) takes the argmax, which matches the greedy decoding "
                         "used at inference. A positive value samples instead, for generating "
                         "several different variants of the same utterance.")
    ar.add_argument("--top-k", type=int, default=0,
                    help="Restrict sampling to the k most likely ids (0 = no restriction). "
                         "Only meaningful with --temperature > 0.")
    ar.add_argument("--seed", type=int, default=42,
                    help="Seeds both the clean/AR split and sampling (default: 42).")
    args = parser.parse_args()

    use_ar = args.model_ar is not None
    if use_ar and args.codec_dir is None:
        parser.error("--model-ar requires --codec-dir: teacher forcing needs the ground-truth codec")
    if not 0.0 <= args.clean_fraction <= 1.0:
        parser.error(f"--clean-fraction must be in [0, 1], got {args.clean_fraction}")
    if use_ar and args.layers != EchoAR.NUM_TOKEN_LAYERS:
        parser.error(
            f"--model-ar predicts {EchoAR.NUM_TOKEN_LAYERS} token layers, "
            f"but --layers is {args.layers}"
        )

    print_header("Distillation Latents - Precompute (Mimi → BlueCodec)")
    print_separator()

    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    # --- Device -------------------------------------------------------------
    device = _select_device()
    print_section("Device")
    print_info("Selected", str(device), Colors.OKCYAN)

    # --- Discover input files -----------------------------------------------
    from_codecs = args.codec_dir is not None
    input_dir = Path(args.codec_dir if from_codecs else args.audio_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        kind = "Codec" if from_codecs else "Audio"
        print_error(f"{kind} directory not found: {input_dir}")
        sys.exit(1)

    if from_codecs:
        files = sorted(p for p in input_dir.rglob("*.npz") if p.is_file())
    else:
        exts = set(_AUDIO_EXTS)
        if args.ext:
            exts.update(e.lower() for e in args.ext)

        files = sorted(
            p for p in input_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in exts
        )
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        kind = "codec .npz files" if from_codecs else "audio files"
        print_error(f"No {kind} found in {input_dir}")
        sys.exit(1)

    print_section("Input")
    print_info("Codec dir" if from_codecs else "Audio dir", str(input_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Mimi layers", str(args.layers))
    print_info("Files found", str(len(files)))

    # --- Load the AR model (optional) ----------------------------------------
    ar_model: EchoAR | None = None
    phoneme_map: dict[str, str] = {}
    tokenizer: Tokenizer | None = None
    if use_ar:
        print_section("Loading EchoAR model")
        t_model = time.perf_counter()
        ckpt_path = Path(args.model_ar)
        if not ckpt_path.is_file():
            print_error(f"AR checkpoint not found: {ckpt_path}")
            sys.exit(1)
        phonemes_csv = Path(args.phonemes) if args.phonemes else input_dir.parent / "phonemes.csv"
        if not phonemes_csv.is_file():
            print_error(f"Phoneme CSV not found: {phonemes_csv} (pass --phonemes)")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ar_model = EchoAR().to(device)
        ar_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
        ar_model.eval()
        tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
        phoneme_map = _load_phoneme_map(phonemes_csv)

        print_info("Checkpoint", str(ckpt_path), Colors.OKCYAN)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            print_info("Trained", f"epoch {ckpt['epoch']}, val_loss {ckpt.get('val_loss', float('nan')):.4f}")
        print_info("Phonemes", f"{phonemes_csv} ({len(phoneme_map)} entries)")
        print_info("Clean fraction", f"{args.clean_fraction:.3f}")
        print_info("Sampling",
                   "greedy (argmax)" if args.temperature <= 0.0
                   else f"temperature {args.temperature:g}"
                        + (f", top-k {args.top_k}" if args.top_k > 0 else ""))
        print_info("Model load time", f"{time.perf_counter() - t_model:.3f}s", Colors.OKCYAN)

    # --- Load Mimi -----------------------------------------------------------
    print_section("Loading Mimi model")
    t_model = time.perf_counter()
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    feature_extractor = None
    if not from_codecs:
        # Only the encode path needs it; reading codecs from disk does not.
        feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")
        mimi_sr = feature_extractor.sampling_rate
        assert mimi_sr == _MIMI_SR, f"Expected Mimi sample rate {_MIMI_SR}, got {mimi_sr}"
    mimi_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{_MIMI_SR} Hz")
    print_info("Mode", "decode only (codecs read from disk)" if from_codecs else "encode → decode")
    print_info("Model load time", f"{mimi_time:.3f}s", Colors.OKCYAN)

    # --- Load BlueCodec ------------------------------------------------------
    print_section("Loading BlueCodec model")
    t_model = time.perf_counter()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    blue_time = time.perf_counter() - t_model
    print_info("Sample rate", f"{_BLUE_SR} Hz")
    print_info("Latent channels", str(_NUM_CHANNELS))
    print_info("Model load time", f"{blue_time:.3f}s", Colors.OKCYAN)

    # --- Process one file at a time ------------------------------------------
    print_section("Processing")
    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0
    n_clean, n_ar = 0, 0
    disagree_sum, disagree_files = 0.0, 0

    if use_ar:
        torch.manual_seed(args.seed)

    progress = tqdm(files, desc="Processing", unit="file")
    for src in progress:
        try:
            if from_codecs:
                # Step 1+2 replaced: read the codec and truncate it to --layers.
                with np.load(src) as data:
                    code_arr = data["codes"][: args.layers].astype(np.int64)
                codes = torch.from_numpy(code_arr).unsqueeze(0).to(device)  # (1, layers, T)
                orig_len = code_arr.shape[1] * _MIMI_FRAME

                if use_ar:
                    # Keyed on the file name rather than on iteration order, so
                    # the clean/AR split survives --limit, reordering and reruns.
                    rel_name = src.relative_to(input_dir).name
                    keep_clean = random.Random(f"{args.seed}:{rel_name}").random() < args.clean_fraction
                    phonemes = phoneme_map.get(rel_name)
                    if phonemes is None:
                        raise KeyError(f"no phoneme entry for {rel_name}")

                    if keep_clean:
                        n_clean += 1
                    else:
                        text = torch.tensor(
                            tokenizer.tokenize(phonemes), dtype=torch.long, device=device
                        ).unsqueeze(0)                              # (1, S)
                        pred = _teacher_forced_codes(
                            ar_model, codes, text,
                            temperature=args.temperature, top_k=args.top_k,
                        )                                           # (1, layers, T)
                        disagree_sum += float((pred != codes).float().mean())
                        disagree_files += 1
                        codes = pred
                        n_ar += 1
            else:
                # Step 1: Load + resample to 24 kHz mono (Mimi input).
                audio, _ = librosa.load(str(src), sr=_MIMI_SR, mono=True)
                orig_len = len(audio)

                # Step 2: Mimi encode.
                inputs = feature_extractor(
                    raw_audio=[audio.astype(np.float32).tolist()],
                    sampling_rate=_MIMI_SR,
                    return_tensors="pt",
                )
                input_values = inputs["input_values"].to(device)

                with torch.no_grad():
                    enc = mimi.encode(input_values)
                codes = enc.audio_codes[:, : args.layers, :]      # (1, layers, T_codec)

            # Step 3: Mimi decode back to audio.
            with torch.no_grad():
                dec_out = mimi.decode(codes)
            # dec_out returns (audio_values, ...) or MimiDecoderOutput
            if isinstance(dec_out, tuple):
                decoded_audio = dec_out[0]
            else:
                decoded_audio = dec_out.audio_values
            decoded_audio = decoded_audio.squeeze(1)              # (1, T_audio)

            # Step 4: Resample the decoded audio to 44.1 kHz and encode with BlueCodec.
            audio_mimi = decoded_audio[:, :orig_len]              # (1, T)
            audio_441 = torchaudio.functional.resample(audio_mimi, _MIMI_SR, _BLUE_SR)

            with torch.no_grad():
                latents = blue.encode(audio_441.to(device))
            latents_np = latents.detach().cpu().numpy().astype(np.float32)
            latents_np = latents_np.squeeze(0)                    # (1, C, T) -> (C, T)

            rel = src.relative_to(input_dir)
            _save_latents(latents_np, output_dir / rel.with_suffix(".npz"))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print_error(f"Failed to process {src.name}: {e}")
            n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    elapsed = time.perf_counter() - t_total

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Files processed", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    if use_ar:
        print_info("Ground-truth codecs", f"{n_clean} ({n_clean / max(1, n_clean + n_ar):.1%})",
                   Colors.OKCYAN)
        print_info("AR teacher-forced", f"{n_ar} ({n_ar / max(1, n_clean + n_ar):.1%})",
                   Colors.OKCYAN)
        if disagree_files:
            print_info("Mean token disagreement", f"{disagree_sum / disagree_files:.3f}",
                       Colors.OKCYAN)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()