#!/usr/bin/env python3
"""Precompute distillation latents: Mimi encode → truncate → decode → BlueCodec encode.

Pipeline: audio -> Mimi codec (first --layers layers) -> decoded audio -> BlueCodec latents.

With ``--codec-dir`` the Mimi encode step is skipped: the codecs are read from
``.npz`` files (a ``codes`` array of shape ``(layers, T_codec)``, as written by
``precompute_codecs.py``) and the pipeline starts at the truncate step.

With ``--model-ar`` the codec layer is replaced by the AR model's *own* tokens,
so the flow-matching model trains on the kind of input it actually meets at
inference — AR output, drift included — instead of on ground truth it will
never see.

Those tokens are drawn by **block-wise scheduled sampling** rather than by free
running. The model is fed its own tokens inside each block of ``--block-size``
frames and re-synchronised to the ground-truth history at every block boundary.
Frame i therefore still describes the same instant as frame i of the target, so
the pair stays aligned, while the tokens carry genuine free-running drift over
block-length stretches.

That alignment is the whole point. Free running produces a sequence with a
length *and a timeline* of its own: clipping it to the right length lines up the
ends but nothing in between, and the resulting pairs correlate ~0.13 with their
targets against ~0.64 for the ground-truth path. Flow matching on pairs that
loose cannot learn a transport at all — it falls back on predicting the mean of
every plausible target, which is exactly the blur the AR-input training was
supposed to fix. Block-wise sampling keeps ~0.40 at k=16 while sitting ~5x
closer to true free-running output than clean data does.

``--block-size 0`` restores the old free-running behaviour, misalignment
included; it is kept for comparison, not for training data.

The ground-truth codec fixes both the timeline and the length, so ``--model-ar``
requires ``--codec-dir``. Generation runs batched over ``--batch-size`` files.

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
    python scripts/precompute_distils.py --codec-dir data/ljspeech/codecs \\
        --output-dir data/ljspeech/distils_ar \\
        --model-ar checkpoints/ljspeech/echo_ar_best.pt \\
        --block-size 8 --batch-size 16
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
def _generated_codes(
    model: EchoAR,
    text: torch.Tensor,                                  # (1, S) phoneme ids
    max_frames: int,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    The AR model's own tokens for an utterance, decoded free-running from text.

    Nothing of the ground-truth codec enters here: the model starts at BOS and
    feeds on its own frames until it emits EOS or hits ``max_frames``. That is
    exactly the input the flow-matching model meets at inference, drift and all.

    The returned length is whatever the model chose, so it will rarely match the
    real utterance; the caller re-aligns it (zero-pad or clip) afterwards.
    """
    gen = model.generate(
        text,
        max_frames=max_frames,
        temperature=temperature,
        top_k=top_k,
    )                                                              # (1, T_gen, layers)
    return gen.transpose(1, 2).contiguous()                        # (1, layers, T_gen)


def _draw(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """Token ids from ``(..., vocab)`` logits, with the special ids suppressed.

    Unlike free running, block-wise sampling never decides the length — the
    ground-truth codec does — so EOS is suppressed along with the other special
    ids. Everything that survives addresses a real codebook entry, which is what
    Mimi's decoder needs.
    """
    for special in (config.prosody_bos, config.prosody_pad,
                    config.prosody_eos, config.prosody_mask):
        logits[..., special] = float("-inf")

    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    if top_k > 0:
        kth = logits.topk(min(top_k, logits.shape[-1]), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = logits.softmax(dim=-1)
    flat = torch.multinomial(probs.reshape(-1, probs.shape[-1]), num_samples=1)
    return flat.reshape(probs.shape[:-1])


@torch.no_grad()
def _block_sampled_codes(
    model: EchoAR,
    codes: torch.Tensor,                                 # (B, layers, T) ground truth, padded
    valid: torch.Tensor,                                 # (B, T) bool, True on real frames
    text: torch.Tensor,                                  # (B, S) phoneme ids, padded
    text_mask: torch.Tensor,                             # (B, S) bool
    block: int,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """The AR's own tokens, re-synchronised to the truth every ``block`` frames.

    Position i reads the model's own tokens for the frames since the last block
    boundary and the ground-truth ones before it, so each generated frame still
    stands for the same instant as frame i of the target: the length is exact by
    construction and no re-alignment is needed downstream.

    Rather than roll out T/block prefixes one frame at a time, this runs
    ``block`` full passes over the whole batch. After pass p, every position is
    conditioned on its own predictions for up to p preceding frames, so after
    ``block`` passes the within-block history is entirely the model's own —
    the same fixed point the sequential rollout reaches, at a fraction of the
    cost and batched across files.
    """
    gt = codes.transpose(1, 2)                                     # (B, T, layers)
    T = gt.shape[1]
    boundary = (torch.arange(T, device=gt.device) % block == 0)[None, :, None]

    gen = gt.clone()
    for _ in range(block):
        # History: the model's own tokens inside a block, the truth at its start.
        history = torch.where(boundary, gt, gen)                   # (B, T, layers)
        inputs = torch.full_like(gt, config.prosody_bos)
        inputs[:, 1:] = history[:, :-1]                            # frame i reads frame i - 1

        # Intra-frame conditioning reads the lower layers of the frame being
        # predicted, which at inference are the model's own — so they come from
        # the previous pass, not from the truth.
        cond = gen[..., :-1] if model.film is not None else None
        logits = model(inputs, text, valid, text_mask, cond_tokens=cond)
        gen = _draw(logits, temperature, top_k)                    # (B, T, layers)

        # Padded frames are never read (causal attention plus the padding mask)
        # and never written out; keeping the truth there avoids feeding the next
        # pass junk through the one path that does cross frames.
        gen = torch.where(valid[..., None], gen, gt)

    return gen.transpose(1, 2).contiguous()                        # (B, layers, T)


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

    ar = parser.add_argument_group("AR generation")
    ar.add_argument("--model-ar", type=str, default=None,
                    help="Path to an EchoAR checkpoint. When given, the codec tokens are "
                         "replaced by the model's own, drawn by block-wise scheduled "
                         "sampling. Requires --codec-dir.")
    ar.add_argument("--block-size", type=int, default=16, metavar="K",
                    help="Frames the model runs on its own history before being "
                         "re-synchronised to the truth (default: 16). Smaller keeps the pair "
                         "better aligned, larger looks more like free-running output; 8-16 is "
                         "the useful range. 0 selects free running, which does not preserve "
                         "the timeline and is kept only for comparison.")
    ar.add_argument("--batch-size", type=int, default=8, metavar="N",
                    help="Files generated at once (default: 8). Block-wise sampling costs "
                         "--block-size forward passes per batch, so this is the main speed knob.")
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
        parser.error("--model-ar requires --codec-dir: the ground-truth codec fixes the target length")
    if not 0.0 <= args.clean_fraction <= 1.0:
        parser.error(f"--clean-fraction must be in [0, 1], got {args.clean_fraction}")
    if use_ar and args.layers != EchoAR.NUM_TOKEN_LAYERS:
        parser.error(
            f"--model-ar predicts {EchoAR.NUM_TOKEN_LAYERS} token layers, "
            f"but --layers is {args.layers}"
        )
    if args.block_size < 0:
        parser.error(f"--block-size must be >= 0 (0 selects free running), got {args.block_size}")
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")

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
        ar_model.load_weights(ckpt["model"] if "model" in ckpt else ckpt)
        ar_model.eval()
        tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
        phoneme_map = _load_phoneme_map(phonemes_csv)

        print_info("Checkpoint", str(ckpt_path), Colors.OKCYAN)
        if isinstance(ckpt, dict) and "epoch" in ckpt:
            print_info("Trained", f"epoch {ckpt['epoch']}, val_loss {ckpt.get('val_loss', float('nan')):.4f}")
        print_info("Phonemes", f"{phonemes_csv} ({len(phoneme_map)} entries)")
        print_info("Clean fraction", f"{args.clean_fraction:.3f}")
        if args.block_size > 0:
            print_info("Token source",
                       f"block-wise scheduled sampling, k={args.block_size} "
                       f"({args.block_size} passes per batch of {args.batch_size})",
                       Colors.OKCYAN)
        else:
            print_info("Token source",
                       "free running — the generated timeline does not match the target's, "
                       "so these pairs are not suitable for training",
                       Colors.WARNING)
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
    len_ratio_sum, len_ratio_files = 0.0, 0
    tok_match_sum, tok_match_files = 0.0, 0

    if use_ar:
        torch.manual_seed(args.seed)

    # Block-wise sampling is the only stage that batches: it is `block` forward
    # passes over whole utterances, and everything after it is per-file work at
    # a length that differs from file to file.
    batched_ar = use_ar and args.block_size > 0
    chunk_size = args.batch_size if batched_ar else 1

    progress = tqdm(total=len(files), desc="Processing", unit="file")
    for start in range(0, len(files), chunk_size):
        chunk = files[start : start + chunk_size]

        # --- Load the codecs (or encode the audio) for the whole chunk -------
        prepared: list[tuple[Path, torch.Tensor, int]] = []       # src, codes, orig_len
        ar_jobs: list[int] = []                                   # indices into `prepared`
        ar_texts: list[torch.Tensor] = []
        for src in chunk:
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
                        keep_clean = (random.Random(f"{args.seed}:{rel_name}").random()
                                      < args.clean_fraction)
                        phonemes = phoneme_map.get(rel_name)
                        if phonemes is None:
                            raise KeyError(f"no phoneme entry for {rel_name}")

                        if keep_clean:
                            n_clean += 1
                        else:
                            text = torch.tensor(
                                tokenizer.tokenize(phonemes), dtype=torch.long, device=device
                            )                                       # (S,)
                            if batched_ar:
                                ar_jobs.append(len(prepared))
                                ar_texts.append(text)
                            else:
                                T_ref = codes.shape[2]
                                pred = _generated_codes(
                                    ar_model, text.unsqueeze(0),
                                    # A generous ceiling: the model is free to run
                                    # long, the excess is clipped away below anyway.
                                    max_frames=int(T_ref * 2) + 50,
                                    temperature=args.temperature, top_k=args.top_k,
                                )                                   # (1, layers, T_gen)
                                len_ratio_sum += pred.shape[2] / max(1, T_ref)
                                len_ratio_files += 1
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
                    codes = enc.audio_codes[:, : args.layers, :]  # (1, layers, T_codec)

                prepared.append((src, codes, orig_len))
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to read {src.name}: {e}")
                n_fail += 1
                progress.update(1)

        # --- Replace the AR files' tokens, all of them in one batch ----------
        if ar_jobs:
            try:
                lengths = [prepared[i][1].shape[2] for i in ar_jobs]
                text_lengths = [t.shape[0] for t in ar_texts]
                T_max, S_max = max(lengths), max(text_lengths)
                B, L = len(ar_jobs), args.layers

                gt = torch.full((B, L, T_max), config.prosody_pad,
                                dtype=torch.long, device=device)
                valid = torch.zeros(B, T_max, dtype=torch.bool, device=device)
                text = torch.full((B, S_max), config.text_pad, dtype=torch.long, device=device)
                text_mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)
                for b, (idx, tok_ids) in enumerate(zip(ar_jobs, ar_texts)):
                    t_len = lengths[b]
                    gt[b, :, :t_len] = prepared[idx][1][0]
                    valid[b, :t_len] = True
                    text[b, : text_lengths[b]] = tok_ids
                    text_mask[b, : text_lengths[b]] = True

                gen = _block_sampled_codes(
                    ar_model, gt, valid, text, text_mask, args.block_size,
                    temperature=args.temperature, top_k=args.top_k,
                )                                                  # (B, layers, T_max)

                for b, idx in enumerate(ar_jobs):
                    src, codes, orig_len = prepared[idx]
                    kept = gen[b : b + 1, :, : lengths[b]].contiguous()
                    tok_match_sum += float((kept == codes).float().mean())
                    tok_match_files += 1
                    prepared[idx] = (src, kept, orig_len)
            except Exception as e:  # noqa: BLE001
                # Falling back to the ground-truth codec would silently mix
                # distributions, so the batch is dropped instead.
                print_error(f"Failed to generate tokens for {len(ar_jobs)} file(s): {e}")
                for idx in sorted(ar_jobs, reverse=True):
                    prepared.pop(idx)
                    n_fail += 1
                    n_ar -= 1
                    progress.update(1)

        for src, codes, orig_len in prepared:
            try:
                # Step 3: Mimi decode back to audio.
                if codes.shape[2] == 0:
                    # The AR emitted EOS immediately; nothing to decode, and the
                    # zero-padding below turns it into pure silence of the right length.
                    decoded_audio = torch.zeros(1, 0, device=device)
                else:
                    with torch.no_grad():
                        dec_out = mimi.decode(codes)
                    # dec_out returns (audio_values, ...) or MimiDecoderOutput
                    if isinstance(dec_out, tuple):
                        decoded_audio = dec_out[0]
                    else:
                        decoded_audio = dec_out.audio_values
                    decoded_audio = decoded_audio.squeeze(1)      # (1, T_audio)

                # Step 4: Force the decoded audio onto the real utterance's length.
                # Block-wise sampling already returns exactly the ground-truth
                # frame count, so this is a no-op there; it still matters for the
                # free-running path, whose length is its own.
                audio_mimi = decoded_audio[:, :orig_len]          # (1, T)
                if audio_mimi.shape[1] < orig_len:
                    audio_mimi = torch.nn.functional.pad(
                        audio_mimi, (0, orig_len - audio_mimi.shape[1])
                    )
                audio_441 = torchaudio.functional.resample(audio_mimi, _MIMI_SR, _BLUE_SR)

                with torch.no_grad():
                    latents = blue.encode(audio_441.to(device))
                latents_np = latents.detach().cpu().numpy().astype(np.float32)
                latents_np = latents_np.squeeze(0)                # (1, C, T) -> (C, T)

                rel = src.relative_to(input_dir)
                _save_latents(latents_np, output_dir / rel.with_suffix(".npz"))
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                print_error(f"Failed to process {src.name}: {e}")
                n_fail += 1

            progress.update(1)
            progress.set_postfix(ok=n_ok, fail=n_fail)

    progress.close()

    elapsed = time.perf_counter() - t_total

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Files processed", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    if use_ar:
        print_info("Ground-truth codecs", f"{n_clean} ({n_clean / max(1, n_clean + n_ar):.1%})",
                   Colors.OKCYAN)
        print_info("AR generated", f"{n_ar} ({n_ar / max(1, n_clean + n_ar):.1%})",
                   Colors.OKCYAN)
        if len_ratio_files:
            print_info("Mean generated/real length",
                       f"{len_ratio_sum / len_ratio_files:.3f}", Colors.OKCYAN)
        if tok_match_files:
            # How much of the ground truth survived: 1.0 would mean the sampling
            # changed nothing, low values mean plenty of drift to learn from.
            print_info("Token agreement with truth",
                       f"{tok_match_sum / tok_match_files:.3f} "
                       f"(lengths preserved exactly)", Colors.OKCYAN)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()