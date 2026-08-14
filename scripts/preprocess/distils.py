#!/usr/bin/env python3
"""Precompute distillation latents: Mimi codec -> decoded audio -> BlueCodec latents.

With --codec-dir the Mimi encode step is skipped and the codecs are read from
disk instead.

With --model-ar the codec tokens are replaced by the AR model's own, drawn by
block-wise scheduled sampling: the model runs on its own history inside each
block of --block-size frames, then re-synchronises to the truth at the boundary.
That keeps generated frame i aligned with target frame i, which free running does
not — and flow matching on misaligned pairs learns a blur instead of a transport.
--block-size 0 restores free running, kept only for comparison.

Usage:
    python scripts/preprocess/distils.py --audio-dir data/audio --output-dir data/distils
    python scripts/preprocess/distils.py --codec-dir data/ljspeech/codecs \\
        --output-dir data/ljspeech/distils_ar \\
        --model-ar checkpoints/ljspeech/echo_ar_best.pt --clean-fraction 0.33
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
import time
import warnings

import librosa
import numpy as np
import torch
import torchaudio
from bluecodec import BlueCodec
from tqdm import tqdm
from transformers import AutoFeatureExtractor, MimiModel

from __common__ import (
    BLUE_SR,
    MIMI_FRAME,
    MIMI_SR,
    decode_mimi,
    find_audio_files,
    load_checkpoint,
    load_pairs_csv,
    load_tokenizer,
    save_npz,
    select_device,
)
from __style__ import (
    Colors, print_error, print_header, print_info, print_section,
    print_separator, print_success,
)

from echo import config
from echo.ar_model import EchoAR


# ------------------
# AR token sampling
# ------------------

def _draw(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    """
    Token ids from `(..., vocab)` logits, with every special id suppressed.

    NOTE: the ground-truth codec fixes the length here, so EOS is suppressed too
    and everything drawn addresses a real codebook entry.
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

    return torch.multinomial(
        probs.reshape(-1, probs.shape[-1]), num_samples=1
    ).reshape(probs.shape[:-1])


@torch.no_grad()
def _block_sampled_codes(
    model: EchoAR,
    codes: torch.Tensor,                                 # (B, layers, T) truth, padded
    valid: torch.Tensor,                                 # (B, T) bool
    text: torch.Tensor,                                  # (B, S) phoneme ids, padded
    text_mask: torch.Tensor,                             # (B, S) bool
    block: int,
    temperature: float = 0.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    The AR's own tokens, re-synchronised to the truth every `block` frames.

    NOTE: rather than roll out T/block prefixes one frame at a time, this runs
    `block` full passes over the batch. After pass p every position is
    conditioned on its own predictions for up to p preceding frames, reaching the
    same fixed point as a sequential rollout at a fraction of the cost.
    """

    gt = codes.transpose(1, 2)                                       # (B, T, layers)
    boundary = (torch.arange(gt.shape[1], device=gt.device) % block == 0)[None, :, None]

    gen = gt.clone()
    for _ in range(block):
        # The model's own tokens inside a block, the truth at its start.
        history = torch.where(boundary, gt, gen)                     # (B, T, layers)
        inputs = torch.full_like(gt, config.prosody_bos)
        inputs[:, 1:] = history[:, :-1]                              # frame i reads frame i - 1

        # Intra-frame conditioning reads the frame being predicted, so it comes
        # from the previous pass rather than from the truth.
        cond = gen[..., :-1] if model.predictor.uses_cond else None
        gen = _draw(model(inputs, text, valid, text_mask, cond_tokens=cond),
                    temperature, top_k)                              # (B, T, layers)

        # Padded frames are never read or written out; keeping the truth there
        # avoids feeding the next pass junk through the one time-crossing path.
        gen = torch.where(valid[..., None], gen, gt)

    return gen.transpose(1, 2).contiguous()                          # (B, layers, T)


def _batch_ar_inputs(
    codes: list[torch.Tensor],                           # each (1, layers, T)
    texts: list[torch.Tensor],                           # each (S,)
    layers: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Right-pad a chunk of utterances into (codes, valid, text, text_mask, lengths)."""
    lengths = [c.shape[2] for c in codes]
    text_lengths = [t.shape[0] for t in texts]
    B, T_max, S_max = len(codes), max(lengths), max(text_lengths)

    gt = torch.full((B, layers, T_max), config.prosody_pad, dtype=torch.long, device=device)
    valid = torch.zeros(B, T_max, dtype=torch.bool, device=device)
    text = torch.full((B, S_max), config.text_pad, dtype=torch.long, device=device)
    text_mask = torch.zeros(B, S_max, dtype=torch.bool, device=device)

    for b, (code, tok_ids) in enumerate(zip(codes, texts)):
        gt[b, :, :lengths[b]] = code[0]
        valid[b, :lengths[b]] = True
        text[b, :text_lengths[b]] = tok_ids
        text_mask[b, :text_lengths[b]] = True

    return gt, valid, text, text_mask, lengths


# -----------------
# Codec conversion
# -----------------

def _encode_audio(src: Path, mimi, feature_extractor, layers: int, device) -> tuple[torch.Tensor, int]:
    """Audio file -> (codec tokens, original sample count)."""
    audio, _ = librosa.load(str(src), sr=MIMI_SR, mono=True)
    inputs = feature_extractor(
        raw_audio=[audio.astype(np.float32).tolist()],
        sampling_rate=MIMI_SR,
        return_tensors="pt",
    )
    with torch.no_grad():
        enc = mimi.encode(inputs["input_values"].to(device))

    return enc.audio_codes[:, :layers, :], len(audio)                # (1, layers, T_codec)


def _read_codes(src: Path, layers: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Codec .npz -> (codec tokens, the audio length they cover)."""
    with np.load(src) as data:
        arr = data["codes"][:layers].astype(np.int64)

    return torch.from_numpy(arr).unsqueeze(0).to(device), arr.shape[1] * MIMI_FRAME


def _to_distil(codes: torch.Tensor, orig_len: int, mimi, blue, device) -> np.ndarray:
    """Mimi-decode, force onto the true length, resample, BlueCodec-encode."""
    if codes.shape[2] == 0:
        # The AR emitted EOS immediately; the padding below makes it silence.
        audio = torch.zeros(1, 0, device=device)
    else:
        with torch.no_grad():
            audio = decode_mimi(mimi, codes)                         # (1, T_audio)

    # Block-wise sampling already returns the ground-truth frame count, so this
    # only bites on the free-running path, whose length is its own.
    audio = audio[:, :orig_len]
    if audio.shape[1] < orig_len:
        audio = torch.nn.functional.pad(audio, (0, orig_len - audio.shape[1]))

    with torch.no_grad():
        latents = blue.encode(
            torchaudio.functional.resample(audio, MIMI_SR, BLUE_SR).to(device)
        )

    return latents.detach().cpu().numpy().astype(np.float32).squeeze(0)   # (C, T)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute distillation latents: Mimi -> decode -> BlueCodec."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=str, help="Directory of input audio files.")
    source.add_argument("--codec-dir", type=str,
                        help="Directory of precomputed Mimi codec .npz files, skipping "
                             "the Mimi encode step.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write .npz latent files to.")
    parser.add_argument("--layers", type=int, default=EchoAR.NUM_TOKEN_LAYERS,
                        help=f"Mimi codec layers to keep (default: "
                             f"{EchoAR.NUM_TOKEN_LAYERS}, what EchoAR predicts).")
    parser.add_argument("--ext", action="append",
                        help="Additional audio extension to include (may be repeated).")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N input files.")

    ar = parser.add_argument_group("AR generation")
    ar.add_argument("--model-ar", type=str, default=None,
                    help="EchoAR checkpoint. Replaces the codec tokens with the model's "
                         "own, by block-wise scheduled sampling. Requires --codec-dir.")
    ar.add_argument("--block-size", type=int, default=16, metavar="K",
                    help="Frames the model runs on its own history before re-synchronising "
                         "(default: 16; 8-16 is the useful range, 0 selects free running).")
    ar.add_argument("--batch-size", type=int, default=8, metavar="N",
                    help="Files generated at once (default: 8). Sampling costs --block-size "
                         "passes per batch, so this is the main speed knob.")
    ar.add_argument("--phonemes", type=str, default=None,
                    help="phonemes.csv mapping <name>.npz to phonemes "
                         "(default: <codec-dir>/../phonemes.csv).")
    ar.add_argument("--clean-fraction", type=float, default=1.0 / 3.0,
                    help="Fraction of files keeping their ground-truth codec, so the set "
                         "still covers the resynthesis path (default: 1/3).")
    ar.add_argument("--temperature", type=float, default=0.0,
                    help="0 (default) takes the argmax, matching inference.")
    ar.add_argument("--top-k", type=int, default=0,
                    help="Restrict sampling to the k most likely ids (0 = off).")
    ar.add_argument("--seed", type=int, default=42,
                    help="Seeds both the clean/AR split and sampling (default: 42).")
    args = parser.parse_args()

    if args.model_ar is not None:
        if args.codec_dir is None:
            parser.error("--model-ar requires --codec-dir: the ground-truth codec fixes "
                         "the target length")
        if args.layers != EchoAR.NUM_TOKEN_LAYERS:
            parser.error(f"--model-ar predicts {EchoAR.NUM_TOKEN_LAYERS} token layers, "
                         f"but --layers is {args.layers}")
    if not 0.0 <= args.clean_fraction <= 1.0:
        parser.error(f"--clean-fraction must be in [0, 1], got {args.clean_fraction}")
    if args.block_size < 0:
        parser.error(f"--block-size must be >= 0, got {args.block_size}")
    if args.batch_size < 1:
        parser.error(f"--batch-size must be >= 1, got {args.batch_size}")

    return args


def main() -> None:
    args = _parse_args()
    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    use_ar = args.model_ar is not None
    from_codecs = args.codec_dir is not None
    input_dir = Path(args.codec_dir if from_codecs else args.audio_dir)
    output_dir = Path(args.output_dir)
    device = select_device()

    print_header("Distillation Latents - Precompute (Mimi → BlueCodec)")
    print_separator()

    # --- Inputs -------------------------------------------------------------
    if not input_dir.is_dir():
        print_error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    if from_codecs:
        files = sorted(p for p in input_dir.rglob("*.npz") if p.is_file())[: args.limit]
    else:
        files = find_audio_files(input_dir, args.ext, args.limit)
    if not files:
        print_error(f"No input files found in {input_dir}")
        sys.exit(1)

    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Codec dir" if from_codecs else "Audio dir", str(input_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Mimi layers", str(args.layers))
    print_info("Files found", str(len(files)))

    # --- Models -------------------------------------------------------------
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    # Only the encode path needs the feature extractor.
    feature_extractor = None if from_codecs else AutoFeatureExtractor.from_pretrained("kyutai/mimi")
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))

    ar_model, tokenizer, phoneme_map = None, None, {}
    if use_ar:
        ckpt_path = Path(args.model_ar)
        phonemes_csv = Path(args.phonemes) if args.phonemes else input_dir.parent / "phonemes.csv"
        for label, path in (("AR checkpoint", ckpt_path), ("Phoneme CSV", phonemes_csv)):
            if not path.is_file():
                print_error(f"{label} not found: {path}")
                sys.exit(1)

        ar_model = EchoAR().to(device)
        ckpt = load_checkpoint(ar_model, ckpt_path, device, "EchoAR")
        ar_model.eval()
        tokenizer = load_tokenizer()
        phoneme_map = dict(load_pairs_csv(phonemes_csv))
        torch.manual_seed(args.seed)

        print_info("AR checkpoint", f"{ckpt_path} (epoch {ckpt.get('epoch', '?')})", Colors.OKCYAN)
        print_info("Phonemes", f"{phonemes_csv} ({len(phoneme_map)} entries)")
        print_info("Clean fraction", f"{args.clean_fraction:.3f}")
        print_info("Token source",
                   f"block-wise scheduled sampling, k={args.block_size}" if args.block_size > 0
                   else "free running — the timeline does not match the target's, so these "
                        "pairs are not suitable for training",
                   Colors.OKCYAN if args.block_size > 0 else Colors.WARNING)
        print_info("Sampling",
                   "greedy (argmax)" if args.temperature <= 0.0
                   else f"temperature {args.temperature:g}"
                        + (f", top-k {args.top_k}" if args.top_k > 0 else ""))

    # --- Process ------------------------------------------------------------
    print_section("Processing")
    t_total = time.perf_counter()
    n_ok = n_fail = n_clean = n_ar = 0
    len_ratios: list[float] = []
    tok_matches: list[float] = []

    # Block-wise sampling is the only stage that batches; everything after it is
    # per-file work at a length that differs from file to file.
    batched_ar = use_ar and args.block_size > 0
    chunk_size = args.batch_size if batched_ar else 1

    progress = tqdm(total=len(files), desc="Processing", unit="file")
    for start in range(0, len(files), chunk_size):
        prepared: list[tuple[Path, torch.Tensor, int]] = []          # src, codes, orig_len
        ar_jobs: list[int] = []                                      # indices into `prepared`
        ar_texts: list[torch.Tensor] = []

        # --- Read the codecs (or encode the audio) for the whole chunk -------
        for src in files[start: start + chunk_size]:
            try:
                if not from_codecs:
                    codes, orig_len = _encode_audio(
                        src, mimi, feature_extractor, args.layers, device
                    )
                    prepared.append((src, codes, orig_len))
                    continue

                codes, orig_len = _read_codes(src, args.layers, device)
                if use_ar:
                    # Keyed on the file name rather than iteration order, so the
                    # clean/AR split survives --limit, reordering and reruns.
                    name = src.relative_to(input_dir).name
                    phonemes = phoneme_map.get(name)
                    if phonemes is None:
                        raise KeyError(f"no phoneme entry for {name}")

                    if random.Random(f"{args.seed}:{name}").random() < args.clean_fraction:
                        n_clean += 1
                    else:
                        text = torch.tensor(
                            tokenizer.tokenize(phonemes), dtype=torch.long, device=device
                        )
                        if batched_ar:
                            ar_jobs.append(len(prepared))
                            ar_texts.append(text)
                        else:
                            # A generous ceiling: the excess is clipped away below.
                            T_ref = codes.shape[2]
                            codes = ar_model.generate(
                                text.unsqueeze(0), max_frames=int(T_ref * 2) + 50,
                                temperature=args.temperature, top_k=args.top_k,
                            ).transpose(1, 2).contiguous()
                            len_ratios.append(codes.shape[2] / max(1, T_ref))
                        n_ar += 1

                prepared.append((src, codes, orig_len))
            except Exception as e:                                   # noqa: BLE001
                print_error(f"Failed to read {src.name}: {e}")
                n_fail += 1
                progress.update(1)

        # --- Replace the AR files' tokens, all of them in one batch ----------
        if ar_jobs:
            try:
                gt, valid, text, text_mask, lengths = _batch_ar_inputs(
                    [prepared[i][1] for i in ar_jobs], ar_texts, args.layers, device
                )
                gen = _block_sampled_codes(
                    ar_model, gt, valid, text, text_mask, args.block_size,
                    temperature=args.temperature, top_k=args.top_k,
                )
                for b, idx in enumerate(ar_jobs):
                    src, codes, orig_len = prepared[idx]
                    kept = gen[b: b + 1, :, : lengths[b]].contiguous()
                    tok_matches.append(float((kept == codes).float().mean()))
                    prepared[idx] = (src, kept, orig_len)
            except Exception as e:                                   # noqa: BLE001
                # Falling back to the ground truth would silently mix
                # distributions, so the whole batch is dropped instead.
                print_error(f"Failed to generate tokens for {len(ar_jobs)} file(s): {e}")
                for idx in sorted(ar_jobs, reverse=True):
                    prepared.pop(idx)
                    n_fail += 1
                    n_ar -= 1
                    progress.update(1)

        # --- Mimi decode -> BlueCodec encode --------------------------------
        for src, codes, orig_len in prepared:
            try:
                latents = _to_distil(codes, orig_len, mimi, blue, device)
                save_npz(output_dir / src.relative_to(input_dir).with_suffix(".npz"),
                         latents=latents)
                n_ok += 1
            except Exception as e:                                   # noqa: BLE001
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
        total = max(1, n_clean + n_ar)
        print_info("Ground-truth codecs", f"{n_clean} ({n_clean / total:.1%})", Colors.OKCYAN)
        print_info("AR generated", f"{n_ar} ({n_ar / total:.1%})", Colors.OKCYAN)
        if len_ratios:
            print_info("Mean generated/real length",
                       f"{sum(len_ratios) / len(len_ratios):.3f}", Colors.OKCYAN)
        if tok_matches:
            # 1.0 would mean the sampling changed nothing; lower means more drift
            # for the flow-matching model to learn from.
            print_info("Token agreement with truth",
                       f"{sum(tok_matches) / len(tok_matches):.3f}", Colors.OKCYAN)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
