#!/usr/bin/env python3
"""Generate audio with a trained EchoAR prosody model.

The model decodes Mimi codec tokens autoregressively: it appends one frame at a
time until it emits EOS, and Mimi then decodes that token grid to a waveform.

A multispeaker checkpoint is prompted with a *reference* — a recording of the
voice to clone plus that recording's transcript. Both halves ride in front of
the target: the text encoder reads ``[ref_text] <sep> [text]`` and the frame
stream becomes ``[ref_frames] <bos> [generated]``. Pass ``--ref-audio`` and
``--ref-text`` together; a checkpoint trained with references will not sound
right without them, because it has never once decoded from a bare BOS.

Decoding is greedy unless ``--temperature`` is given. Greedy is deterministic and
mode-seeking on a stream this noisy; sampling near 1.0 matches the training
data's token statistics far more closely.

Only the first ``EchoAR.NUM_TOKEN_LAYERS`` codebooks are modelled, so expect
coarse audio — this checks prosody and timing, not fidelity.

Usage:
    python scripts/run/ar.py --model checkpoints/librispeech/echo_ar_multispeaker.pt \\
        --ref-audio prompt.flac --ref-text "the transcript of that recording" \\
        --text "hello world" --output out.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

# The shared helpers (__common__, __style__, ...) sit one level up, in scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import librosa
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
)
from __phonemize__ import add_phonemize_args, phonemize_args
from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR


# The window of reference lengths the model was actually trained on. Training
# draws whole same-speaker utterances of at least MIN_REF_FRAMES (the floor in
# scripts/train/ar.py), and the corpus' longest is MAX_REF_FRAMES. Outside it the
# prompt still decodes, but the prefix is something the model has never seen.
MIN_REF_FRAMES = 50
MAX_REF_FRAMES = 250


def _encode_reference(mimi: MimiModel, path: Path, device: torch.device) -> torch.Tensor:
    """Mimi-encode a reference recording onto the frame grid the model reads.

    Mirrors scripts/preprocess/codecs.py so a prompt encoded here is bit-for-bit
    what training would have loaded from the corpus: 24 kHz mono, rounded up to a
    whole frame — Mimi's encoder is causal, so only the frame straddling the end
    of the signal depends on what follows it — and only the modelled codebooks
    kept.
    """

    audio, _ = librosa.load(str(path), sr=MIMI_SR, mono=True)    # (samples,) at 24 kHz
    if audio.size == 0:
        raise ValueError(f"{path} decoded to no audio")

    hop = round(MIMI_SR / mimi.config.frame_rate)                # 1920 samples/frame
    padded = np.zeros(-(-audio.size // hop) * hop, dtype=np.float32)
    padded[: audio.size] = audio

    with torch.no_grad():
        enc = mimi.encode(torch.from_numpy(padded)[None, None, :].to(device))

    codes = enc.audio_codes[:, : EchoAR.NUM_TOKEN_LAYERS, :]     # (1, layers, T_ref)

    return codes.transpose(1, 2).long()                          # (1, T_ref, layers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with EchoAR.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a training checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True,
                        help="Raw text to synthesize (phonemized with misaki, the corpus G2P).")

    ref = parser.add_argument_group(
        "Voice reference",
        "A recording of the voice to clone plus its transcript. A multispeaker "
        "checkpoint was trained with one in front of every utterance, so leaving "
        "them out runs the model in a configuration it has never seen.",
    )
    ref.add_argument("--ref-audio", "--ref_audio", dest="ref_audio", type=str, default=None,
                     metavar="PATH",
                     help="Recording of the target voice (anything librosa can read). "
                          f"Training used {MIN_REF_FRAMES / MIMI_FPS:.0f}-"
                          f"{MAX_REF_FRAMES / MIMI_FPS:.0f}s of speech.")
    ref.add_argument("--ref-text", "--ref_text", dest="ref_text", type=str, default=None,
                     metavar="TEXT",
                     help="Raw transcript of --ref-audio, phonemized with the same front "
                          "end as --text.")

    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on generated frames (default: 1000, i.e. 80s).")
    parser.add_argument("--temperature", type=float, default=0.0, metavar="T",
                        help="Sampling temperature (default: 0.0 = greedy). Around 1.0 the "
                             "emitted token statistics track the training data much more "
                             "closely than greedy does; greedy is deterministic and blander.")
    parser.add_argument("--top-k", type=int, default=0, metavar="K",
                        help="Restrict each draw to the K most likely ids (default: 0 = off). "
                             "Only applies when --temperature > 0.")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio path.")
    add_phonemize_args(parser)
    args = parser.parse_args()

    if args.temperature < 0:
        parser.error("--temperature must be >= 0 (0 selects greedy decoding)")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0 (0 disables top-k)")
    # The two halves of the reference are one conditioning signal, not two: the
    # encoder splices the transcript in front of the target text while the frames
    # go in front of the target frames. Either alone is not a configuration the
    # model has.
    if bool(args.ref_audio) != bool(args.ref_text):
        missing = "--ref-text" if args.ref_audio else "--ref-audio"
        parser.error(f"{missing} is required alongside the one you gave: the model reads "
                     f"the reference as [ref_text] <sep> [text] over [ref_frames] <bos> "
                     f"[frames], so a recording without its transcript cannot prompt it")
    if args.ref_audio and not Path(args.ref_audio).is_file():
        parser.error(f"--ref-audio not found: {args.ref_audio}")

    device = select_device()

    # --- Text -----------------------------------------------------------------
    # Everything that can be rejected is rejected before the checkpoint is read,
    # so a typo costs a second rather than a model load.
    tokenizer = load_tokenizer()
    phonemes = phonemize_args(args.text, args)
    text_ids = torch.tensor([tokenizer.tokenize(phonemes)], dtype=torch.long, device=device)
    if text_ids.shape[1] == 0:
        parser.error("--text phonemized to nothing; check --lang")

    ref_phonemes = ref_text_ids = None
    if args.ref_text is not None:
        ref_phonemes = phonemize_args(args.ref_text, args)
        ref_text_ids = torch.tensor(
            [tokenizer.tokenize(ref_phonemes)], dtype=torch.long, device=device
        )
        if ref_text_ids.shape[1] == 0:
            parser.error("--ref-text phonemized to nothing; check --lang")
        # Caught here rather than as a bare length error from deep inside the text
        # encoder's rotary table.
        paired = ref_text_ids.shape[1] + 1 + text_ids.shape[1]
        limit = 2 * config.text_len_limit + 1
        if paired > limit:
            parser.error(
                f"the reference transcript and the text come to {paired} tokens together "
                f"(plus the separator), over the encoder's {limit}; shorten one of them "
                f"or raise limits.text_len in models/config.json"
            )

    # --- Model ----------------------------------------------------------------
    model = EchoAR().to(device)
    load_checkpoint(model, args.model, device, "EchoAR")
    model.eval()

    print_header("Echo - Autoregressive Generation")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Checkpoint", args.model)
    print_info("Language", args.language)
    if args.print_phonemes:
        print_info("Phonemes", phonemes, Colors.OKCYAN)
    print_info("Text tokens", str(text_ids.shape[1]))
    print_info("Codec layers", str(EchoAR.NUM_TOKEN_LAYERS))
    print_info("Max frames", str(args.max_frames))

    # --- Reference ------------------------------------------------------------
    # Mimi is needed either way -- to encode the prompt now and to decode the
    # result later -- so it is loaded once, here.
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    ref_codec = None
    if args.ref_audio is not None:
        ref_codec = _encode_reference(mimi, Path(args.ref_audio), device)
        ref_frames = ref_codec.shape[1]
        window = Colors.OKCYAN if MIN_REF_FRAMES <= ref_frames <= MAX_REF_FRAMES else Colors.WARNING
        print_info("Reference", f"{Path(args.ref_audio).name} - {ref_frames} frames "
                                f"({ref_frames / MIMI_FPS:.1f}s), "
                                f"{ref_text_ids.shape[1]} transcript tokens", window)
        if args.print_phonemes:
            print_info("Reference phonemes", ref_phonemes, Colors.OKCYAN)
        if ref_frames < MIN_REF_FRAMES:
            print_info("Reference length",
                       f"under the {MIN_REF_FRAMES / MIMI_FPS:.0f}s floor training used; "
                       f"the voice will carry over weakly", Colors.WARNING)
        elif ref_frames > MAX_REF_FRAMES:
            print_info("Reference length",
                       f"longer than any prompt in training "
                       f"({MAX_REF_FRAMES / MIMI_FPS:.0f}s); trim it if the output drifts",
                       Colors.WARNING)
    else:
        print_info("Reference", "none - decoding from a bare BOS. A multispeaker "
                                "checkpoint never saw this; pass --ref-audio with "
                                "--ref-text.", Colors.WARNING)
    if args.temperature > 0:
        print_info("Decoding", f"sampling (temperature {args.temperature:g}"
                               + (f", top-k {args.top_k}" if args.top_k > 0 else "") + ")",
                   Colors.OKCYAN)
    else:
        print_info("Decoding", "greedy (argmax)")

    # --- Generate -----------------------------------------------------------
    # The returned frames are the generated ones only; generate() drops the
    # reference prefix it decoded behind.
    codes = model.generate(text_ids, max_frames=args.max_frames,
                           temperature=args.temperature,
                           top_k=args.top_k,
                           ref_codec=ref_codec,
                           ref_text=ref_text_ids)                    # (1, T, layers)
    frames = codes.shape[1]

    print_section("Generation")
    if frames == 0:
        print_info("Frames", "0 — the model emitted EOS immediately", Colors.FAIL)
        print_info("Hint", "undertrained checkpoint, a text the model never saw, or a "
                           "missing reference on a checkpoint trained with one",
                   Colors.WARNING)
        return
    if frames >= args.max_frames:
        print_info("Frames", f"{frames} — hit --max-frames without emitting EOS",
                   Colors.WARNING)
    else:
        print_info("Frames", str(frames), Colors.OKGREEN)
    print_info("Duration", f"{frames / MIMI_FPS:.2f}s")

    # --- Mimi decode --------------------------------------------------------
    print_section("Decoding")

    with torch.no_grad():
        audio = decode_mimi(mimi, codes.transpose(1, 2))             # (1, T_audio)

    torchaudio.save(str(Path(args.output)), audio.float().cpu(), MIMI_SR)

    print_separator()
    print_info("Saved", args.output, Colors.OKGREEN)


if __name__ == "__main__":
    main()
