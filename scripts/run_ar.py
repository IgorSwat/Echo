#!/usr/bin/env python3
"""Generate audio with a trained EchoAR prosody model.

The model decodes Mimi codec tokens autoregressively from text alone: it starts
from a ``[BOS, BOS]`` frame and greedily appends one frame at a time until it
emits EOS. The resulting token grid is decoded to a waveform by Mimi itself.

Only the first ``EchoAR.NUM_TOKEN_LAYERS`` codebooks are modelled, so Mimi
reconstructs from those alone — expect coarse audio; this checks prosody and
timing, not fidelity.

Usage:
    python scripts/run_ar.py --model checkpoints/echo_ar_final.pt \\
        --text "həlˈOʊ wˈɜːld" --output out.wav
"""

from __future__ import annotations

import argparse
import sys
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
from transformers import MimiModel

from __style__ import Colors, print_header, print_info, print_section, print_separator

from echo import config
from echo.ar_model import EchoAR
from echo.tokenizer import Tokenizer

# Mimi operates at 24 kHz with a 12.5 Hz token grid.
_MIMI_SR = 24000
_MIMI_FPS = 12.5


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def _decode_codes(mimi: MimiModel, codes: torch.Tensor) -> torch.Tensor:
    """Mimi-decode ``(B, layers, T)`` token ids into a ``(B, T_audio)`` waveform."""
    dec_out = mimi.decode(codes)
    # dec_out returns (audio_values, ...) or MimiDecoderOutput
    if isinstance(dec_out, tuple):
        audio = dec_out[0]
    else:
        audio = dec_out.audio_values

    return audio.squeeze(1)                                          # (B, T_audio)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate audio with EchoAR.")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to a training checkpoint (.pt).")
    parser.add_argument("--text", type=str, required=True,
                        help="Phoneme string to synthesize.")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on generated frames (default: 1000, i.e. 80s).")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="Output audio path.")
    args = parser.parse_args()

    device = _select_device()

    # --- Model --------------------------------------------------------------
    model = EchoAR().to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    tokenizer = Tokenizer(_REPO_ROOT / "models" / "phoneme_vocab.json")
    text_ids = torch.tensor([tokenizer.tokenize(args.text)], dtype=torch.long, device=device)

    print_header("Echo - Autoregressive Generation")
    print_separator()
    print_section("Setup")
    print_info("Device", str(device), Colors.OKCYAN)
    print_info("Checkpoint", args.model)
    print_info("Text tokens", str(text_ids.shape[1]))
    print_info("Codec layers", str(EchoAR.NUM_TOKEN_LAYERS))
    print_info("Max frames", str(args.max_frames))

    # --- Generate -----------------------------------------------------------
    codes = model.generate(text_ids, max_frames=args.max_frames)     # (1, T, layers)
    frames = codes.shape[1]

    print_section("Generation")
    if frames == 0:
        print_info("Frames", "0 — the model emitted EOS immediately", Colors.FAIL)
        print_info("Hint", "undertrained checkpoint, or a text the model never saw",
                   Colors.WARNING)
        return
    if frames >= args.max_frames:
        print_info("Frames", f"{frames} — hit --max-frames without emitting EOS",
                   Colors.WARNING)
    else:
        print_info("Frames", str(frames), Colors.OKGREEN)
    print_info("Duration", f"{frames / _MIMI_FPS:.2f}s")

    # --- Mimi decode --------------------------------------------------------
    print_section("Decoding")
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()

    audio = _decode_codes(mimi, codes.transpose(1, 2))               # (1, T_audio)
    audio = audio.float().cpu()

    torchaudio.save(str(Path(args.output)), audio, _MIMI_SR)

    print_separator()
    print_info("Saved", args.output, Colors.OKGREEN)


if __name__ == "__main__":
    main()
