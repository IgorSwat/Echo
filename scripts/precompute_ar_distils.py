#!/usr/bin/env python3
"""Precompute distillation latents from AR-predicted codecs: AR → Mimi decode → BlueCodec encode.

For every row of a ``phonemes.csv`` file (``<name>.npz|<phonemes>``) the script
builds a two-layer Mimi codec grid and turns it into a BlueCodec latent:

  * with ``--model-ar``, the AR model generates a codec from the phonemes. If its
    length is within ``--k`` frames of the ground-truth codec (from ``--codec-dir``)
    it is truncated to — or padded from — the real codec's first two layers.
    If the length gap is larger, the real codec is used instead.
  * without ``--model-ar``, the real codec's first two layers are used directly.

Output files are ``.npz`` (zlib compressed) containing a single ``latents`` array
of shape ``(num_channels, T_latent)`` of float32 values.

Usage:
    python scripts/precompute_ar_distils.py --phonemes data/kanclerz/phonemes.csv \\
        --codec-dir data/kanclerz/codecs --output-dir data/kanclerz/ar_distils \\
        --model-ar checkpoints/echo_ar_final.pt
"""

from __future__ import annotations

import argparse
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
from tqdm import tqdm  # noqa: E402
from transformers import MimiModel  # noqa: E402
from bluecodec import BlueCodec  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)

from echo.ar_model import EchoAR  # noqa: E402
from echo.tokenizer import Tokenizer  # noqa: E402

_MIMI_SR = 24000
_BLUE_SR = 44100
_NUM_CHANNELS = 24
_NUM_LAYERS = EchoAR.NUM_TOKEN_LAYERS


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    """Parse ``<name>|<phonemes>`` rows, skipping blanks and malformed lines."""
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or "|" not in line:
                continue
            name, text = line.split("|", 1)
            rows.append((name.strip(), text.strip()))
    return rows


def _save_latents(latents: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, latents=latents.astype(np.float32))


def _merge_codes(generated: torch.Tensor, real: torch.Tensor, k: int) -> tuple[torch.Tensor, str]:
    """Reconcile an AR codec ``(layers, T_gen)`` with the real one ``(layers, T_real)``.

    Returns the codec to synthesize from plus a short tag describing the choice.
    """
    t_gen, t_real = generated.shape[1], real.shape[1]
    if abs(t_gen - t_real) > k:
        return real, "real"
    if t_gen >= t_real:
        return generated[:, :t_real], "truncated"
    return torch.cat([generated, real[:, t_gen:]], dim=1), "filled"


@torch.no_grad()
def _decode_codes(mimi: MimiModel, codes: torch.Tensor) -> torch.Tensor:
    """Mimi-decode ``(B, layers, T)`` token ids into a ``(B, T_audio)`` waveform."""
    dec_out = mimi.decode(codes)
    # dec_out returns (audio_values, ...) or MimiDecoderOutput
    audio = dec_out[0] if isinstance(dec_out, tuple) else dec_out.audio_values
    return audio.squeeze(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute distillation latents from AR codecs: AR → Mimi → BlueCodec."
    )
    parser.add_argument("--phonemes", type=str, required=True,
                        help="Path to a phonemes.csv manifest (<name>.npz|<phonemes>).")
    parser.add_argument("--codec-dir", type=str, required=True,
                        help="Directory containing ground-truth codec .npz files.")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to write .npz latent files to.")
    parser.add_argument("--model-ar", type=str, default=None,
                        help="AR checkpoint (.pt). If omitted, the real codec is used as-is.")
    parser.add_argument("--k", type=int, default=5,
                        help="Max |len(AR) - len(real)| frame gap to still use the AR output (default: 5).")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Hard cap on generated frames (default: 1000).")
    parser.add_argument("--vocab", type=str, default=str(_REPO_ROOT / "models" / "phoneme_vocab.json"),
                        help="Phoneme vocabulary used by the AR model.")
    parser.add_argument("--limit", "--samples", dest="limit", type=int, default=None,
                        help="Only process the first N entries (for testing).")
    args = parser.parse_args()

    print_header("AR Distillation Latents - Precompute (AR → Mimi → BlueCodec)")
    print_separator()

    warnings.filterwarnings("ignore", message=".*An output with one or more elements was resized.*")

    # --- Device -------------------------------------------------------------
    device = _select_device()
    print_section("Device")
    print_info("Selected", str(device), Colors.OKCYAN)

    # --- Input --------------------------------------------------------------
    manifest = Path(args.phonemes)
    codec_dir = Path(args.codec_dir)
    output_dir = Path(args.output_dir)
    if not manifest.is_file():
        print_error(f"Manifest not found: {manifest}")
        sys.exit(1)
    if not codec_dir.is_dir():
        print_error(f"Codec directory not found: {codec_dir}")
        sys.exit(1)

    rows = _read_manifest(manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        print_error(f"No usable entries in {manifest}")
        sys.exit(1)

    print_section("Input")
    print_info("Manifest", str(manifest), Colors.OKCYAN)
    print_info("Codec dir", str(codec_dir), Colors.OKCYAN)
    print_info("Output dir", str(output_dir), Colors.OKCYAN)
    print_info("Codec layers", str(_NUM_LAYERS))
    print_info("Length tolerance", f"{args.k} frames")
    print_info("Entries found", str(len(rows)))

    # --- Load AR model -------------------------------------------------------
    model: EchoAR | None = None
    tokenizer: Tokenizer | None = None
    if args.model_ar:
        print_section("Loading AR model")
        t_model = time.perf_counter()
        model = EchoAR().to(device)
        ckpt = torch.load(args.model_ar, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval()
        tokenizer = Tokenizer(Path(args.vocab))
        print_info("Checkpoint", args.model_ar)
        print_info("Model load time", f"{time.perf_counter() - t_model:.3f}s", Colors.OKCYAN)
    else:
        print_section("AR model")
        print_info("Mode", "disabled — using ground-truth codecs", Colors.WARNING)

    # --- Load Mimi -----------------------------------------------------------
    print_section("Loading Mimi model")
    t_model = time.perf_counter()
    mimi = MimiModel.from_pretrained("kyutai/mimi").to(device).eval()
    print_info("Sample rate", f"{_MIMI_SR} Hz")
    print_info("Model load time", f"{time.perf_counter() - t_model:.3f}s", Colors.OKCYAN)

    # --- Load BlueCodec ------------------------------------------------------
    print_section("Loading BlueCodec model")
    t_model = time.perf_counter()
    blue = BlueCodec.from_pretrained("notmax123/blue-codec", device=str(device))
    print_info("Sample rate", f"{_BLUE_SR} Hz")
    print_info("Latent channels", str(_NUM_CHANNELS))
    print_info("Model load time", f"{time.perf_counter() - t_model:.3f}s", Colors.OKCYAN)

    # --- Process -------------------------------------------------------------
    print_section("Processing")
    t_total = time.perf_counter()
    n_ok, n_fail = 0, 0
    tags = {"real": 0, "truncated": 0, "filled": 0}

    progress = tqdm(rows, desc="Processing", unit="file")
    for name, text in progress:
        try:
            codec_path = codec_dir / Path(name).with_suffix(".npz").name
            with np.load(codec_path) as data:
                real = torch.from_numpy(data["codes"][:_NUM_LAYERS].astype(np.int64)).to(device)

            codes, tag = real, "real"
            if model is not None:
                text_ids = torch.tensor(
                    [tokenizer.tokenize(text)], dtype=torch.long, device=device
                )
                gen = model.generate(text_ids, max_frames=args.max_frames)   # (1, T, layers)
                gen = gen[0].transpose(0, 1)                                 # (layers, T)
                if gen.shape[1] > 0:
                    codes, tag = _merge_codes(gen, real, args.k)
            tags[tag] += 1

            # Mimi decode -> 24 kHz waveform, then resample for BlueCodec.
            audio = _decode_codes(mimi, codes.unsqueeze(0))                  # (1, T_audio)
            audio_441 = torchaudio.functional.resample(audio, _MIMI_SR, _BLUE_SR)

            with torch.no_grad():
                latents = blue.encode(audio_441.to(device))
            latents_np = latents.detach().cpu().numpy().astype(np.float32).squeeze(0)

            _save_latents(latents_np, output_dir / Path(name).with_suffix(".npz").name)
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            print_error(f"Failed to process {name}: {e}")
            n_fail += 1

        progress.set_postfix(ok=n_ok, fail=n_fail)

    elapsed = time.perf_counter() - t_total

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Files processed", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Files failed", str(n_fail), Colors.FAIL)
    if model is not None:
        print_info("AR truncated", str(tags["truncated"]))
        print_info("AR filled", str(tags["filled"]))
        print_info("Fallback to real", str(tags["real"]), Colors.WARNING)
    print_info("Total time", f"{elapsed:.2f}s", Colors.OKCYAN)
    if n_ok:
        print_info("Throughput", f"{n_ok / elapsed:.2f} files/s", Colors.OKCYAN)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
