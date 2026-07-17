#!/usr/bin/env python3
"""Prepare a training manifest from an audio|text pair file.

Input format (one pair per line, pipe-separated):
    clone_0000.wav|We will send you a renewal notice in the post in October.

The script:
  1. Phonemizes the transcription using the ``misaki`` (espeak) g2p with the
     language given by ``--lang``.
  2. Renames the audio filename to its codec counterpart (``.wav`` -> ``.npz``).

Output format (one pair per line, pipe-separated):
    clone_0000.npz|<phonemes>

Usage:
    python scripts/prepare_training.py --input manifest.txt --out manifest_phon.txt --lang en
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set ESPEAK_DATA_PATH before any espeak/phonemizer import so the C library
# picks up the correct data directory (the bundled espeakng_loader path may
# point to a CI build path that doesn't exist on the host).
try:
    import espeakng_loader
    os.environ.setdefault("ESPEAK_DATA_PATH", espeakng_loader.get_data_path())
except ImportError:
    pass

# Make the ``echo`` package and ``__style__`` importable when running this
# script directly, regardless of the current working directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tqdm import tqdm  # noqa: E402

from __style__ import (  # noqa: E402
    Colors,
    print_header,
    print_section,
    print_info,
    print_separator,
    print_success,
    print_error,
)


class Phonemizer:
    """espeak g2p wrapper."""

    def __init__(self, lang_code: str) -> None:
        # ``misaki`` expects ``EspeakWrapper.set_data_path`` which was removed
        # in newer ``phonemizer`` (3.3+). The ``ESPEAK_DATA_PATH`` env var is
        # already set at the top of this module; here we just stub the missing
        # classmethod so the import doesn't crash.
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        if not hasattr(EspeakWrapper, "set_data_path"):
            @classmethod
            def set_data_path(cls, path):
                pass
            EspeakWrapper.set_data_path = set_data_path

        from misaki import espeak
        self.g2p = espeak.EspeakG2P(language=lang_code)

    def phonemize(self, text: str) -> str:
        phonemes, _ = self.g2p(text)
        return phonemes


# Codec file extension that replaces the audio extension.
_CODEC_EXT = ".npz"


def _to_codec_name(audio_name: str) -> str:
    """``clone_0000.wav`` -> ``clone_0000.npz``."""
    base = audio_name.rsplit(".", 1)[0]
    return base + _CODEC_EXT


def main() -> None:
    parser = argparse.ArgumentParser(description="Phonemize a training manifest and rename audio -> codec files.")
    parser.add_argument("--input", type=str, required=True, help="Input manifest (audio|text per line).")
    parser.add_argument("--output", type=str, required=True, help="Output manifest path.")
    parser.add_argument("--lang", type=str, default="en", help="Phonemization language code (default: en).")
    args = parser.parse_args()

    print_header("Prepare Training Manifest")
    print_separator()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.is_file():
        print_error(f"Input file not found: {in_path}")
        sys.exit(1)

    # --- Load input ---------------------------------------------------------
    print_section("Input")
    with in_path.open("r", encoding="utf-8") as f:
        raw_lines = [ln.strip() for ln in f if ln.strip()]
    print_info("File", str(in_path), Colors.OKCYAN)
    print_info("Pairs", str(len(raw_lines)))
    print_info("Language", args.lang, Colors.OKCYAN)

    # --- Phonemize ----------------------------------------------------------
    print_section("Phonemizing")
    phonemizer = Phonemizer(args.lang)

    results: list[str] = []
    n_ok, n_fail, n_empty = 0, 0, 0

    for line in tqdm(raw_lines, desc="Phonemize", unit="pair"):
        if "|" not in line:
            n_fail += 1
            continue
        audio_name, text = line.split("|", 1)
        audio_name = audio_name.strip()
        text = text.strip()

        try:
            phonemes = phonemizer.phonemize(text)
        except Exception as e:  # noqa: BLE001
            print_error(f"Phonemize failed for '{audio_name}': {e}")
            n_fail += 1
            continue

        if not phonemes:
            n_empty += 1
            n_fail += 1
            continue

        codec_name = _to_codec_name(audio_name)
        results.append(f"{codec_name}|{phonemes}")
        n_ok += 1

    # --- Write output -------------------------------------------------------
    print_section("Output")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(r + "\n")
    print_info("File", str(out_path), Colors.OKCYAN)
    print_info("Written", str(n_ok), Colors.OKGREEN)

    # --- Summary ------------------------------------------------------------
    print_separator("═", 60)
    print_info("Pairs ok", str(n_ok), Colors.OKGREEN)
    if n_fail:
        print_info("Pairs failed", str(n_fail), Colors.FAIL)
    if n_empty:
        print_info("Empty phonemes", str(n_empty), Colors.WARNING)
    print_success("Done.")
    print_separator("═", 60)


if __name__ == "__main__":
    main()
