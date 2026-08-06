"""Grapheme-to-phoneme front end shared by the ``run_*`` scripts.

The models are trained on phoneme strings, but typing IPA by hand is painful, so
the scripts take raw text and convert it here with the ``phonemizer`` package
(eSpeak NG backend).

eSpeak spells English diphthongs out in full (``oʊ``, ``eɪ``, ``aɪ``, ``aʊ``,
``ɔɪ``) while ``models/phoneme_vocab.json`` — and the training corpus — encode
each of them as a single symbol (``O``, ``A``, ``I``, ``W``, ``Y``). The
substitution below closes that gap; pass ``vocab_map=False`` to get raw eSpeak
IPA instead.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

# eSpeak's multi-character diphthongs -> the corpus' single-symbol spelling.
# Applied longest-first, which the ordering here already satisfies.
_DIPHTHONGS = {
    "oʊ": "O",
    "eɪ": "A",
    "aɪ": "I",
    "aʊ": "W",
    "ɔɪ": "Y",
}

DEFAULT_LANGUAGE = "en-us"


def add_phonemize_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared G2P flags on an ``ArgumentParser``."""
    parser.add_argument("--lang", "--language", dest="language", type=str, default=DEFAULT_LANGUAGE,
                        help=f"eSpeak language/voice for phonemization (default: {DEFAULT_LANGUAGE}).")
    parser.add_argument("--no-stress", dest="stress", action="store_false",
                        help="Drop the ˈ/ˌ stress marks from the phonemes.")
    parser.add_argument("--raw-ipa", dest="vocab_map", action="store_false",
                        help="Keep eSpeak's full diphthongs (oʊ, eɪ, …) instead of "
                             "folding them to the corpus symbols (O, A, …).")
    parser.add_argument("--print-phonemes", action="store_true",
                        help="Print the phonemized text before synthesizing.")


@lru_cache(maxsize=None)
def _backend(language: str, stress: bool):
    from phonemizer.backend import EspeakBackend

    return EspeakBackend(language, with_stress=stress, preserve_punctuation=True)


def phonemize(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    stress: bool = True,
    vocab_map: bool = True,
) -> str:
    """Convert raw text to the phoneme string the models were trained on."""
    out = _backend(language, stress).phonemize([text], strip=True)[0].strip()
    if vocab_map:
        for src, dst in _DIPHTHONGS.items():
            out = out.replace(src, dst)
    return out


def phonemize_args(text: str, args: argparse.Namespace) -> str:
    """``phonemize`` driven by the flags added by :func:`add_phonemize_args`."""
    return phonemize(
        text,
        language=getattr(args, "language", DEFAULT_LANGUAGE),
        stress=getattr(args, "stress", True),
        vocab_map=getattr(args, "vocab_map", True),
    )
