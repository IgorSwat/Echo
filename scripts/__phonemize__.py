"""Grapheme-to-phoneme front end shared by the ``run_*`` scripts.

The models are trained on phoneme strings, but typing IPA by hand is painful, so
the scripts take raw text and convert it here.

This must be the *same* G2P the corpus was built with, or inference feeds the
text encoder symbols it has never seen. That G2P is `misaki` (the engine behind
Kokoro), which spells English with a set of single-character symbols eSpeak does
not use at all -- ``ʧ``, ``ʤ``, ``ᵊ``, ``T`` for the flap, and ``A I W Y O`` for
the diphthongs -- and, conversely, never emits eSpeak's length marks (``ː``) or
``ɚ``. Running eSpeak here instead put roughly seven untrained tokens into every
utterance: ``ː`` alone averaged 5.6 per line and appears exactly zero times in
data/librispeech/phonemes_train.csv.

The configuration below reproduces the corpus **exactly** -- 400/400 utterances
re-phonemized from metadata.csv match phonemes_train.csv character for character.
The espeak fallback is part of that: without it, misaki emits ``❓`` for any word
outside its dictionary (Shrewsbury, Dorothy, ...), and the corpus has real
phonemes there.
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path

DEFAULT_LANGUAGE = "en-us"

# misaki's stand-in for a word it cannot pronounce. It is not in
# models/phoneme_vocab.json, so the tokenizer would drop it silently -- hence the
# warning in `phonemize`.
UNKNOWN = "❓"

_BRITISH = {"en-gb", "en-uk", "gb", "uk", "british"}
_AMERICAN = {"en-us", "en", "us", "american"}


def add_phonemize_args(parser: argparse.ArgumentParser) -> None:
    """Register the shared G2P flags on an ``ArgumentParser``."""
    parser.add_argument("--lang", "--language", dest="language", type=str,
                        default=DEFAULT_LANGUAGE,
                        help=f"Accent for phonemization: en-us or en-gb "
                             f"(default: {DEFAULT_LANGUAGE}). The corpus is en-us; "
                             f"en-gb is a different phoneme distribution than the "
                             f"models were trained on.")
    parser.add_argument("--no-stress", dest="stress", action="store_false",
                        help="Drop the ˈ/ˌ stress marks from the phonemes. The corpus "
                             "keeps them, so this is a deliberate mismatch.")
    parser.add_argument("--print-phonemes", action="store_true",
                        help="Print the phonemized text before synthesizing.")


def _install_espeak_shim() -> None:
    """Make misaki's espeak fallback importable against phonemizer 3.3.0.

    ``misaki.espeak`` points the wrapper at the espeak-ng data bundled in
    ``espeakng_loader``, via a ``set_data_path`` classmethod that only exists in
    ``phonemizer-fork``. Plain ``phonemizer`` instead reads the data path out of
    the loaded library, and the bundled library reports the CI path it was built
    on -- a directory that does not exist here. espeak-ng itself honours
    ``ESPEAK_DATA_PATH``, so pointing that at the bundled data fixes the lookup
    and leaves ``set_data_path`` with nothing to do.
    """

    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    os.environ.setdefault(
        "ESPEAK_DATA_PATH", str(Path(espeakng_loader.get_data_path()).parent)
    )
    if not hasattr(EspeakWrapper, "set_data_path"):
        EspeakWrapper.set_data_path = classmethod(lambda cls, path: None)


@lru_cache(maxsize=None)
def _g2p(british: bool):
    """A configured misaki G2P. Cached: building one loads spaCy."""

    _install_espeak_shim()

    from misaki import en, espeak

    return en.G2P(
        trf=False,                                   # en_core_web_sm, as the corpus used
        british=british,
        fallback=espeak.EspeakFallback(british=british),
    )


def _british(language: str) -> bool:
    key = language.strip().lower()
    if key in _BRITISH:
        return True
    if key in _AMERICAN:
        return False

    raise ValueError(
        f"unsupported language {language!r}: misaki's English G2P covers en-us and "
        f"en-gb, and the models were trained on en-us"
    )


def phonemize(
    text: str,
    language: str = DEFAULT_LANGUAGE,
    stress: bool = True,
) -> str:
    """Convert raw text to the phoneme string the models were trained on."""

    phonemes, tokens = _g2p(_british(language))(text)
    phonemes = (phonemes or "").strip()

    if UNKNOWN in phonemes:
        # The fallback normally catches these, so a survivor is worth surfacing:
        # the token is not in the vocabulary and vanishes at tokenization, taking
        # the word with it.
        unpronounced = sorted({
            t.text for t in tokens if t.phonemes and UNKNOWN in t.phonemes
        })
        print(f"warning: no pronunciation for {', '.join(unpronounced) or UNKNOWN}; "
              f"these words will be dropped", file=sys.stderr)

    if not stress:
        phonemes = phonemes.replace("ˈ", "").replace("ˌ", "")

    return phonemes


def phonemize_args(text: str, args: argparse.Namespace) -> str:
    """``phonemize`` driven by the flags added by :func:`add_phonemize_args`."""
    return phonemize(
        text,
        language=getattr(args, "language", DEFAULT_LANGUAGE),
        stress=getattr(args, "stress", True),
    )
