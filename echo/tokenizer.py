from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class Tokenizer:
    """Phoneme <-> token-id mapper backed by a ``phoneme_vocab.json`` file."""

    def __init__(self, vocab_path: str | Path) -> None:
        path = Path(vocab_path)
        if not path.is_file():
            raise FileNotFoundError(f"vocab file not found: {path}")

        try:
            vocab: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"failed to parse vocab file: {path}") from exc

        # Two mappings: phoneme -> token and back.
        self._p2t: dict[str, int] = {}
        self._t2p: dict[int, str] = {}

        for phoneme, token_id in vocab.items():
            if not isinstance(phoneme, str):
                raise ValueError(f"non-string key in vocab: {phoneme!r}")
            if not isinstance(token_id, int):
                raise ValueError(f"non-integer value for phoneme {phoneme!r}: {token_id}")
            if len(phoneme) != 1 and not (phoneme.startswith("<") and phoneme.endswith(">")):
                raise ValueError(
                    f"phoneme must be a single character or <special> token: {phoneme!r}"
                )

            self._p2t[phoneme] = token_id
            self._t2p[token_id] = phoneme

    def tokenize(self, phonemes: str) -> list[int]:
        """Token ids for a phoneme string; unrecognized phonemes are dropped."""

        return [self._p2t[ch] for ch in phonemes if ch in self._p2t]

    def detokenize(self, tokens: list[int]) -> str:
        """The phoneme string for a list of token ids."""

        return "".join(self._t2p[t] for t in tokens if t in self._t2p)

    @property
    def vocab_size(self) -> int:
        return len(self._t2p)
