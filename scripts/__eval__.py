"""ASR transcription and word-error scoring, shared by the ``eval_*`` scripts."""

from __future__ import annotations

import re
import unicodedata

import librosa
import numpy as np
import torch

# Every Whisper variant wants 16 kHz mono.
ASR_SR = 16000

# Whisper's own English normalizer, when it is available.
#
# WER is only meaningful if both sides are written the same way, and the two
# sides here disagree on more than punctuation: Whisper writes "11" where the
# corpus transcript writes "eleven", and expands contractions differently.
# Whisper ships the normalizer it was evaluated with, which settles all of that
# consistently — scoring against anything else invents errors the model did not
# make.
try:
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

    _NORMALIZER = EnglishTextNormalizer({})
except ImportError:                                                  # pragma: no cover
    _NORMALIZER = None


def normalize_text(text: str) -> list[str]:
    """Normalize a transcript to a comparable word list."""
    if _NORMALIZER is not None:
        return _NORMALIZER(text).split()

    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().replace("-", " ").replace("—", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)

    return text.split()


def edit_ops(ref: list[str], hyp: list[str]) -> tuple[int, int, int]:
    """
    Levenshtein alignment of two word sequences -> (subs, deletions, insertions).
    """

    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = 1 + min(d[i - 1, j - 1], d[i - 1, j], d[i, j - 1])

    subs = dels = ins = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i, j] == d[i - 1, j - 1]:
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + 1:
            subs += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return subs, dels, ins


class ASR:
    """
    Whisper on MLX where available, otherwise the transformers implementation.
    """

    MLX_DEFAULT = "mlx-community/whisper-small-mlx"
    HF_DEFAULT = "openai/whisper-small"

    def __init__(
        self,
        backend: str,
        model: str | None,
        language: str,
        device: torch.device,
    ) -> None:
        if backend == "auto":
            backend = "mlx" if self._mlx_available() else "transformers"
        if backend == "mlx" and not self._mlx_available():
            raise SystemExit(
                "--asr-backend mlx needs the mlx-whisper package: pip install mlx-whisper"
            )

        self.backend = backend
        self.language = language
        self.name = model or (self.MLX_DEFAULT if backend == "mlx" else self.HF_DEFAULT)

        if backend == "mlx":
            import mlx_whisper                                       # noqa: PLC0415

            self._mlx = mlx_whisper
        else:
            from transformers import pipeline                        # noqa: PLC0415

            # Whisper's own generation defaults handle >30 s inputs by chunking.
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.name,
                device=device,
                chunk_length_s=30,
            )

    @staticmethod
    def _mlx_available() -> bool:
        from importlib.util import find_spec                         # noqa: PLC0415

        return find_spec("mlx_whisper") is not None

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """``audio`` is mono float32 at ``sample_rate``; resampled to 16 kHz here."""
        if sample_rate != ASR_SR:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=ASR_SR)
        audio = audio.astype(np.float32)

        if self.backend == "mlx":
            out = self._mlx.transcribe(
                audio, path_or_hf_repo=self.name, language=self.language, verbose=None,
            )
            return out["text"]

        return self._pipe(
            {"raw": audio, "sampling_rate": ASR_SR},
            generate_kwargs={"language": self.language, "task": "transcribe"},
        )["text"]
