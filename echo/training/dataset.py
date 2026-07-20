from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from echo.tokenizer import Tokenizer


class EchoDataset(Dataset):
    """
    Dataset of (text tokens, audio codec tokens) pairs.

    Args:
        phonemes_csv: path to a CSV file with lines ``<codec_file>|<phoneme_string>``.
        codec_dir: directory containing ``.npz`` codec files (shape ``(L, T)``).
        tokenizer: :class:`Tokenizer` instance for phoneme → token conversion.
    """

    def __init__(
        self, 
        phonemes_csv: str | Path,
        codec_dir: str | Path,
        tokenizer: Tokenizer
    ) -> None:
        self._codec_dir = Path(codec_dir)
        self._tokenizer = tokenizer

        self._samples: list[tuple[str, str]] = []
        with open(phonemes_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                
                npz_name, phonemes = line.split("|", 1)
                self._samples.append((npz_name.strip(), phonemes.strip()))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        npz_name, phonemes = self._samples[idx]
        text_ids = torch.tensor(self._tokenizer.tokenize(phonemes), dtype=torch.long)

        arr = np.load(self._codec_dir / npz_name)["codes"]          # (L, T)
        audio_ids = torch.from_numpy(arr.T.copy()).long()           # (T, L)

        return text_ids, audio_ids