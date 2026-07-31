from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from echo.tokenizer import Tokenizer


class EchoDataset(Dataset):
    """
    Dataset of (text tokens, audio latent, distil latent) triples.

    Audio latents are optionally **channel-normalized**: per-channel mean and
    std loaded from a ``latent_stats.npz`` file (produced by
    ``scripts/compute_latent_stats.py``) are applied so each of the
    ``latent_dim`` channels is approximately zero-mean unit-variance. This
    aligns the data distribution with the unit-Gaussian noise prior used in
    flow matching. The same stats file must be used to denormalize latents at
    inference time before codec decoding.

    Args:
        phonemes_csv: path to a CSV file with lines ``<latent_file>|<phoneme_string>``.
        latent_dir: directory containing ``.npz`` latent files (shape ``(C, T)``).
        distils_dir: directory containing ``.npz`` distil latent files.
        tokenizer: :class:`Tokenizer` instance for phoneme → token conversion.
        latent_stats: optional path to a ``latent_stats.npz`` file containing
            ``mean`` and ``std`` arrays of shape ``(latent_dim,)``. When
            provided, latents are z-scored per channel on load. Defaults to
            ``<latent_dir>/latent_stats.npz`` if that file exists, otherwise
            no normalization is applied.
    """

    def __init__(
        self,
        phonemes_csv: str | Path,
        latent_dir: str | Path,
        distils_dir: str | Path,
        tokenizer: Tokenizer,
        latent_stats: str | Path | None = None,
    ) -> None:
        self._latent_dir = Path(latent_dir)
        self._distils_dir = Path(distils_dir)
        self._tokenizer = tokenizer

        # Resolve the stats path: explicit argument, else default location.
        if latent_stats is not None:
            stats_path = Path(latent_stats)
        else:
            stats_path = self._latent_dir / "latent_stats.npz"

        self._latent_mean: torch.Tensor | None = None
        self._latent_std: torch.Tensor | None = None
        if stats_path.is_file():
            stats = np.load(stats_path)
            self._latent_mean = torch.from_numpy(stats["mean"].astype(np.float32))  # (C,)
            self._latent_std = torch.from_numpy(stats["std"].astype(np.float32))    # (C,)

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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        npz_name, phonemes = self._samples[idx]
        text_ids = torch.tensor(self._tokenizer.tokenize(phonemes), dtype=torch.long)

        arr = np.load(self._latent_dir / npz_name)["latents"]      # (C, T)
        latent = torch.from_numpy(arr.T.copy()).float()            # (T, C)

        d_arr = np.load(self._distils_dir / npz_name)["latents"]   # (C, T)
        distil = torch.from_numpy(d_arr.T.copy()).float()          # (T, C)

        if self._latent_mean is not None:
            latent = (latent - self._latent_mean) / self._latent_std
            distil = (distil - self._latent_mean) / self._latent_std

        return text_ids, latent, distil