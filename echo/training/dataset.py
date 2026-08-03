from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from echo.tokenizer import Tokenizer


class EchoDataset(Dataset):
    """
    Dataset of text tokens paired with any combination of audio latents, distil
    latents and discrete codec tokens.

    Each field is loaded only when its ``load_*`` flag is set, so a training run
    pays for what it uses: flow matching wants ``latent`` + ``distil``, the
    autoregressive model wants ``codec``. Samples are returned as dicts holding
    exactly the enabled fields.

    Audio latents are optionally **channel-normalized**: per-channel mean and
    std loaded from a ``latent_stats.npz`` file (produced by
    ``scripts/compute_latent_stats.py``) are applied so each of the
    ``latent_dim`` channels is approximately zero-mean unit-variance. This
    aligns the data distribution with the unit-Gaussian noise prior used in
    flow matching. The same stats file must be used to denormalize latents at
    inference time before codec decoding. Codec tokens are discrete and are
    never normalized.

    Note that latents and codec tokens live on different temporal grids (the
    codec stream is considerably coarser), so their lengths are tracked and
    padded independently.

    Args:
        phonemes_csv: path to a CSV file with lines ``<npz_file>|<phoneme_string>``.
        latent_dir: directory containing ``.npz`` latent files (key ``latents``,
            shape ``(C, T)``).
        distils_dir: directory containing ``.npz`` distil latent files.
        tokenizer: :class:`Tokenizer` instance for phoneme → token conversion.
        latent_stats: optional path to a ``latent_stats.npz`` file containing
            ``mean`` and ``std`` arrays of shape ``(latent_dim,)``. When
            provided, latents are z-scored per channel on load. Defaults to
            ``<latent_dir>/latent_stats.npz`` if that file exists, otherwise
            no normalization is applied.
        codec_dir: directory containing ``.npz`` codec files (key ``codes``,
            shape ``(num_layers, T)`` of integer token ids).
        codec_layers: how many leading codebook layers to keep (default 2).
        load_latent / load_distil / load_codec: per-field switches. A field that
            is enabled requires its directory to be given.
    """

    # npz payload keys.
    LATENT_KEY = "latents"
    CODEC_KEY = "codes"

    def __init__(
        self,
        phonemes_csv: str | Path,
        latent_dir: str | Path | None,
        distils_dir: str | Path | None,
        tokenizer: Tokenizer,
        latent_stats: str | Path | None = None,
        codec_dir: str | Path | None = None,
        codec_layers: int = 2,
        load_latent: bool = True,
        load_distil: bool = True,
        load_codec: bool = True,
    ) -> None:
        self._tokenizer = tokenizer
        self._codec_layers = codec_layers

        self._load_latent = load_latent
        self._load_distil = load_distil
        self._load_codec = load_codec

        self._latent_dir = Path(latent_dir) if latent_dir is not None else None
        self._distils_dir = Path(distils_dir) if distils_dir is not None else None
        self._codec_dir = Path(codec_dir) if codec_dir is not None else None

        # An enabled field without a directory is a configuration error, not
        # something to silently skip.
        for name, enabled, directory in (
            ("latent", load_latent, self._latent_dir),
            ("distil", load_distil, self._distils_dir),
            ("codec", load_codec, self._codec_dir),
        ):
            if enabled and directory is None:
                raise ValueError(
                    f"{name} loading is enabled but no directory was given; "
                    f"pass {name}_dir=... or load_{name}=False"
                )
        if codec_layers < 1:
            raise ValueError(f"codec_layers must be >= 1, got {codec_layers}")

        # Resolve the stats path: explicit argument, else default location.
        if latent_stats is not None:
            stats_path = Path(latent_stats)
        elif self._latent_dir is not None:
            stats_path = self._latent_dir / "latent_stats.npz"
        else:
            stats_path = None

        self._latent_mean: torch.Tensor | None = None
        self._latent_std: torch.Tensor | None = None
        if stats_path is not None and stats_path.is_file():
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

    def _load_frames(self, directory: Path, npz_name: str) -> torch.Tensor:
        """Load a ``(C, T)`` latent file as a normalized ``(T, C)`` tensor."""
        arr = np.load(directory / npz_name)[self.LATENT_KEY]        # (C, T)
        frames = torch.from_numpy(arr.T.copy()).float()             # (T, C)

        if self._latent_mean is not None:
            frames = (frames - self._latent_mean) / self._latent_std

        return frames                                               # (T, C)

    def _load_codes(self, npz_name: str) -> torch.Tensor:
        """Load a ``(num_layers, T)`` codec file as a ``(T, codec_layers)`` tensor."""
        codes = np.load(self._codec_dir / npz_name)[self.CODEC_KEY]  # (num_layers, T)
        # Tolerate a leading singleton (batch) dim, as the latent path does.
        if codes.ndim == 3 and codes.shape[0] == 1:
            codes = codes[0]
        if codes.ndim != 2:
            raise ValueError(
                f"{npz_name}: expected codec of shape (num_layers, T), got {codes.shape}"
            )
        if codes.shape[0] < self._codec_layers:
            raise ValueError(
                f"{npz_name}: codec has {codes.shape[0]} layers, "
                f"but codec_layers={self._codec_layers} were requested"
            )

        codes = codes[: self._codec_layers]                          # (codec_layers, T)

        return torch.from_numpy(codes.T.copy()).long()               # (T, codec_layers)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        npz_name, phonemes = self._samples[idx]

        sample: dict[str, torch.Tensor] = {
            "text": torch.tensor(self._tokenizer.tokenize(phonemes), dtype=torch.long),
        }

        if self._load_latent:
            sample["latent"] = self._load_frames(self._latent_dir, npz_name)
        if self._load_distil:
            sample["distil"] = self._load_frames(self._distils_dir, npz_name)
        if self._load_codec:
            sample["codec"] = self._load_codes(npz_name)

        return sample
