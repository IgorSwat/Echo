from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from echo.tokenizer import Tokenizer


class EchoDataset(Dataset):
    """
    Text tokens paired with audio latents, distil latents and/or codec tokens.

    Latents are channel-normalized so the data matches the unit-Gaussian prior
    flow matching assumes. `norm_mode` picks where the statistics come from:

      "dataset"   corpus-wide mean/std, read from `latent_stats.npz`.
      "instance"  the utterance's own, taken from the *distil* — at inference the
                  distil is the only tensor that exists, so only its statistics
                  can invert the model's output.
    """

    NORM_MODES = ("dataset", "instance")

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
        norm_mode: str = "dataset",
    ) -> None:
        if norm_mode not in self.NORM_MODES:
            raise ValueError(f"norm_mode must be one of {self.NORM_MODES}, got {norm_mode!r}")
        if codec_layers < 1:
            raise ValueError(f"codec_layers must be >= 1, got {codec_layers}")

        self._tokenizer = tokenizer
        self._codec_layers = codec_layers
        self._norm_mode = norm_mode

        # Each field maps to the directory it loads from, or None when disabled.
        self._dirs: dict[str, Path | None] = {}
        for name, enabled, directory in (
            ("latent", load_latent, latent_dir),
            ("distil", load_distil, distils_dir),
            ("codec", load_codec, codec_dir),
        ):
            if enabled and directory is None:
                raise ValueError(
                    f"{name} loading is enabled but no directory was given; "
                    f"pass {name}_dir=... or load_{name}=False"
                )
            self._dirs[name] = Path(directory) if enabled else None

        self._stats = self._load_stats(latent_stats)
        self._samples = self._load_index(Path(phonemes_csv))

    # ---------
    # Indexing
    # ---------

    @staticmethod
    def _load_index(phonemes_csv: Path) -> list[tuple[str, str]]:
        """
        Parse the `<npz_name>|<phonemes>` manifest.
        """

        samples: list[tuple[str, str]] = []
        with open(phonemes_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                npz_name, phonemes = line.split("|", 1)
                samples.append((npz_name.strip(), phonemes.strip()))

        return samples

    def _load_stats(
        self, latent_stats: str | Path | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Corpus-wide (mean, std) for "dataset" mode, or None if unavailable.
        """

        if latent_stats is not None:
            path = Path(latent_stats)
        elif self._dirs["latent"] is not None:
            path = self._dirs["latent"] / "latent_stats.npz"
        else:
            return None

        if not path.is_file():
            return None

        stats = np.load(path)

        return (
            torch.from_numpy(stats["mean"].astype(np.float32)),      # (C,)
            torch.from_numpy(stats["std"].astype(np.float32)),       # (C,)
        )

    def __len__(self) -> int:
        return len(self._samples)

    # ---------
    # Loading
    # ---------

    def _load_frames(self, field: str, npz_name: str) -> torch.Tensor:
        """
        A `(C, T)` latent file as an unnormalized `(T, C)` tensor.
        """

        arr = np.load(self._dirs[field] / npz_name)["latents"]       # (C, T)

        return torch.from_numpy(arr.T.copy()).float()                # (T, C)

    def _load_codes(self, npz_name: str) -> torch.Tensor:
        """
        A `(num_layers, T)` codec file as a `(T, codec_layers)` tensor.
        """

        codes = np.load(self._dirs["codec"] / npz_name)["codes"]     # (num_layers, T)
        
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

        return torch.from_numpy(codes[: self._codec_layers].T.copy()).long()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        npz_name, phonemes = self._samples[idx]

        sample: dict[str, torch.Tensor] = {
            "text": torch.tensor(self._tokenizer.tokenize(phonemes), dtype=torch.long),
        }
        for field in ("latent", "distil"):
            if self._dirs[field] is not None:
                sample[field] = self._load_frames(field, npz_name)

        # Both streams share one set of statistics. In "instance" mode they come
        # from the distil, so the same numbers are available at inference.
        if self._norm_mode == "instance":
            reference = sample.get("distil", sample.get("latent"))
            stats = (reference.mean(0), reference.std(0).clamp_min(1e-5))
        else:
            stats = self._stats

        if stats is not None:
            mean, std = stats
            for field in ("latent", "distil"):
                if field in sample:
                    sample[field] = (sample[field] - mean) / std

        if self._dirs["codec"] is not None:
            sample["codec"] = self._load_codes(npz_name)

        return sample
