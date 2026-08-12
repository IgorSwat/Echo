from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from echo import config
from echo.tokenizer import Tokenizer


class EchoDataset(Dataset):
    """
    Text tokens paired with audio latents, distil latents and/or codec tokens.
    """

    NORM_MODES = ("dataset", "instance")

    # Sidecar holding one frame count per codec file, so the length floor for
    # references costs nothing at sample time. Lives beside the codecs it
    # describes and is shared by every manifest that reads them.
    CODEC_LENGTH_CACHE = "codec_lengths.npz"

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
        load_reference: bool = False,
        min_ref_frames: int = 50,                    # 4.0 s at 12.5 Hz
        max_ref_frames: int | None = None,
        reference_seed: int | None = None,
        max_text_tokens: int | None = config.text_len_limit,
    ) -> None:
        if norm_mode not in self.NORM_MODES:
            raise ValueError(f"norm_mode must be one of {self.NORM_MODES}, got {norm_mode!r}")
        if codec_layers < 1:
            raise ValueError(f"codec_layers must be >= 1, got {codec_layers}")
        if min_ref_frames < 1:
            raise ValueError(f"min_ref_frames must be >= 1, got {min_ref_frames}")
        if max_ref_frames is not None and max_ref_frames < min_ref_frames:
            raise ValueError(
                f"max_ref_frames ({max_ref_frames}) must be >= "
                f"min_ref_frames ({min_ref_frames})"
            )

        self._tokenizer = tokenizer
        self._codec_layers = codec_layers
        self._norm_mode = norm_mode
        self._reference_seed = reference_seed

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

        # Both models size their rotary tables from `text_len_limit`, and the AR
        # model's paired mode has to fit two transcripts plus a separator inside
        # twice it, so a transcript over the limit is not merely long -- it
        # cannot be encoded at all. Dropping those utterances here keeps the
        # failure out of the training loop, where it arrives thousands of steps
        # in as a shape error. `dropped_long_text` is the count, for the caller
        # to report rather than lose silently.
        self.dropped_long_text = 0
        if max_text_tokens is not None:
            kept = [
                s for s in self._samples
                if len(tokenizer.tokenize(s[1])) <= max_text_tokens
            ]
            self.dropped_long_text = len(self._samples) - len(kept)
            self._samples = kept

        # speaker -> indices of its utterances that may serve as a reference.
        self._by_speaker: dict[str, np.ndarray] | None = None
        if load_reference:
            self._build_reference_index(min_ref_frames, max_ref_frames)

    # ---------
    # Indexing
    # ---------

    @staticmethod
    def _load_index(phonemes_csv: Path) -> list[tuple[str, str, str | None]]:
        """
        Parse the `<npz_name>|<phonemes>[|<speaker_id>]` manifest.
        """

        samples: list[tuple[str, str, str | None]] = []
        with open(phonemes_csv, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                npz_name, phonemes = parts[0], parts[1]
                if not npz_name.endswith(".npz"):
                    npz_name += ".npz"
                speaker = parts[2] if len(parts) > 2 and parts[2] else None
                samples.append((npz_name, phonemes, speaker))

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

    # ------------------
    # Reference sampling
    # ------------------

    def _codec_lengths(self) -> np.ndarray:
        """
        Frame count per manifest entry, cached beside the codec files.
        """

        codec_dir = self._dirs["codec"]
        cache_path = codec_dir / self.CODEC_LENGTH_CACHE

        known: dict[str, int] = {}
        if cache_path.is_file():
            cached = np.load(cache_path)
            known = dict(zip(cached["names"].tolist(), cached["lengths"].tolist()))

        names = [name for name, _, _ in self._samples]
        missing = [name for name in dict.fromkeys(names) if name not in known]

        if missing:
            for name in missing:
                known[name] = int(np.load(codec_dir / name)["codes"].shape[-1])

            entries = sorted(known.items())
            try:
                np.savez(
                    cache_path,
                    names=np.array([k for k, _ in entries]),
                    lengths=np.array([v for _, v in entries], dtype=np.int32),
                )
            except OSError:
                # A read-only corpus is still perfectly usable; the lengths are
                # already in hand, they just have to be recomputed next run.
                pass

        return np.array([known[name] for name in names], dtype=np.int64)

    def _build_reference_index(
        self, min_ref_frames: int, max_ref_frames: int | None
    ) -> None:
        """
        Group every eligible utterance under its speaker, once, up front.
        """

        if self._dirs["codec"] is None:
            raise ValueError(
                "load_reference=True needs the codec tokens; "
                "pass codec_dir=... or load_reference=False"
            )
        if any(speaker is None for _, _, speaker in self._samples):
            raise ValueError(
                "load_reference=True needs a speaker column in the manifest "
                "(`<npz_name>|<phonemes>|<speaker_id>`)"
            )

        lengths = self._codec_lengths()
        eligible = lengths >= min_ref_frames
        if max_ref_frames is not None:
            eligible &= lengths <= max_ref_frames

        by_speaker: dict[str, list[int]] = {}
        for idx, (_, _, speaker) in enumerate(self._samples):
            if eligible[idx]:
                by_speaker.setdefault(speaker, []).append(idx)

        stranded = sorted({s for _, _, s in self._samples} - by_speaker.keys())
        if stranded:
            window = (
                f">= {min_ref_frames}" if max_ref_frames is None
                else f"in [{min_ref_frames}, {max_ref_frames}]"
            )
            raise ValueError(
                f"{len(stranded)} speaker(s) have no utterance {window} frames "
                f"long and so can never be referenced: {stranded[:5]}"
                f"{'...' if len(stranded) > 5 else ''}. Lower min_ref_frames, "
                f"raise max_ref_frames, or drop them from the manifest."
            )

        # Numpy rather than lists: with forked dataloader workers, touching the
        # refcounts of a hundred thousand Python objects undoes copy-on-write.
        self._by_speaker = {
            speaker: np.array(indices, dtype=np.int64)
            for speaker, indices in by_speaker.items()
        }

    def _reference_index(self, idx: int, speaker: str) -> int:
        """
        Pick another utterance by `speaker`, in constant time.

        Drawing and then stepping past a self-hit is unbiased enough and, unlike
        rejection sampling, cannot stall on a speaker with few candidates.
        """

        pool = self._by_speaker[speaker]
        n = len(pool)

        if self._reference_seed is None:
            j = random.randrange(n)
        else:
            # Seeded by the sample, so a validation pass sees the same
            # references on every epoch and its loss is comparable across them.
            j = int(np.random.default_rng((self._reference_seed, idx)).integers(n))

        if pool[j] == idx and n > 1:
            j = (j + 1) % n

        return int(pool[j])

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

    def _tokenize(self, phonemes: str) -> torch.Tensor:
        return torch.tensor(self._tokenizer.tokenize(phonemes), dtype=torch.long)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        npz_name, phonemes, speaker = self._samples[idx]

        sample: dict[str, torch.Tensor] = {"text": self._tokenize(phonemes)}
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

        # The reference is a second utterance by this speaker, read in full: the
        # AR model needs its transcript alongside its frames, and cropping the
        # audio would leave the two out of step.
        if self._by_speaker is not None:
            ref_name, ref_phonemes, _ = self._samples[self._reference_index(idx, speaker)]
            sample["ref_text"] = self._tokenize(ref_phonemes)
            sample["ref_codec"] = self._load_codes(ref_name)

        return sample
