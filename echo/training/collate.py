from __future__ import annotations

import torch

from echo import config


def _mask(lengths: torch.Tensor, total: int) -> torch.Tensor:
    """Boolean key-padding mask (True = valid) of shape ``(B, total)``."""
    return torch.arange(total).unsqueeze(0) < lengths.unsqueeze(1)


def collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Pads a batch of :class:`~echo.training.dataset.EchoDataset` samples into
    uniform tensors and builds boolean key-padding masks (True = valid,
    False = padding).

    Only the fields present in the samples are collated, so this works for any
    combination of the dataset's ``load_*`` switches. Text is padded with
    ``config.text_pad``, latents are zero-padded and codec tokens are padded
    with ``config.prosody_pad``.

    Args:
        batch: list of dicts holding a ``text`` entry plus any of:
            text            — ``(T_text,)``          long
            latent          — ``(T_audio, C)``       float
            distil          — ``(T_audio, C)``       float
            codec           — ``(T_codec, L)``       long

    Returns a dict with, for every field present in the batch, the padded
    tensor and its ``<field>_key_padding_mask``:
        text                          — ``(B, S)``       long
        latent                        — ``(B, T, C)``    float
        distil                        — ``(B, T, C)``    float
        codec                         — ``(B, T_c, L)``  long
        text_key_padding_mask         — ``(B, S)``       bool
        latent_key_padding_mask       — ``(B, T)``       bool
        distil_key_padding_mask       — ``(B, T)``       bool
        codec_key_padding_mask        — ``(B, T_c)``     bool

    Latents and codec tokens sit on different frame rates, so each gets its own
    padded length; ``latent`` and ``distil`` share one temporal grid because the
    model consumes them together.
    """
    fields = batch[0].keys()
    B = len(batch)
    out: dict[str, torch.Tensor] = {}

    # --- Text ---
    texts = [s["text"] for s in batch]
    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
    S = int(text_lengths.max())
    padded_texts = torch.full((B, S), config.text_pad, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, : t.size(0)] = t

    out["text"] = padded_texts
    out["text_key_padding_mask"] = _mask(text_lengths, S)

    # --- Audio & distil latents (shared temporal grid) ---
    # Squeeze leading singleton dim if present (e.g. (1, T, C) -> (T, C)).
    frames = {
        name: [s[name].squeeze(0) if s[name].dim() == 3 and s[name].size(0) == 1 else s[name]
               for s in batch]
        for name in ("latent", "distil") if name in fields
    }
    if frames:
        lengths = {
            name: torch.tensor([f.size(0) for f in seqs], dtype=torch.long)
            for name, seqs in frames.items()
        }
        T = int(max(int(l.max()) for l in lengths.values()))
        # Round up to the nearest multiple of 8 so the U-Net's 3 stride-2 downsamples
        # and exact-doubling upsamples perfectly reconstruct the temporal length.
        T = ((T + 7) // 8) * 8

        for name, seqs in frames.items():
            C = seqs[0].size(-1)
            padded = torch.zeros((B, T, C), dtype=torch.float32)
            for i, f in enumerate(seqs):
                padded[i, : f.size(0)] = f

            out[name] = padded
            out[f"{name}_key_padding_mask"] = _mask(lengths[name], T)

    # --- Codec tokens ---
    if "codec" in fields:
        codes = [s["codec"] for s in batch]
        codec_lengths = torch.tensor([c.size(0) for c in codes], dtype=torch.long)
        T_c = int(codec_lengths.max())
        L = codes[0].size(-1)
        padded_codes = torch.full((B, T_c, L), config.prosody_pad, dtype=torch.long)
        for i, c in enumerate(codes):
            padded_codes[i, : c.size(0)] = c

        out["codec"] = padded_codes
        out["codec_key_padding_mask"] = _mask(codec_lengths, T_c)

    return out
