from __future__ import annotations

import torch

from echo import config


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """
    Pads a batch of (text, audio_latent, distil_latent) triples into uniform
    tensors and builds boolean key-padding masks (True = valid, False = padding).

    Text is padded with ``config.text_pad``; audio and distil latents are zero-padded.

    Args:
        batch: list of ``(text, audio_latent, distil_latent)`` tuples where
            text            — ``(T_text,)``           long
            audio_latent    — ``(T_audio, C)``        float
            distil_latent   — ``(T_audio, C)``        float

    Returns dict with:
        text                          — ``(B, S)``      long
        latent                        — ``(B, T, C)``   float
        distil                        — ``(B, T, C)``   float
        text_key_padding_mask         — ``(B, S)``      bool  (True = valid)
        latent_key_padding_mask       — ``(B, T)``      bool  (True = valid)
        distil_key_padding_mask       — ``(B, T)``      bool  (True = valid)
    """
    texts, audios, distils = zip(*batch)
    B = len(texts)
    C = config.latent_dim

    # --- Text ---
    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
    S = int(text_lengths.max())
    padded_texts = torch.full((B, S), config.text_pad, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, : t.size(0)] = t

    text_mask = torch.arange(S).unsqueeze(0) < text_lengths.unsqueeze(1)   # (B, S)

    # --- Audio latent ---
    # Squeeze leading singleton dim if present (e.g. (1, T, C) -> (T, C)).
    audios = [a.squeeze(0) if a.dim() == 3 and a.size(0) == 1 else a for a in audios]
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    T = int(audio_lengths.max())
    # Round up to the nearest multiple of 8 so the U-Net's 3 stride-2 downsamples
    # and exact-doubling upsamples perfectly reconstruct the temporal length.
    T = ((T + 7) // 8) * 8
    padded_audios = torch.zeros((B, T, C), dtype=torch.float32)
    for i, a in enumerate(audios):
        padded_audios[i, : a.size(0)] = a

    audio_mask = torch.arange(T).unsqueeze(0) < audio_lengths.unsqueeze(1)  # (B, T)

    # --- Distil latent ---
    distils = [d.squeeze(0) if d.dim() == 3 and d.size(0) == 1 else d for d in distils]
    padded_distils = torch.zeros((B, T, C), dtype=torch.float32)
    for i, d in enumerate(distils):
        padded_distils[i, : d.size(0)] = d

    distil_mask = audio_mask  # same lengths as audio

    return {
        "text": padded_texts,
        "latent": padded_audios,
        "distil": padded_distils,
        "text_key_padding_mask": text_mask,
        "latent_key_padding_mask": audio_mask,
        "distil_key_padding_mask": distil_mask,
    }
