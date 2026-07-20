from __future__ import annotations

from echo import config

import torch


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pads text and audio sequences in a batch to equal lengths.

    Text is padded with TEXT_PAD_ID token, audio frames are padded with the
    CODEC_PAD_ID vector on every codebook layer.

    Returns (text, audio_codec):
    * text — (B, T_text)
    * audio_codec — (B, T_audio, NUM_CODEBOOKS)
    """
    
    texts, audios = zip(*batch)

    B = len(texts)
    if B != len(audios):
        raise ValueError(f"text/audio count mismatch: {B} vs {len(audios)}")

    T_text = max(t.size(0) for t in texts)
    T_audio = max(a.size(0) for a in audios)
    C = config.NUM_CODEBOOKS

    padded_texts = torch.full((B, T_text), config.TEXT_PAD_ID, dtype=torch.long)
    padded_audios = torch.full((B, T_audio, C), config.CODEC_PAD_ID, dtype=torch.long)

    for i, (t, a) in enumerate(batch):
        padded_texts[i, : t.size(0)] = t
        padded_audios[i, : a.size(0)] = a

    return padded_texts, padded_audios