from __future__ import annotations

import random

from echo import config

import torch


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
    dataset=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Samples a shared ``(ref_text, ref_audio_codec)`` pair from ``dataset``
    (uniform over the full training set, independent of the batch) and pads
    the per-item text and audio targets.

    Text is padded with ``TEXT_PAD_ID``; audio frames are padded with the
    ``CODEC_PAD_ID`` vector on every codebook layer. The shared reference
    tensors are returned as-is (batch dim added via ``unsqueeze(0)``), so they
    are not padded.

    Returns:
        ref_text          — ``(1, T_ref_text)``
        ref_audio_codec   — ``(1, T_ref_audio, NUM_CODEBOOKS)``
        text              — ``(B, T_text)``
        audio_codec       — ``(B, T_audio, NUM_CODEBOOKS)``
        text_lengths      — ``(B,)`` true target-text lengths
        audio_lengths     — ``(B,)`` true target-audio lengths
    """
    texts, audios = zip(*batch)
    texts, audios = list(texts), list(audios)

    B = len(texts)
    C = config.NUM_CODEBOOKS

    # Shared reference: drawn from the whole dataset, independent of the batch.
    if dataset is None:
        # Fallback: sample from the current batch (keeps the function callable
        # without binding, e.g. for ad-hoc tests).
        j = random.randrange(B)
        ref_text = texts[j].unsqueeze(0)
        ref_audio = audios[j].unsqueeze(0)
    else:
        ref_text, ref_audio = dataset[random.randrange(len(dataset))]
        ref_text = ref_text.unsqueeze(0)
        ref_audio = ref_audio.unsqueeze(0)

    # Broadcast the shared reference to the full batch dimension so the model
    # sees a consistent batch size across ref and target tensors.
    ref_text = ref_text.expand(B, -1).contiguous()
    ref_audio = ref_audio.expand(B, -1, -1).contiguous()

    # Pad text targets.
    T_text = max(t.size(0) for t in texts)
    padded_texts = torch.full((B, T_text), config.TEXT_PAD_ID, dtype=torch.long)
    for i, t in enumerate(texts):
        padded_texts[i, : t.size(0)] = t

    # Pad audio targets.
    T_audio = max(a.size(0) for a in audios)
    padded_audios = torch.full((B, T_audio, C), config.CODEC_PAD_ID, dtype=torch.long)
    for i, a in enumerate(audios):
        padded_audios[i, : a.size(0)] = a

    text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
    audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
    return ref_text, ref_audio, padded_texts, padded_audios, text_lengths, audio_lengths
