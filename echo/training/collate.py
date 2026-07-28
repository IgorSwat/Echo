from __future__ import annotations

import random

from echo import config

import torch

# TODO: This is dubious and should be rewritten

# def collate_fn(
#     batch: list[tuple[torch.Tensor, torch.Tensor]],
#     dataset=None,
#     reference_index: int | None = None,
# ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Samples a shared ``(ref_text, ref_audio_latent)`` pair from ``dataset``
#     (or uses ``reference_index`` when provided) and pads the per-item text
#     and audio latent targets.

#     Text is padded with ``TEXT_PAD_ID``; audio latents are padded with zeros.

#     Returns:
#         ref_text          — ``(1, T_ref_text)``
#         ref_audio_latent  — ``(1, T_ref_audio, latent_dim)``
#         text              — ``(B, T_text)``
#         audio_latent      — ``(B, T_audio, latent_dim)``
#         text_lengths      — ``(B,)`` true target-text lengths
#         audio_lengths     — ``(B,)`` true target-audio lengths
#     """
#     texts, audios = zip(*batch)
#     texts, audios = list(texts), list(audios)

#     B = len(texts)
#     C = config.latent_dim

#     # Shared reference: drawn from the whole dataset, independent of the batch.
#     if dataset is None:
#         j = random.randrange(B)
#         ref_text = texts[j].unsqueeze(0)
#         ref_audio = audios[j].unsqueeze(0)
#     else:
#         index = reference_index if reference_index is not None else random.randrange(len(dataset))
#         ref_text, ref_audio = dataset[index]
#         ref_text = ref_text.unsqueeze(0)
#         ref_audio = ref_audio.unsqueeze(0)

#     # Broadcast the shared reference to the full batch dimension.
#     ref_text = ref_text.expand(B, -1).contiguous()
#     ref_audio = ref_audio.expand(B, -1, -1).contiguous()

#     # Pad text targets.
#     T_text = max(t.size(0) for t in texts)
#     padded_texts = torch.full((B, T_text), config.text_pad_id, dtype=torch.long)
#     for i, t in enumerate(texts):
#         padded_texts[i, : t.size(0)] = t

#     # Pad audio latent targets (zero-padded).
#     T_audio = max(a.size(0) for a in audios)
#     padded_audios = torch.zeros((B, T_audio, C), dtype=torch.float32)
#     for i, a in enumerate(audios):
#         padded_audios[i, : a.size(0)] = a

#     text_lengths = torch.tensor([t.size(0) for t in texts], dtype=torch.long)
#     audio_lengths = torch.tensor([a.size(0) for a in audios], dtype=torch.long)
#     return ref_text, ref_audio, padded_texts, padded_audios, text_lengths, audio_lengths