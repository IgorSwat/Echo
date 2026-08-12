from __future__ import annotations

import torch

from echo import config


def _mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Boolean key-padding mask (True = valid), of shape `(B, max(lengths))`.
    """

    return torch.arange(int(lengths.max())).unsqueeze(0) < lengths.unsqueeze(1)


def _padded(seqs: list[torch.Tensor], lengths: torch.Tensor, fill: float) -> torch.Tensor:
    """
    Stack variable-length `(T, ...)` tensors into `(B, max(lengths), ...)`,
    cutting each to its entry in `lengths` and filling the rest.
    """

    out = torch.full(
        (len(seqs), int(lengths.max()), *seqs[0].shape[1:]), fill, dtype=seqs[0].dtype
    )
    for i, seq in enumerate(seqs):
        n = int(lengths[i])
        out[i, :n] = seq[:n]

    return out


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Pad a batch of :class:`~echo.training.dataset.EchoDataset` samples and build
    a `_key_padding_mask` for each.

    Only the fields the samples actually carry are collated, so this works for
    any combination of the dataset's `load_*` switches.

        text        (B, S)      long   | latent  (B, T, C)   float
        codec       (B, T_c, L) long   | distil  (B, T, C)   float
        ref_text    (B, S_r)    long   | ref_codec (B, T_r, L) long
    """
    fields = batch[0].keys()
    out: dict[str, torch.Tensor] = {}

    # --- Token streams: each one padded on its own axis ---
    # The reference halves are separate entries rather than being spliced onto
    # the target here; the model merges them itself, since only it knows where
    # the separators go.
    for name, fill in (
        ("text", config.text_pad),
        ("ref_text", config.text_pad),
        ("codec", config.prosody_pad),
        ("ref_codec", config.prosody_pad),
    ):
        if name not in fields:
            continue
        seqs = [s[name] for s in batch]
        lengths = torch.tensor([t.size(0) for t in seqs], dtype=torch.long)
        out[name] = _padded(seqs, lengths, fill)
        out[f"{name}_key_padding_mask"] = _mask(lengths)

    # --- Latents: `latent` and `distil` share one temporal grid ---
    frames = {
        name: [
            s[name].squeeze(0) if s[name].dim() == 3 and s[name].size(0) == 1 else s[name]
            for s in batch                                  # tolerate a leading (1, T, C)
        ]
        for name in ("latent", "distil") if name in fields
    }
    
    if frames:
        # The model consumes the two together, so both are cut to the shorter and
        # share one mask.
        lengths = torch.tensor(
            [min(seqs[i].size(0) for seqs in frames.values()) for i in range(len(batch))],
            dtype=torch.long,
        )
        mask = _mask(lengths)
        for name, seqs in frames.items():
            out[name] = _padded(seqs, lengths, 0.0)
            out[f"{name}_key_padding_mask"] = mask

    return out
