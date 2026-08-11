from typing import Optional

import torch


# -------
# Masking
# -------

def pad_attn_mask(
    key_padding_mask: torch.Tensor,                                 # (B, S)
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Broadcastable attention mask for a key-padding masking.
    """

    if key_padding_mask.shape != (batch_size, seq_len):
        raise ValueError(
            f"expected key_padding_mask shape {(batch_size, seq_len)}, "
            f"got {tuple(key_padding_mask.shape)}"
        )

    return key_padding_mask.view(batch_size, 1, 1, seq_len)         # (B, 1, 1, S)


def causal_attn_mask(
    q_len: int,
    k_len: int,
    start_pos: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Causal attention mask - stopping model from seeing the future.
    """

    q_pos = torch.arange(start_pos, start_pos + q_len, device=device).view(q_len, 1)
    k_pos = torch.arange(k_len, device=device).view(1, k_len)

    return (k_pos <= q_pos).view(1, 1, q_len, k_len)                # (1, 1, T, S)


# ------------------
# Rotary frequencies
# ------------------

def geometric_freqs(
    head_dim: int,
    theta: float,
    max_pos: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Classic RoPE table: a fixed geometric progression of frequencies, evaluated at integer positions ``0 ... max_pos - 1``.

    Nothing here is learned, so the table depends only on the head dim and
    can be cached once we know the sequence length limit.
    """

    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    ang = torch.outer(torch.arange(max_pos, device=device, dtype=torch.float32), freqs)

    return ang.cos(), ang.sin()                                     # each (max_pos, head_dim/2)


def absolute_freqs(
    theta: torch.Tensor,                                            # (rotary_dim,) learnable
    seq_len: int,
    device: torch.device,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Raw integer positions times a learnable per-stream theta.

    No length enters, so a position's encoding never changes as the sequence
    grows — the property autoregressive decoding needs. 
    """

    pos = torch.arange(
        start_pos, start_pos + seq_len, device=device, dtype=torch.float32
    ).view(1, seq_len, 1)
    ang = pos * theta.view(1, 1, -1)                                # (1, T, rotary_dim)

    return ang.cos(), ang.sin()


def normalized_freqs(
    theta: torch.Tensor,                                            # (rotary_dim,) learnable
    batch_size: int,
    seq_len: int,
    padding_mask: Optional[torch.Tensor],                           # (B, T) or None
    device: torch.device,
    start_pos: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Similar to absolute_freqs, but scaled into [0, 1] over each sequence's own valid region.

    Both streams then land on the same scale, so the alignment diagonal adapts to
    each utterance's rate rather than to its absolute length.
    """

    end = start_pos + seq_len
    pos = torch.arange(start_pos, end, device=device, dtype=torch.float32).view(1, seq_len, 1)
    if padding_mask is not None:
        lengths = padding_mask.sum(dim=1).clamp_min(1).view(-1, 1, 1).float()
    else:
        # Mid-decode the total length is unknown, so the sequence seen so far
        # stands in for it — the same value the uncached path would use here.
        lengths = torch.full((batch_size, 1, 1), float(end), device=device)
    ang = (pos / lengths) * theta.view(1, 1, -1)                    # (B, T, rotary_dim)

    return ang.cos(), ang.sin()


# ------------------
# Rotary application
# ------------------

def rotate_pairs(
    x: torch.Tensor,                                                # (B, nh, T, hd)
    cos: torch.Tensor,                                              # (max_pos, hd/2)
    sin: torch.Tensor,                                              # (max_pos, hd/2)
    start_pos: int = 0,
) -> torch.Tensor:
    """
    Rotate **adjacent** feature pairs (0,1), (2,3), ... by the angle for their
    absolute position, slicing the frequency table at `start_pos`.
    """

    T = x.size(2)

    x_rot = x.float().reshape(*x.shape[:-1], -1, 2)
    c = cos[start_pos:start_pos + T].view(1, 1, T, -1)
    s = sin[start_pos:start_pos + T].view(1, 1, T, -1)
    out = torch.empty_like(x_rot)
    out[..., 0] = x_rot[..., 0] * c - x_rot[..., 1] * s
    out[..., 1] = x_rot[..., 0] * s + x_rot[..., 1] * c

    return out.flatten(-2).to(x.dtype)                              # (B, nh, T, hd)


def rotate_halves(
    x: torch.Tensor,                                                # (B, nh, T, hd)
    cos: torch.Tensor,                                              # (B, T, rotary_dim)
    sin: torch.Tensor,                                              # (B, T, rotary_dim)
) -> torch.Tensor:
    """
    Rotate paired *halves* of the head dim, (i, i + rotary_dim).

    Mathematically the same rotation as :func:`rotate_pairs` on a different
    pairing of the features. Angles arrive per batch row here, since the
    normalized scheme scales them by each sequence's own length.
    """

    rd = sin.shape[-1]
    c = cos.unsqueeze(1)                                            # (B, 1, T, rd)
    s = sin.unsqueeze(1)                                            # (B, 1, T, rd)
    x1, x2 = x[..., :rd], x[..., rd:2 * rd]                         # paired halves
    rot = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    tail = x[..., 2 * rd:]                                          # untouched dims (odd hd)
    if tail.shape[-1] > 0:
        rot = torch.cat([rot, tail], dim=-1)

    return rot                                                      # (B, nh, T, hd)
