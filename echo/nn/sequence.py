from typing import Optional, Union

import torch


def valid_mask(
    mask: Optional[torch.Tensor],                                   # (B, S) or None
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.Tensor:
    """
    A key-padding mask, materialized as all-valid when the caller passed None.
    """

    if mask is not None:
        return mask

    return torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)


def merge_padded(
    prefix: torch.Tensor,                                           # (B, P, ...) long/float
    prefix_mask: Optional[torch.Tensor],                            # (B, P) bool or None
    suffix: torch.Tensor,                                           # (B, S, ...) same trailing dims
    suffix_mask: Optional[torch.Tensor],                            # (B, S) bool or None
    fill: Union[int, float],                                        # value for the padded tail
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Join two right-padded batches row-wise into ``[prefix valid][suffix valid]``.

    Returns the merged batch, its key-padding mask, and the per-row index where
    the suffix begins.
    """

    if prefix.shape[2:] != suffix.shape[2:]:
        raise ValueError(
            f"prefix and suffix must share trailing dims, got "
            f"{tuple(prefix.shape[2:])} and {tuple(suffix.shape[2:])}"
        )

    B, P = prefix.shape[:2]
    S = suffix.shape[1]
    device = prefix.device

    prefix_mask = valid_mask(prefix_mask, B, P, device)
    suffix_mask = valid_mask(suffix_mask, B, S, device)

    offset = prefix_mask.sum(1)                                     # (B,) suffix start
    total = offset + suffix_mask.sum(1)                             # (B,) merged length
    L = int(total.max())

    merged = torch.full(
        (B, L, *prefix.shape[2:]), fill, dtype=prefix.dtype, device=device
    )
    rows = torch.arange(B, device=device)[:, None]

    # Both inputs are right-padded, so a valid token at index j already sits at
    # its own offset within its half; only the suffix has to be shifted along.
    cols = torch.arange(P, device=device).expand(B, P)
    merged[rows.expand(B, P)[prefix_mask], cols[prefix_mask]] = prefix[prefix_mask]

    cols = offset[:, None] + torch.arange(S, device=device)
    merged[rows.expand(B, S)[suffix_mask], cols[suffix_mask]] = suffix[suffix_mask]

    merged_mask = torch.arange(L, device=device)[None, :] < total[:, None]

    return merged, merged_mask, offset                              # (B, L, ...), (B, L), (B,)
