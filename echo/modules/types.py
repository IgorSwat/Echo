from __future__ import annotations

from dataclasses import dataclass

import torch

# Per-layer key/value cache for the transformer decoder.  One (k, v) tuple per
# block, both tensors of shape ``(B, num_heads, seq_len, head_dim)``.  ``None``
# entries signal an empty cache (used when the cache is first created or when a
# layer has not yet been populated).
KVCache = list[tuple[torch.Tensor, torch.Tensor] | None]
