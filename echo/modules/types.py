from __future__ import annotations

import torch


# --- Attention key/value caches ---
LayerCache = tuple[torch.Tensor, torch.Tensor]
HybridLayerCache = tuple[LayerCache, LayerCache]  # Seperate for queries (growing) and keys (fixed)

# What a whole stack hands back: one entry per block, in block order.
KVCache = list[LayerCache]
HybridKVCache = list[HybridLayerCache]
