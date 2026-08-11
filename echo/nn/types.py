from __future__ import annotations

import torch


# Keys + Values cached by a single layer.
LayerCache = tuple[torch.Tensor, torch.Tensor]
# A hybrid layer caches both:
# - the self-attention stream (grows every step)
# - the cross-attention context (built once, then fixed)
HybridLayerCache = tuple[LayerCache, LayerCache]

# What a whole stack hands back: one entry per block, in block order.
KVCache = list[LayerCache]
HybridKVCache = list[HybridLayerCache]
