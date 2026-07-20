from echo import config

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# This is a decoder-like version of self attention.
# The difference between this and bidirectional attention is, that this one
# can only look backwards, and uses casual masking to achieve this.
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with an explicit (optional) KV cache.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        # Runtime check to ensure proper head vs hidden shapes
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.nh = num_heads
        self.hd = d_model // num_heads

        # Fused QKV projection (one matmul instead of three).
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        self.attn_drop_value = dropout	
        self.resid_drop = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.qkv.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.qkv.bias)
        nn.init.normal_(self.proj.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        # batch_size, temporal, hidden_dim
        B, T, D = x.shape

		# In no-KV-cache or prefill mode, we calculate QKV for the entire input sequence.
        # In KV-cache single-step mode, T = 1 and we only calculate QKV for a single element.
        qkv = self.qkv(x)                       # (B, T, 3D)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, T, self.nh, self.hd).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
        v = v.view(B, T, self.nh, self.hd).transpose(1, 2)

        # In KV-cache single-step mode, we get the cached values from passed cache_kv tuple,
        # and append the newly calculated KV at the end.
        # Not needed in no-KV-cache mod, since then k and v already cover entire sequence.
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)            # always return K,V — caller decides whether to keep/discard

        # In prefill (no cache) we apply a causal mask; 
		# In KV-cache single-step mode the single new query sits at the end of the cached sequence, 
		# so attending to all cached keys is already causal and no mask is needed.
        is_causal = kv_cache is None
        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop_value if self.training else 0.0,
            is_causal=is_causal,
        )                                          # (B, nh, T, hd)
        
		# Now we want to mix the information obtained from per-head attentions.
        # Actually a similar idea to mixing kernels in multi-kernel conv in Protophone.
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        out = self.resid_drop(self.proj(out))
        
        return out, new_kv