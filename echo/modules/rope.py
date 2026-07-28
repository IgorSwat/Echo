import torch


def rope_cos_sin(dim: int, max_pos: int, device: torch.device, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)               # (max_pos, dim//2)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Apply rotary embeddings.
    x:      (B, nh, T, hd)
    cos,sin: (max_pos, hd//2)
    """
    T = x.size(2)
    
    x_rot = x.float().reshape(*x.shape[:-1], -1, 2)              # (B, nh, T, hd//2, 2)
    c = cos[start_pos:start_pos + T].view(1, 1, T, -1)           # (1, 1, T, hd//2)
    s = sin[start_pos:start_pos + T].view(1, 1, T, -1)

    out = torch.empty_like(x_rot)
    out[..., 0] = x_rot[..., 0] * c - x_rot[..., 1] * s
    out[..., 1] = x_rot[..., 0] * s + x_rot[..., 1] * c

    return out.flatten(-2).to(x.dtype)