from __future__ import annotations

from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F_torch

from echo.config import TextEncoderConfig


# =================================
# Text Encoder - Convolution blocks
# =================================

class _GatedConvBlock(nn.Module):
    """
    Implements Gated Convolution mechanism for language,
    described in: https://arxiv.org/abs/1612.08083.

    Input : (B, T, D_in) continuous latent representation of sequence
    Output: (B, T, K, D_out) continuous latent
    """
    def __init__(self, in_dim, out_dim, kernel_size, num_kernels, dropout=0.0):
        super().__init__()

        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = (kernel_size - 1) // 2 # To keep the temporal dimension intact

        # We need num_kernels, each having an output of 2*out_dim
        # For simplicity, we can just use grouped convolution or output all feature maps and reshape
        self.conv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=2 * out_dim * num_kernels,
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            bias=True
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        # NOTE: we can drop transpose() and permute() if needed
        x = x.transpose(1, 2)  # (batch_size, in_dim, seq_len) 
        out = self.conv(x)  # (batch_size, 2*out_dim*num_kernels, seq_len)
        batch_size, _, seq_len = out.shape

        # Reshape to (batch_size, num_kernels, 2*out_dim, seq_len)
        out = out.view(batch_size, self.num_kernels, -1, seq_len)

        # GLU over the 2*out_dim dimension
        # The first half is values, the second half is for GLU gating
        v, g = out.chunk(2, dim=2)
        out = v * torch.sigmoid(g)  # (batch_size, num_kernels, out_dim, seq_len)

        if self.dropout:
            out = self.dropout(out)

        # Rearrange to (batch_size, seq_len, num_kernels, out_dim)
        out = out.permute(0, 3, 1, 2).contiguous()

        return out


class _MultiKernelConv(nn.Module):
    """
    Implements Gated Convolutions with a set of various kernel sizes.
    Uses kernels of size 1 (identity), 3, 5 and 7.

    Input : (B, T, D_in) continuous latent representation of sequence
    Output: (B, T, K, D_out) continuous latent, where K = [k1] + K3 + K5 + K7
    """
    def __init__(self, in_dim, out_dim, k1=True, k3=0, k5=0, k7=0, dropout=0.0):
        super().__init__()
        self.k1 = k1
        # Project the identity (k1) path to out_dim so it concatenates with
        # the conv outputs. When in_dim == out_dim we keep it truly intact
        # (no projection) to preserve the identity semantics.
        self.k1_proj = (
            nn.Linear(in_dim, out_dim, bias=False)
            if k1 and in_dim != out_dim
            else None
        )
        self.conv3 = _GatedConvBlock(in_dim, out_dim, kernel_size=3, num_kernels=k3, dropout=dropout) if k3 > 0 else None
        self.conv5 = _GatedConvBlock(in_dim, out_dim, kernel_size=5, num_kernels=k5, dropout=dropout) if k5 > 0 else None
        self.conv7 = _GatedConvBlock(in_dim, out_dim, kernel_size=7, num_kernels=k7, dropout=dropout) if k7 > 0 else None

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        outs = []
        if self.k1:
            identity = self.k1_proj(x) if self.k1_proj is not None else x
            outs.append(identity.unsqueeze(2))  # (batch_size, seq_len, 1, out_dim)
        if self.conv3:
            outs.append(self.conv3(x))
        if self.conv5:
            outs.append(self.conv5(x))
        if self.conv7:
            outs.append(self.conv7(x))

        # Concatenate along the num_kernels dimension
        out = torch.cat(outs, dim=2)  # (batch_size, seq_len, total_kernels, out_dim)

        return out


class _SelectiveConv(nn.Module):
    """
    Performs channel-wise attention over the multi-kernel convolution outputs.

    Input : (B, T, D_in)
    Output: (B, T, D_out)

    1. Apply MultiKernelConv -> (B, T, K, D_out)
    2. Linear on each channel vector (last dim) -> (B, T, K, D_out)
    3. Average across K -> (B, T, D_out)
    4. Linear -> query q: (B, T, D_out)
    5. Scaled dot-product attention between q and channel vectors, weighted sum -> (B, T, D_out)
    """

    def __init__(self, in_dim, out_dim, k1=True, k3=0, k5=0, k7=0, dropout=0.0):
        super().__init__()

        self.multi_kernel_conv = _MultiKernelConv(
            in_dim=in_dim,
            out_dim=out_dim,
            k1=k1,
            k3=k3,
            k5=k5,
            k7=k7,
            dropout=dropout,
        )

        # Linear layers for obtaining query for channel-wise attention pooling
        self.linear1 = nn.Linear(out_dim, out_dim, bias=True)
        self.linear2 = nn.Linear(out_dim, out_dim, bias=True)

    def forward(self, x):
        # x: (batch_size, seq_len, in_dim)
        conv_out = self.multi_kernel_conv(x)  # (B, T, K, D_out)
        out_dim = conv_out.shape[-1]

        # Transform each channel vector (last dim): (B, T, K, D_out) -> (B, T, K, D_out)
        context = self.linear1(conv_out)

        # Average across the channel (K) dimension -> (B, T, D_out)
        context_vec = context.mean(dim=2)

        # Query vector: (B, T, D_out)
        q = self.linear2(context_vec)

        # Scaled dot-product attention between q and each channel vector (keys)
        # scores: (B, T, K)
        scale = 1.0 / math.sqrt(out_dim)
        scores = (conv_out * q.unsqueeze(2)).sum(dim=-1) * scale
        weights = torch.softmax(scores, dim=-1)  # (B, T, K)

        # Weighted sum of channel vectors -> (B, T, D_out)
        out = (conv_out * weights.unsqueeze(-1)).sum(dim=2)

        return out


# ===========================
# Text Encoder - CNN forehead
# ===========================

class _Forehead(nn.Module):
    """
    A sequence of SelectiveConv layers.

    layers is a tuple of dicts, one per layer. Each dict carries the full
    per-layer config: in_dim, out_dim, k1, k3, k5, k7. dropout is shared
    across all layers.

    Input : (B, T, layers[0]['in_dim'])
    Output: (B, T, layers[-1]['out_dim'])
    """

    def __init__(self, layers, dropout=0.0):
        super().__init__()

        modules = []
        for layer_cfg in layers:
            modules.append(_SelectiveConv(
                in_dim=layer_cfg['in_dim'],
                out_dim=layer_cfg['out_dim'],
                k1=layer_cfg['k1'],
                k3=layer_cfg['k3'],
                k5=layer_cfg['k5'],
                k7=layer_cfg['k7'],
                dropout=dropout,
            ))
        self.layers = nn.ModuleList(modules)

    def forward(self, x):
        # x: (batch_size, seq_len, layers[0]['in_dim'])
        for layer in self.layers:
            x = layer(x)
        return x
    

# ============
# Text Encoder
# ============

class TextEncoder(nn.Module):
    """
    Full text encoder: embedding -> _Forehead -> positional embedding
    -> TransformerEncoder.

    Args:
        config: ``TextEncoderConfig`` instance (see echo.config).

    Input : (B, T) integer token ids (vocab < TEXT_VOCAB_SIZE)
    Output: (B, T, d_hidden)
    """

    def __init__(self, config: TextEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # --- 1. Token embeddings ------------------------------------------
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)

        # --- 2. CNN forehead: maps embedding_dim -> d_hidden --------------
        self.forehead = _Forehead(
            layers=config.layers,
            dropout=config.dropout,
        )

        # --- 3. Learned positional embeddings -----------------------------
        self.pos_embed = nn.Embedding(config.max_text_length, config.d_hidden)

        # --- 4. Bidirectional transformer encoder -------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_hidden,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_hidden),
            enable_nested_tensor=False,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a sequence of text tokens into continuous latents.

        Args:
            tokens: (B, T) integer token ids.
            key_padding_mask: (B, T) boolean, True = padded/ignored time step.
        Returns:
            (B, T, d_hidden) latent representation.
        """
        # ---- Step 1: Token embeddings -> (B, T, embedding_dim) ----------
        x = self.embedding(tokens)

        # ---- Step 2: Forehead -> (B, T, d_hidden) ------------------------
        x = self.forehead(x)

        # ---- Step 3: Add learned positional embeddings -------------------
        T = x.shape[1]
        positions = torch.arange(T, device=x.device).clamp(
            max=self.config.max_text_length - 1
        )
        x = x + self.pos_embed(positions)

        # ---- Step 4: Transformer encoder -> (B, T, d_hidden) ------------
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)

        return x