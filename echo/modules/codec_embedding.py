from __future__ import annotations

import torch
import torch.nn as nn

from echo import config


class CodecEmbedding(nn.Module):
    """
    Per-codebook token embeddings with an optional MLP compressor that
    fuses per-layer info into a single vector per time step.
    """

    def __init__(
        self,
        vocab_size: int,                # Number of possible tokens
        num_codebook_layers: int,            # Number of layers (16 by default)
        token_embedding_dim: int,      # Embedding per single token
        codebook_embedding_dim: int,   # Fused embedding per codebook frame
        mlp_hidden_dim: int,           # Dimensionality of MLP hidden layer(s)
        mlp_num_layers: int,           # Total linear layers (e.g. 2 = 1 hidden + 1 output)
        mlp_dropout: float,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebook_layers = num_codebook_layers
        self.token_embedding_dim = token_embedding_dim
        self.codebook_embedding_dim = codebook_embedding_dim

        # All layers have it's own embedding table, but we fuse them together with linearization technique.
        self.embedding = nn.Embedding(num_codebook_layers * vocab_size, token_embedding_dim)

        in_dim = num_codebook_layers * token_embedding_dim
        layers: list[nn.Module] = []
        if mlp_num_layers == 1:
            layers.append(nn.Linear(in_dim, codebook_embedding_dim))
        else:
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.GELU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            for _ in range(mlp_num_layers - 2):
                layers.append(nn.Linear(mlp_hidden_dim, mlp_hidden_dim))
                layers.append(nn.GELU())
                if mlp_dropout > 0:
                    layers.append(nn.Dropout(mlp_dropout))
            layers.append(nn.Linear(mlp_hidden_dim, codebook_embedding_dim))

        self.fuse_mlp = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.INIT_STD)
        for c in range(self.num_codebook_layers):
            nn.init.zeros_(self.embedding.weight[c * self.vocab_size + config.CODEC_PAD_ID])

        for m in self.fuse_mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=config.INIT_STD)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed_codebooks(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Look up per-codebook token embeddings. Shape: (..., T, C) -> (..., T, C, Dt).
        """

        offsets = torch.arange(self.num_codebook_layers, device=codes.device) * self.vocab_size
        indices = codes + offsets
        embedding = self.embedding(indices)

        # Mask-out the pad tokens
        return embedding.masked_fill((codes == config.CODEC_PAD_ID).unsqueeze(-1), 0.0)

    def forward(self, codes: torch.Tensor, fuse: bool = True) -> torch.Tensor:
        """
        Input:  audio codec (B, T, C)
        Output: if compress=False -> (B, T, C, token_embedding_dim)
                if compress=True  -> (B, T, codebook_embedding_dim) after MLP fusion
        """

        emb = self.embed_codebooks(codes)            # (B, T, C, token_embedding_dim)
        if not fuse:
            return emb

        B, T = codes.shape[:2]
        flat = emb.reshape(B, T, -1)                # (B, T, C * token_embedding_dim)
        fused_emb = self.fuse_mlp(flat)

        return fused_emb