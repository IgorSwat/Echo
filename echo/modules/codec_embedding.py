import math

import torch
import torch.nn as nn

from echo import config


class CodecEmbedding(nn.Module):
    """
    Sum-pooled, per-codebook embedding for the audio codec grid
    (positional encoding handled jointly at model level).
    """

    def __init__(
        self,
        vocab_size: int = config.CODEC_VOCAB_SIZE,
        num_codebooks: int = config.NUM_CODEBOOKS,
        emb_dim: int = config.CODEC_EMB_DIM,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.emb_dim = emb_dim

        # One linearized table: codebook c, token t -> row c * vocab_size + t.
        self.embedding = nn.Embedding(num_codebooks * vocab_size, emb_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.INIT_STD)

        # Zero-init the pad token embedding for every codebook layer.
        for c in range(self.num_codebooks):
            nn.init.zeros_(self.embedding.weight[c * self.vocab_size + config.CODEC_PAD_ID])

    def embed_codebooks(self, codes: torch.Tensor) -> torch.Tensor:
        """Embed every codebook independently, keeping PAD exactly zero."""
        offsets = torch.arange(self.num_codebooks, device=codes.device) * self.vocab_size
        indices = codes + offsets

        embedding = self.embedding(indices)
        return embedding.masked_fill((codes == config.CODEC_PAD_ID).unsqueeze(-1), 0.0)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Input:  audio codec (B, T, NUM_CODEBOOKS)
        Output: audio embeddings (B, T, D_emb) — no positional encoding.
        """

        emb = self.embed_codebooks(codes)             # (B, T, NUM_CODEBOOKS, D)

        # Codebook are hierarchical, so we want to mix them up to get
        # a single representation for entire sequence in given time step.
        # Since embeddings are pretty vast (17M parameters), we don't need separate linear projections - 
        # a simple summation should be enough.
        return emb.sum(dim=-2) / math.sqrt(self.num_codebooks)  # (B, T, D)
