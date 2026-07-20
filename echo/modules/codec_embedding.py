import torch
import torch.nn as nn

from echo import config


class CodecEmbedding(nn.Module):
    """
    Sum-pooled, per-codebook embedding for the audio codec grid.
    """

    def __init__(
        self,
        vocab_size: int = config.CODEC_VOCAB_SIZE,
        num_codebooks: int = config.NUM_CODEBOOKS,
        emb_dim: int = config.CODEC_EMB_DIM,
        max_len: int = config.CODEC_POS_SIZE,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.num_codebooks = num_codebooks
        self.emb_dim = emb_dim
        self.max_len = max_len

        # One linearized table: codebook c, token t -> row c * vocab_size + t.
        self.embedding = nn.Embedding(num_codebooks * vocab_size, emb_dim)

        # Learned positional embedding shared across the whole audio segment.
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=config.INIT_STD)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=config.INIT_STD)
        
        # Zero-init the pad token embedding for every codebook layer.
        for c in range(self.num_codebooks):
            nn.init.zeros_(self.embedding.weight[c * self.vocab_size + config.CODEC_PAD_ID])

    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # Per-codebook offset broadcast along the codebook (last) axis.
        # For example, second layer of codebook has offset = vocab_size = 2048.
        offsets = torch.arange(self.num_codebooks, device=codes.device) * self.vocab_size

        # Now we add layer offset to each token.
        # For example, token 5 in second layer becomes 2048 + 5 = 2053.
        codes = codes + offsets                       # (B, T, NUM_CODEBOOKS)

        # After extending the token indices, we can treat 2D embedding
        # as a simple 1D embeddings task.
        emb = self.embedding(codes)                   # (B, T, NUM_CODEBOOKS, D)

        # Codebook are hierarchical, so we want to mix them up to get
        # a single representation for entire sequence in given time step.
        # Since embeddings are pretty vast (17M parameters), we don't need separate linear projections - 
        # a simple summation should be enough.
        return emb.sum(dim=-2)                        # (B, T, D)

    def forward(
        self,
        codes: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Input: audio codec (B, T, 16)
        Output: audio embeddings (B, T, D_emb)
        """

        # Pure embeddings
        t = codes.size(1)   # Important: we assume temporal-first layout.
        out = self._embed_codes(codes)

        # Position embeddings
        positions = torch.arange(position_offset, position_offset + t, device=codes.device)
        out = out + self.pos_embedding(positions)   # (T, D) broadcasts over (B, T, D)

        return out
