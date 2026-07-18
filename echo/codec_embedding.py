import torch
import torch.nn as nn

from echo import config


class CodecEmbedding(nn.Module):
    """Sum-pooled, per-codebook embedding for the audio codec grid."""

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
        
        # Treat id 0 of every codebook as a neutral / padding token.
        for c in range(self.num_codebooks):
            nn.init.zeros_(self.embedding.weight[c * self.vocab_size])

    def _embed_codes(self, codes: torch.Tensor) -> torch.Tensor:
        # Per-codebook offset broadcast along the codebook (last) axis.
        offsets = torch.arange(self.num_codebooks, device=codes.device) * self.vocab_size
        codes = codes + offsets                       # (B, T, NUM_CODEBOOKS)
        emb = self.embedding(codes)                   # (B, T, NUM_CODEBOOKS, D)
        return emb.sum(dim=-2)                        # (B, T, D)

    def forward(
        self,
        codes: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        t = codes.size(1)
        out = self._embed_codes(codes)

        positions = torch.arange(position_offset, position_offset + t, device=codes.device)

        return out + self.pos_embedding(positions)    # (T, D) broadcasts over (B, T, D)
