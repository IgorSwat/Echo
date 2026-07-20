from echo import config

import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    """
    Phoneme token embedding + learned positional encoding.
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        pos_size: int,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(pos_size, emb_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=config.INIT_STD)
        nn.init.zeros_(self.token_embed.weight[config.TEXT_PAD_ID])
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=config.INIT_STD)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, T, emb_dim)"""
        T = text.size(1)
        pos_ids = torch.arange(T, device=text.device)
        return self.token_embed(text) + self.pos_embed(pos_ids)