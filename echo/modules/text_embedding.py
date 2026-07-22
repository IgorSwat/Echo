from echo import config

import torch
import torch.nn as nn


class TextEmbedding(nn.Module):
    """
    Phoneme token embedding (positional encoding handled jointly at model level).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=config.text_pad_id)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=config.init_std)
        nn.init.zeros_(self.token_embed.weight[config.text_pad_id])

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """(B, T) → (B, T, emb_dim)"""
        embedding = self.token_embed(text)

        # Mask PAD tokens
        return embedding.masked_fill((text == config.text_pad_id).unsqueeze(-1), 0.0)
