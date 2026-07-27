from __future__ import annotations

import torch
import torch.nn as nn

from echo import config
from echo.modules.transformer import TransformerDecoder


class CodebookProjector(nn.Module):
    """
    Mini autoregressive transformer operating on the codebook-layer axis.

    Three modes:
      - ``forward``   — teacher-forcing training
      - ``prefill``   — inference: context chunks → logits for layer 0
      - ``step``      — inference: one token embedding → logits for next layer
    """

    def __init__(
        self,
        num_codebooks: int,
        no_context_chunks: int,     # Decides on how many smaller chunks will the context vector be splitted
        d_model: int,               
        token_embedding_dim: int,
        d_context: int,             # Context dimensionality (intermediate_dim in our model)
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.no_context_chunks = no_context_chunks
        self.num_codebooks = num_codebooks
        self.seq_len = no_context_chunks + num_codebooks

        self.tok_proj = nn.Linear(token_embedding_dim, d_model)
        self.ctx_proj = nn.Linear(d_context, no_context_chunks * d_model)
        self.heads = nn.ModuleList(
            nn.Linear(d_model, vocab_size) for _ in range(num_codebooks)
        )

        # Additional per-layer embeddings - maybe necessary, maybe not. Certainly cost almost nothing.
        self.pos_embed = nn.Embedding(self.seq_len, d_model) 

        self.transformer = TransformerDecoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.tok_proj, self.ctx_proj, *self.heads]:
            nn.init.normal_(m.weight, mean=0.0, std=config.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_embed.weight, mean=0.0, std=config.init_std)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        embeddings:  (B, T, C, token_embedding_dim)
        hidden:      (B, T, d_context)
        Returns:     (B, T, C, vocab_size)
        """

        B, T, C, _D = embeddings.shape
        K = self.no_context_chunks
        device = embeddings.device

        # First, 'split' one big context vector into a few smaller ones,
        # and append to the start of the sequence as a CLS equivalent.
        ctx = self.ctx_proj(hidden).reshape(B, T, K, self.d_model)         # (B, T, K, d_model)
        ctx = ctx + self.pos_embed(torch.arange(K, device=device))

        # Then, project input token embeddings to model hidden dimension (and apply positional embeddings).
        tok = self.tok_proj(embeddings)                                     # (B, T, C, d_model)
        tok = tok + self.pos_embed(torch.arange(K, K + C, device=device))

        # Now concatenate into one sequence.
        x = torch.cat([ctx, tok], dim=2)                                    # (B, T, K+C, d_model)
        x = x.reshape(B * T, K + C, self.d_model)

        # Run mini-transformer.
        # We skip KV cache because of how short the sequence is.
        out, _ = self.transformer(x, kv_cache=None)                         # (B*T, K+C, d_model)

        pred_hidden = out[:, K - 1 : K - 1 + C]                             # (B*T, C, d_model)
        logits = torch.stack(
            [head(pred_hidden[:, codebook]) for codebook, head in enumerate(self.heads)],
            dim=1,
        )                                                                    # (B*T, C, V)

        return logits.reshape(B, T, C, -1)

    def prefill(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden:  (B, d_context)
        Returns: logits for codebook layer 0  (B, vocab_size)
        """

        B = hidden.size(0)
        K = self.no_context_chunks
        device = hidden.device

        ctx = self.ctx_proj(hidden).reshape(B, K, self.d_model)
        ctx = ctx + self.pos_embed(torch.arange(K, device=device))
        self._ctx = ctx                                                     # (B, K, d_model)
        self._tokens: list[torch.Tensor] = []                                # accumulate per-layer embeddings

        # Based only on the context (splitted into chunks), we predict the first codebook token.
        out, _ = self.transformer(ctx, kv_cache=None)                       # (B, K, d_model)
        first_token_logits = self.heads[0](out[:, -1])                       # (B, V)
        
        return first_token_logits

    def step(self, token_embedding: torch.Tensor) -> torch.Tensor:
        """
        token_embedding:  (B, token_embedding_dim)
        Returns:          logits for the next codebook layer  (B, vocab_size)
        """
        
        c = len(self._tokens)
        device = token_embedding.device

        tok = self.tok_proj(token_embedding)                                  # (B, d_model)
        tok = tok + self.pos_embed(torch.tensor([self.no_context_chunks + c], device=device))
        self._tokens.append(tok.unsqueeze(1))                            # (B, 1, d_model)

        x = torch.cat([self._ctx] + self._tokens, dim=1)                      # (B, K + c + 1, d_model)
        out, _ = self.transformer(x, kv_cache=None)                           # (B, K + c + 1, d_model)

        return self.heads[c + 1](out[:, -1])                                  # (B, V)
