#!/usr/bin/env python3
"""Print parameter counts for both decoder models.

Run:
    python scripts/count_params.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from echo.ar_decoder import ARDecoder
from echo.nar_decoder import NARDecoder
from echo.config import (
    AR_D_FF, AR_D_MODEL, AR_N_HEADS, AR_N_LAYERS,
    NAR_D_FF, NAR_D_MODEL, NAR_N_HEADS, NAR_N_LAYERS,
)


def main() -> None:
    print("---------  AR Decoder  ---------")
    ar = ARDecoder(d_model=AR_D_MODEL, n_heads=AR_N_HEADS,
                   d_ff=AR_D_FF, n_layers=AR_N_LAYERS)
    print(f"  Total parameters:          {ar.num_parameters():,}")
    print(f"  Non-embedding parameters:  {ar.num_parameters(exclude_embeddings=True):,}")

    print()
    print("---------  NAR Decoder ---------")
    nar = NARDecoder(d_model=NAR_D_MODEL, n_heads=NAR_N_HEADS,
                     d_ff=NAR_D_FF, n_layers=NAR_N_LAYERS)
    print(f"  Total parameters:          {nar.num_parameters():,}")
    print(f"  Non-embedding parameters:  {nar.num_parameters(exclude_embeddings=True):,}")

    print()
    total = ar.num_parameters() + nar.num_parameters()
    total_non_emb = (ar.num_parameters(exclude_embeddings=True)
                     + nar.num_parameters(exclude_embeddings=True))
    print(f"  Combined total:            {total:,}")
    print(f"  Combined non-embedding:    {total_non_emb:,}")


if __name__ == "__main__":
    main()