"""Central configuration for the Echo text-to-speech model.

The file is organized in clearly separated sections, one per model
component, plus a "common" section for hyperparameters shared across
the whole pipeline and a "constants" section for fixed values that are
part of the problem definition rather than tunable hyperparameters.
"""

# =========
# Constants
# =========

# Codebook geometry ----------------------------------------------------------
# Each audio frame is represented as a stack of RVQ codebooks.
NUM_CODEBOOKS: int = 16          # number of codebook layers per time step
CODEC_VOCAB_SIZE: int = 2048     # number of real token values per codebook
MASK_TOKEN_ID: int = CODEC_VOCAB_SIZE  # 2048 : reserved MASK token id
TEXT_VOCAB_SIZE: int = 256       # number of distinct phoneme tokens

# Sequence bounds ------------------------------------------------------------
MAX_AUDIO_LENGTH: int = 512      # maximum number of audio time steps (codec temporal dimension)
MAX_TEXT_LENGTH: int = 512       # maximum number of text time steps (phoneme tokens)


