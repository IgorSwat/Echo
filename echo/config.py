"""Central configuration for the Echo text-to-speech model.

The file is organized in clearly separated sections, one per model
component, plus a "common" section for hyperparameters shared across the
whole pipeline and a "constants" section for fixed values that are part of
the problem definition rather than tunable hyperparameters.
"""

# =========
# Constants
# =========

# Codebook geometry ----------------------------------------------------------
# Each audio frame is represented as a stack of RVQ codebooks.
NUM_CODEBOOKS: int = 16          # number of codebook layers per time step
CODEC_VOCAB_SIZE: int = 2048     # number of real token values per codebook
MASK_TOKEN_ID: int = CODEC_VOCAB_SIZE  # 2048 : reserved MASK token id
TEXT_VOCAB_SIZE: int = 128       # number of distinct phoneme tokens (0..127)

# Special tokens --------------------------------------------------------------
# <SEP> and <EOS> are represented by dedicated learned embedding vectors
# (single nn.Parameters), not by entries in any vocab table.
#
# In the *output* logits the EOS token occupies the slot immediately after
# the last real codec token, i.e. index CODEC_VOCAB_SIZE.  The output
# vocabulary therefore has CODEC_VOCAB_SIZE + 1 classes.
EOS_TOKEN_ID: int = CODEC_VOCAB_SIZE            # 2048
OUTPUT_VOCAB_SIZE: int = CODEC_VOCAB_SIZE + 1   # 2049  (codec tokens + EOS)

# Padding tokens --------------------------------------------------------------
# Needed for batched training.  Each modality gets its own pad id.
#
# TEXT_PAD_ID reuses index 0 of the text embedding table -- this is the
# phoneme-vocab "<pad>" slot, which never appears in a real phoneme
# sequence, so it is effectively a dedicated learned pad embedding.
#
# AUDIO_PAD_ID is an *extra* row appended to the audio embedding table
# (audio_embedding has CODEC_VOCAB_SIZE + 1 rows; the last row is the pad).
# Note: this coincides numerically with EOS_TOKEN_ID, but the *input* and
# *output* spaces are separate, so there is no conflict.
TEXT_PAD_ID: int = 0
AUDIO_PAD_ID: int = CODEC_VOCAB_SIZE            # 2048 : extra row in audio_embedding

# Sequence bounds ------------------------------------------------------------
MAX_AUDIO_LENGTH: int = 512      # maximum number of audio time steps (codec temporal dimension)
MAX_TEXT_LENGTH: int = 512       # maximum number of text time steps (phoneme tokens)


# =======================
# AR Decoder Hyperparams
# =======================

AR_D_MODEL: int = 768       # hidden dimension
AR_N_HEADS: int = 12        # number of attention heads (head_dim = 64)
AR_D_FF: int = 3072         # feed-forward intermediate dimension (4 x d_model)
AR_N_LAYERS: int = 8        # number of transformer decoder layers
AR_DROPOUT: float = 0.25     # dropout probability


# ====================
# AR Decoder Training
# ====================

AR_BATCH_SIZE: int = 16          # samples per training step
AR_LEARNING_RATE: float = 3e-4   # peak AdamW learning rate
AR_WEIGHT_DECAY: float = 0.1     # AdamW weight decay (non-bias/no-norm params)
AR_BETAS: tuple = (0.9, 0.95)    # AdamW beta1, beta2 (GPT-style)
AR_GRAD_CLIP: float = 1.0        # max gradient norm (0 to disable)
AR_LABEL_SMOOTHING: float = 0.1  # cross-entropy label smoothing
AR_EOS_LOSS_WEIGHT: float = 2.0  # extra weight on the EOS-prediction position
AR_WARMUP_STEPS: int = 1000     # linear LR warmup steps
AR_NUM_EPOCHS: int = 20          # training epochs
AR_VAL_FRACTION: float = 0.02    # fraction of data used for validation
AR_LOG_EVERY: int = 50           # log training loss every N steps
AR_VAL_EVERY: int = 1000         # evaluate on val set every N steps
AR_SAVE_EVERY: int = 2000        # save checkpoint every N steps
AR_SEED: int = 1337              # RNG seed for shuffling / splits
AR_NUM_WORKERS: int = 4          # DataLoader workers


# =======================
# NAR Decoder Hyperparams
# =======================
# The NAR decoder is a bidirectional encoder (no causal mask, no EOS, no
# text) that predicts layer i+1 codec tokens given layer i tokens.  It uses
# 16 separate embedding tables — one per codebook layer — with weight-tying
# between the input embedding of the *target* layer and the output
# projection.  Typically ~half the capacity of the AR decoder.

NAR_D_MODEL: int = 512       # hidden dimension
NAR_N_HEADS: int = 8         # number of attention heads (head_dim = 64)
NAR_D_FF: int = 2048         # feed-forward intermediate dimension (4 x d_model)
NAR_N_LAYERS: int = 12        # number of encoder layers
NAR_DROPOUT: float = 0.25    # dropout probability


# ====================
# NAR Decoder Training
# ====================

NAR_BATCH_SIZE: int = 16          # samples per training step
NAR_LEARNING_RATE: float = 3e-4   # peak AdamW learning rate
NAR_WEIGHT_DECAY: float = 0.1     # AdamW weight decay (non-bias/no-norm/no-embedding)
NAR_BETAS: tuple = (0.9, 0.95)    # AdamW beta1, beta2
NAR_GRAD_CLIP: float = 1.0        # max gradient norm (0 to disable)
NAR_LABEL_SMOOTHING: float = 0.1  # cross-entropy label smoothing
# Per-layer loss weighting.  The NAR predicts codebooks 1..15 (the 2nd through
# last codebook).  Codebook i is given weight ``NAR_LAYER_WEIGHT_DECAY ** (i-1)``
# in the weighted-average loss, so the 2nd codebook (i=1) has the highest impact
# and each subsequent codebook contributes geometrically less.
NAR_LAYER_WEIGHT_DECAY: float = 0.9
NAR_WARMUP_STEPS: int = 1000     # linear LR warmup steps
NAR_NUM_EPOCHS: int = 20          # training epochs
NAR_VAL_FRACTION: float = 0.02    # fraction of data used for validation
NAR_LOG_EVERY: int = 50           # log training loss every N steps
NAR_VAL_EVERY: int = 1000         # evaluate on val set every N steps
NAR_SAVE_EVERY: int = 2000        # save checkpoint every N steps
NAR_SEED: int = 1337              # RNG seed for shuffling / splits
NAR_NUM_WORKERS: int = 4          # DataLoader workers
