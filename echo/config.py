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


# ======
# Common
# ======

D_LATENT: int = 512              # latent dimension shared by encoders/decoder


# =============
# Audio Encoder
# =============

class AudioEncoderConfig:
    # --- Input / embedding ---------------------------------------------------
    embedding_dim: int = D_LATENT        # d_e  : per-codebook embedding dim
    num_codebooks: int = NUM_CODEBOOKS   
    vocab_size: int = CODEC_VOCAB_SIZE   
    mask_token_id: int = MASK_TOKEN_ID  
    embedding_dropout: float = 0.0       # dropout after codebook summation

    # --- Positional encoding -------------------------------------------------
    max_audio_length: int = MAX_AUDIO_LENGTH  # max. time steps; determines the
                                              # learned positional embedding table

    # --- Transformer encoder -------------------------------------------------
    d_f: int = D_LATENT                  # output (and model) latent dimension
    num_layers: int = 3
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1                 # residual / FFN dropout
    attention_dropout: float = 0.0       # dropout inside attention weights


# ============
# Text Encoder
# ============

class TextEncoderConfig:
    # --- Input / embedding ---------------------------------------------------
    vocab_size: int = TEXT_VOCAB_SIZE     
    embedding_dim: int = 64               # d_emb : token embedding dimension

    # --- Forehead (SelectiveConv stack) -------------------------------------
    d_hidden: int = D_LATENT              # 512 : final hidden dimension
    layers: tuple = (
        dict(in_dim=64,  out_dim=128, k1=True, k3=8, k5=8, k7=8),
        dict(in_dim=128, out_dim=192, k1=True, k3=8, k5=4, k7=4),
        dict(in_dim=192, out_dim=256, k1=True, k3=6, k5=4, k7=2),
        dict(in_dim=256, out_dim=320, k1=True, k3=4, k5=2, k7=0),
        dict(in_dim=320, out_dim=512, k1=True, k3=4, k5=0, k7=0),
    )
    dropout: float = 0.1

    # --- Positional encoding -------------------------------------------------
    max_text_length: int = MAX_TEXT_LENGTH  # max. time steps; determines the
                                           # learned positional embedding table

    # --- Transformer encoder -------------------------------------------------
    num_layers: int = 2
    num_heads: int = 8
    ffn_dim: int = 2048
    transformer_dropout: float = 0.1           # residual / FFN dropout
    attention_dropout: float = 0.0              # dropout inside attention weights


# ================
# Decoder Planner
# ================

class DecoderPlannerConfig:
    # --- Hidden state --------------------------------------------------------
    d_plan: int = 768                    # decoder hidden state dim; may differ
                                         # from d_hid (the encoder latent dim)
    d_hid: int = D_LATENT               

    # --- Transformer decoder -------------------------------------------------
    num_layers: int = 8
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1                 # residual / FFN dropout
    attention_dropout: float = 0.0       # dropout inside attention weights


# =================
# Decoder Executor
# =================

class DecoderExecutorConfig:
    # --- Grid geometry -------------------------------------------------------
    chunk_size: int = 10                  # number of new time steps filled per call
    context_steps: int = 2                # already-filled time steps fed back from
                                          # the previous chunk for continuity (+2)
    num_codebooks: int = NUM_CODEBOOKS    
    vocab_size: int = CODEC_VOCAB_SIZE    
    mask_token_id: int = MASK_TOKEN_ID    

    # --- Planner conditioning ------------------------------------------------
    d_plan: int = 768                      # planner hidden dim (input to executor)
    d_exec: int = 512                      # executor model dim; may differ from
                                           # the shared CodecEmbedding's dim, in
                                           # which case a linear projection maps
                                           # codec embeddings to d_exec

    # --- Transformer encoder -------------------------------------------------
    num_layers: int = 6
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1                  # residual / FFN dropout
    attention_dropout: float = 0.0        # dropout inside attention weights


# ========
# Training
# ========

class TrainingConfig:
    # --- Optimizer ----------------------------------------------------------
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 1.0

    # --- Per-component LR scaling (relative to learning_rate) ---------------
    # The planner produces much larger gradients than other components; give
    # it a lower LR to keep training stable. The codec embedding barely gets
    # gradients (it's summed across 16 codebooks), so boost it.
    lr_planner_scale: float = 0.3
    lr_executor_scale: float = 1.0
    lr_encoder_scale: float = 1.0
    lr_embedding_scale: float = 5.0

    # --- Data ----------------------------------------------------------------
    batch_size: int = 8
    num_workers: int = 2
    max_audio_length: int = MAX_AUDIO_LENGTH
    max_text_length: int = MAX_TEXT_LENGTH

    # --- Executor masking ---------------------------------------------------
    mask_ratio: float = 0.6            # fraction of chunk positions masked for MLM

    # --- Loss weights --------------------------------------------------------
    lambda_stop: float = 1.0           # weight of stop-head BCE loss

    # --- Logging / checkpointing --------------------------------------------
    log_interval: int = 50             # steps between loss logs
    save_interval: int = 2000         # steps between checkpoint saves
    eval_interval: int = 2000          # steps between eval runs

    # --- Phoneme vocab special tokens ---------------------------------------
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
