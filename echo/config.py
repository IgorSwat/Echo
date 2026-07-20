from __future__ import annotations

import json
from pathlib import Path

_cfg_path = Path(__file__).resolve().parent.parent / "models" / "config.json"
_cfg = json.loads(_cfg_path.read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# Codebook geometry
# ---------------------------------------------------------------------------
NUM_CODEBOOKS: int = _cfg["num_codebooks"]
CODEC_VOCAB_SIZE: int = _cfg["vocab_size"]["audio"]
TEXT_VOCAB_SIZE: int = _cfg["vocab_size"]["text"]

# ---------------------------------------------------------------------------
# Sequence bounds
# ---------------------------------------------------------------------------
MAX_AUDIO_LENGTH: int = _cfg["limits"]["audio_seq_len"]
MAX_TEXT_LENGTH: int = _cfg["limits"]["text_seq_len"]
# Upper bound on the full embedded sequence length:
#   <BOS> + ref_text + <REF_TEXT_EOS> + ref_audio + <REF_CODEC_EOS>
#   + text + <TEXT_EOS> + audio
MAX_SEQ_LEN: int = 2 * MAX_TEXT_LENGTH + 2 * MAX_AUDIO_LENGTH + 4

# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------
TEXT_PAD_ID: int = _cfg["special_tokens"]["text_pad"]
CODEC_PAD_ID: int = _cfg["special_tokens"]["audio_pad"]
BOS_ID = _cfg["special_tokens"]["bos"]
REF_TEXT_EOS_ID = _cfg["special_tokens"]["ref_text_eos"]
REF_CODEC_EOS_ID = _cfg["special_tokens"]["ref_codec_eos"]
TEXT_EOS_ID = _cfg["special_tokens"]["text_eos"]
EOS_ID = _cfg["special_tokens"]["eos"]

CODEC_LOGIT_DIM: int = CODEC_VOCAB_SIZE + 1   # + extra EOS token which does not apper in input sequences

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
D_EMB: int = _cfg["embedding_dim"]
D_MODEL: int = _cfg["decoder"]["hidden_dim"]
D_REPR: int = _cfg["intermediate_dim"]

DROPOUT: float = _cfg["decoder"]["dropout"]

# ---------------------------------------------------------------------------
# Embedding tables
# ---------------------------------------------------------------------------
TEXT_EMB_DIM: int = D_EMB
TEXT_POS_SIZE: int = MAX_TEXT_LENGTH
CODEC_EMB_DIM: int = D_EMB
CODEC_POS_SIZE: int = MAX_AUDIO_LENGTH

# ---------------------------------------------------------------------------
# Transformer decoder
# ---------------------------------------------------------------------------
NUM_LAYERS: int = _cfg["decoder"]["no_layers"]
NUM_HEADS: int = _cfg["decoder"]["no_heads"]
FFN_DIM: int = _cfg["decoder"]["ffn_dim"]
FFN_GLU: bool = _cfg["decoder"]["ffn_glu"]

# ---------------------------------------------------------------------------
# Prediction heads
# ---------------------------------------------------------------------------
PRED_NUM_HEADS: int = _cfg["heads"]["no_heads"]
PRED_HIDDEN_DIM: int = _cfg["heads"]["hidden_dim"]
PRED_NUM_LAYERS: int = _cfg["heads"]["no_layers"]
PRED_DROPOUT: float = _cfg["heads"]["dropout"]

# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------
INIT_STD: float = _cfg["init_std"]