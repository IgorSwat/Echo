"""
Supertonic-3 vector estimator configuration.

Constants are ported from supertonic-example/tts.json. The only adaptation to
Echo's data pipeline is the text vocabulary:

  * text_vocab_size 256 (Echo's phoneme vocab) instead of 8322 unicode chars.

As in the original, the estimator operates on FOLDED latents: the 24 codec
channels x 6 sub-frames are packed into 144 channels at 1/6 of the codec frame
rate (folding is a data-side reshape, see supertonic/fold.py).
"""

from typing import Any

# --- Data-level dims ---
latent_dim = 24                     # base codec latent channels
chunk_compress_factor = 6           # temporal fold factor (sub-frames per folded frame)
folded_dim = latent_dim * chunk_compress_factor   # model input/output dim (144)
text_vocab_size = 256

# --- Conditioning ---
text_embedding_dim = 256            # text encoder output / cross-attention kv dim
time_dim = 64                       # sinusoidal time embedding dim
time_hdim = 256                     # time MLP hidden dim
time_scale = 1000.0                 # t is multiplied by this before the frequencies

# --- Style tokens (GST-style global style conditioning) ---
n_style = 50
style_dim = 256

# --- Backbone (vector field) ---
hidden_dim = 512
intermediate_dim = 2048
kernel_size = 5
n_blocks = 4                        # macro-blocks
block_dilations = (1, 2, 4, 8)      # leading dilated ConvNeXt stack of each macro-block
last_dilations = (1, 1, 1, 1)       # trailing ConvNeXt stack

# --- Text cross-attention (rotary, length-normalized positions) ---
text_attn_heads = 8
rotary_dim = 32
attn_scale = 16.0                   # score scale 1/16 (not 1/sqrt(d))

# --- Style attention (tanh keys) ---
style_attn_heads = 2
style_attn_units = 256

# --- Text encoder ---
text_encoder_intermediate_dim = 1024
text_encoder_dilations = (1, 1, 2, 2, 4, 4)
text_encoder_attn_layers = 4
text_encoder_attn_heads = 4
text_encoder_rel_window = 4         # relative position window of the VITS attention

init_std = 0.02
layer_norm_eps = 1e-6

# --- Main processing stack ---
# Mirrors the original macro-block layout: each of the `n_blocks` macro-blocks is
#   dilated ConvNeXt stack -> +time -> conv -> rotary text cross-attention ->
#   conv -> tanh style attention
# followed by one trailing ConvNeXt stack. All blocks keep `hidden_dim`.
blocks: list[dict[str, Any]] = []
for _ in range(n_blocks):
    blocks += [
        {"type": "convnext_stack", "dim": hidden_dim, "intermediate_dim": intermediate_dim,
         "kernel_size": kernel_size, "dilations": list(block_dilations)},
        {"type": "time_condition", "dim": hidden_dim, "time_dim": time_dim},
        {"type": "convnext_stack", "dim": hidden_dim, "intermediate_dim": intermediate_dim,
         "kernel_size": kernel_size, "dilations": [1]},
        {"type": "text_cross_attention", "dim": hidden_dim, "text_dim": text_embedding_dim,
         "num_heads": text_attn_heads, "rotary_dim": rotary_dim, "scale": attn_scale},
        {"type": "convnext_stack", "dim": hidden_dim, "intermediate_dim": intermediate_dim,
         "kernel_size": kernel_size, "dilations": [1]},
        {"type": "style_attention", "dim": hidden_dim, "style_dim": style_dim,
         "n_units": style_attn_units, "num_heads": style_attn_heads, "scale": attn_scale},
    ]
blocks += [
    {"type": "convnext_stack", "dim": hidden_dim, "intermediate_dim": intermediate_dim,
     "kernel_size": kernel_size, "dilations": list(last_dilations)},
]
