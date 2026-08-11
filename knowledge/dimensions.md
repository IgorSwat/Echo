# Echo model — dimension parameters

Echo is a **decoder-only transformer** for TTS. Inputs (text phonemes, `<sep>`, audio codec frames) are embedded into a shared dimension, processed by a causal transformer decoder, then projected through per-codebook prediction MLPs to output logits.

```
[text_emb] + [sep] + [audio_emb]
     ↓           ↓          ↓
  embedding_dim ────────────┘
     ↓   input_proj
  decoder.hidden_dim  ─── transformer (no_layers × [attn + FFN])
     ↓   output_proj
  intermediate_dim  ─── heads (no_heads × MLP)
     ↓
  heads.logit_dim  (per codebook: vocab + EOS)
```

## Parameters

| Param | Config key | Default | Role |
|-------|-----------|---------|------|
| `embedding_dim` | `embedding_dim` | 768 | Width of text phoneme embeddings, codec embeddings, and the `<sep>` token. All three must share this dimension to form a uniform input sequence. |
| `decoder.hidden_dim` | `decoder.hidden_dim` | 768 | Transformer residual stream width across all decoder blocks. Head dim = `hidden_dim / no_heads` (must divide evenly). |
| `intermediate_dim` | `intermediate_dim` | 768 | Width of the transformer output representation fed into prediction heads. |
| `decoder.no_heads` | `decoder.no_heads` | 12 | Number of attention heads per block. Controls how many parallel attention patterns the model can learn. |
| `decoder.no_layers` | `decoder.no_layers` | 6 | Depth — number of decoder blocks stacked. Deeper = more capacity, slower inference. |
| `decoder.ffn_dim` | `decoder.ffn_dim` | 3072 | Inner width of the feed-forward sublayer. Standard ratio is 4× `hidden_dim`. |
| `decoder.ffn_glu` | `decoder.ffn_glu` | false | Gated FFN variant — trades ~15% more params for better throughput at equal width. |
| `decoder.dropout` | `decoder.dropout` | 0.1 | Dropout in attention and FFN sublayers. |
| `heads.no_heads` | `heads.no_heads` | 16 | Number of prediction MLPs — **must equal `num_codebooks`** (one head per audio codec layer). |
| `heads.no_layers` | `heads.no_layers` | 2 | Depth of each prediction MLP. 2 is sufficient; 3 trades speed for marginal quality. |
| `heads.hidden_dim` | `heads.hidden_dim` | 768 | Hidden width inside each prediction MLP. |
| `heads.logit_dim` | `heads.logit_dim` | 2049 | Output vocabulary per codebook head (`audio_vocab_size + 1` for the EOS token). Computed automatically. |
| `heads.dropout` | `heads.dropout` | 0.1 | Dropout inside prediction MLPs. |
| `init_std` | `init_std` | 0.02 | Std of the normal distribution used for linear/embedding weight init. |

## Key constraints

- `decoder.hidden_dim % decoder.no_heads == 0` — enforced at init.
- `input_proj` and `output_proj` are **identity** when `embedding_dim == hidden_dim` and `hidden_dim == intermediate_dim` respectively — saving params when dimensions match.
- `heads.no_heads == num_codebooks` — one logit vector per codebook layer per time step.

## Scaling guide

Keep **head dim at 64** (or 128 for larger models) for optimal GPU utilization.

| Scale | Params (approx) | hidden_dim | no_heads | head_dim | no_layers | ffn_dim |
|-------|----------------|------------|----------|----------|-----------|---------|
| Tiny | ~15M | 256 | 4 | 64 | 4 | 1024 |
| Small | ~50M | 512 | 8 | 64 | 6 | 2048 |
| **Current** | **~100M** | **768** | **12** | **64** | **6** | **3072** |
| Large | ~300M | 1024 | 16 | 64 | 12 | 4096 |
| XL | ~600M | 1536 | 24 | 64 | 16 | 6144 |