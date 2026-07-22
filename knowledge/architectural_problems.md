# Architectural Analysis — Echo TTS Model

## Overview

Checkpoint: `models/checkpoint_epoch020.pt` — epoch 20, global step 56,220
- Train loss: 3.49
- Validation loss: 3.89
- Parameters: 97.9M

The previous eight fixes (right-padded sequences, per-sample positions, key-padding masks, per-sample offset gathering, ignore-index for secondary heads, neutral PAD embeddings, normalized codec sums, intra-frame conditioning) are confirmed working:

- **EOS calibration**: mean EOS probability at true position is 0.976, top-1 is EOS 99.1%, drops to 0.0005 one frame after.
- **PAD embeddings**: text PAD norm 0.0, codec PAD norm 0.0, both receive zero gradient.
- **Modality scales**: balanced — text norm 1.32, audio norm 1.28, ratio 0.97.
- **Batch-peer independence**: confirmed by tests and reference-induced loss variance of 0.05.

---

## Bottleneck 1 (Critical): Intra-Frame Conditioning Cannot Represent Near-Deterministic Codebook Relationships

### Evidence

**Conditional entropy floor** computed from the dataset: for each codebook `c`, `H(c_t | c_t^{<c})` — the entropy of codebook `c` given all lower codebooks at the same time frame.

| Head | Model CE | Conditional entropy floor | Gap | Reducible fraction |
|-----:|---------:|--------------------------:|----:|----:|
| 0 | 0.939 | 6.677 | — | model beats floor via transformer context |
| 1 | 1.867 | 2.918 | — | model beats floor via transformer + conditioning |
| 2 | 2.735 | 1.073 | 1.66 | 61% |
| 3 | 2.797 | 0.318 | 2.48 | 89% |
| 4 | 3.172 | 0.204 | 2.97 | 94% |
| 5 | 3.424 | 0.104 | 3.32 | 97% |
| 6 | 3.612 | 0.055 | 3.56 | 98% |
| 7 | 3.827 | 0.032 | 3.80 | 99% |
| 8 | 4.056 | 0.026 | 4.03 | 99% |
| 9 | 4.289 | 0.026 | 4.26 | 99% |
| 10 | 4.363 | 0.019 | 4.34 | 100% |
| 11 | 4.580 | 0.015 | 4.57 | 100% |
| 12 | 4.726 | 0.017 | 4.71 | 100% |
| 13 | 4.854 | 0.013 | 4.84 | 100% |
| 14 | 4.982 | 0.008 | 4.97 | 100% |
| 15 | 5.144 | 0.007 | 5.14 | 100% |

**Mean conditional entropy**: 0.72 nats
**Mean model CE**: 3.71 nats
**Mean gap**: 3.00 nats

Heads 3–15 are nearly deterministic given heads 0–2 (conditional entropy < 0.32 nats). 88–99.9% of the loss on these heads is reducible.

### Conditioning Ablation

| Head | CE without conditioning | CE with conditioning | Improvement |
|-----:|------------------------:|---------------------:|------------:|
| 0 | 1.02 | 1.02 | 0.00 (expected) |
| 1 | 3.03 | 2.00 | 1.03 |
| 2 | 6.85 | 2.87 | 3.98 |
| 3 | 6.84 | 2.90 | 3.94 |
| 5 | 6.94 | 3.43 | 3.52 |
| 10 | 7.23 | 4.44 | 2.79 |
| 15 | 7.36 | 5.18 | 2.17 |

Without conditioning, heads 2–15 are near random (`ln(2050) = 7.63`). With conditioning, they drop to 2.9–5.2. The conditioning is working and provides enormous improvement, but still leaves a gap of 2–5 nats to the theoretical floor.

### Why the Gap Exists

The current conditioning representation is:

```python
conditioning_c = codec_condition_proj(sum(embed(c_i) for i < c) / sqrt(c))
```

This is a **sum of continuous embeddings projected through a single linear layer**. Heads 3–15 are nearly **deterministic functions** of heads 0–2. A deterministic mapping requires knowing the **exact token IDs** of the lower codebooks, not a continuous approximation of their embeddings.

The embedding sum loses token identity. For example, tokens 5 and 800 in codebook 0 may have similar embeddings, but they imply very different values for codebook 3. The sum cannot distinguish them.

### Remedies

1. **Per-codebook learnable embedding lookup + concatenation**: Instead of summing, concatenate the lower codebook token embeddings and feed them through a small MLP. This preserves per-codebook information.

2. **Token-ID lookup table**: Map each `(codebook, token_id)` pair to a conditioning vector via a dedicated learnable table, then concatenate. This is the most direct representation of the discrete mapping.

3. **Small autoregressive transformer over codebooks within each frame**: A 2–3 layer transformer that processes the 16 codebook positions, with the hidden state as the initial query. This can learn the near-deterministic mapping directly.

4. **Gating mechanism**: Instead of `hidden + conditioning`, use `hidden * gate(conditioning)` (FiLM-style) so the conditioning can selectively modulate the hidden state.

Option 3 is the most principled for hierarchical residual codecs and would likely close most of the gap.

---

## Bottleneck 2 (High): Extreme Exposure Bias

### Evidence

| Metric | Value |
|--------|------:|
| Teacher-forced loss | 3.15 |
| 1-step-shifted input loss | 10.10 |
| Exposure gap | 6.95 nats |

The loss explodes from 3.15 to 10.10 when the input is shifted by just one frame. Free-running generation accuracy confirms this:

| Sample | Head 0 acc | Head 1 acc | Heads 2+ acc |
|-------:|-----------:|-----------:|--------------:|
| 0 | 8% | 8% | 3–5% |
| 100 | 33% | 15% | 4–15% |
| 500 | 4% | 2% | 2–6% |

Even head 0, which has 70% teacher-forced accuracy, drops to 4–33% in free-running mode.

### Why This Happens

The transformer processes the full sequence including teacher-forced audio frames. Each audio prediction position `t` can attend to all previous positions `0..t-1`, which contain the exact ground-truth codec frames. The model learns to copy from the immediately preceding frame rather than learning a robust text-to-audio mapping.

This is inherent to teacher-forced autoregressive training but is extreme here because:
- The audio codec is highly autocorrelated (consecutive frames are very similar).
- The model has a strong shortcut: attend to the previous frame and predict a small delta.
- The transformer has no mechanism to prevent this shortcut.

### Remedies

1. **Scheduled sampling**: With probability `p`, replace the teacher-forced input frame with the model's own prediction from the previous step. Start with `p=0` and increase to `p=0.3–0.5` during training.
2. **Input noise injection**: Add small noise to the teacher-forced codec embeddings during training, forcing the model to be robust to perturbations.
3. **Detach gradients from audio positions**: Prevent gradient flow through the teacher-forced audio input positions, forcing the model to learn from text context rather than copying audio.
4. **Text-only loss component**: Add an auxiliary loss that predicts audio frames using only text and reference (no previous audio), forcing the model to learn a text-to-audio pathway.

---

## Bottleneck 3 (Medium): Reference Conditioning Is Barely Used

### Evidence

| Metric | Value |
|--------|------:|
| Reference loss range across 10 different references | 0.050 |
| Reference loss std | 0.015 |
| Reference audio gradient norm | 0.001 |
| Target text gradient norm | 0.007 |
| Reference-to-target gradient ratio | 1.07 |

Changing the reference audio barely changes the loss. The model is not learning to use the reference for speaker/style conditioning.

### Why This Happens

This is a **single-speaker dataset** (all files are "norbi" clones). With one speaker, the reference provides no useful speaker information, so the model correctly learns to ignore it. This is not a bug — it is the optimal strategy for single-speaker data.

If the intent is voice cloning (using the reference to control speaker identity), this architecture will not learn it from single-speaker data.

### Remedies

If voice cloning is not a goal, no action is needed. If it is:
1. Train on a multi-speaker dataset.
2. Add a speaker-contrastive auxiliary loss that forces the model to use reference information.
3. Replace the random reference sampling with same-speaker references paired with the target.

---

## Bottleneck 4 (Medium): Long Sequences Degrade

### Evidence

| Audio length range | Sample count | Loss |
|-------------------:|:------------:|:----:|
| 10–25 | 2,956 | 2.67 |
| 25–40 | 10,731 | 3.08 |
| 40–60 | 8,064 | 3.28 |
| 60–90 | 2,644 | 3.38 |
| 90–160 | 595 | 3.58 |

The loss increases by **0.9 nats** from short to long sequences.

### Why This Happens

- Position embedding norms: rows 0–400 = 0.451, rows 400–800 = 0.486, rows 800+ = 0.338. Later positions are weaker.
- The dataset has few long sequences (only 595 samples above 90 frames), so long-range positions are undertrained.
- The transformer's effective context may degrade over long sequences.

### Remedies

1. **Use RoPE** instead of learned absolute positions: rotary position embeddings generalize better to unseen lengths and don't require training all position rows.
2. **Bucket by length**: train with length-sorted batches so long sequences appear more frequently.
3. **Oversample long sequences**: increase the weight of long sequences in the loss or use oversampling.

---

## Bottleneck 5 (Low-Medium): Moderate Train/Validation Gap

### Evidence

- Train loss: 3.49
- Validation loss: 3.89
- Gap: 0.40

The gap is moderate and not yet severe, but it will likely grow if training continues.

### Remedies

1. Increase decoder dropout from 0.2 to 0.3.
2. Increase weight decay from 0.02 to 0.05.
3. Use early stopping based on validation loss.
4. The most effective remedy is fixing Bottleneck 1, which will reduce the model's need to memorize training patterns.

---

## Summary: What Is Blocking Further Learning

Ranked by impact:

1. **The intra-frame conditioning representation cannot capture near-deterministic codebook relationships** (3.0 nats reducible loss). The embedding-sum + linear projection is fundamentally insufficient for a discrete near-deterministic mapping. A within-frame autoregressive transformer or a per-codebook lookup table would close most of this gap.

2. **Extreme exposure bias** (7 nat gap). The model copies from teacher-forced inputs and collapses in free-running generation. Scheduled sampling or input noise injection is needed.

3. **Long-sequence degradation** (0.9 nat gap). RoPE or length-aware training would help.

4. **Reference conditioning is unused** (acceptable for single-speaker, but blocks voice cloning).

5. **Moderate overfitting** (0.40 gap) — will improve once Bottleneck 1 is addressed.

### Impact Estimate

If the within-frame codebook prediction could reach the conditional entropy floor, the unweighted mean CE would drop from **3.71 to approximately 0.72**, and the weighted loss (decay=0.9) would drop to approximately **0.4**. This is a theoretical limit, not a prediction, but it shows how much room remains.

### Per-Codebook CE vs. Unigram Entropy

| Head | Model CE | Unigram entropy | Reduction |
|-----:|---------:|----------------:|:----------|
| 0 | 0.94 | 6.69 | 86% |
| 1 | 1.87 | 5.78 | 68% |
| 2 | 2.74 | 6.15 | 55% |
| 3 | 2.80 | 6.21 | 55% |
| 4 | 3.17 | 6.49 | 51% |
| 5 | 3.42 | 6.58 | 48% |
| 6 | 3.61 | 6.64 | 46% |
| 7 | 3.83 | 6.65 | 42% |
| 8 | 4.06 | 6.77 | 40% |
| 9 | 4.29 | 6.80 | 37% |
| 10 | 4.36 | 6.79 | 36% |
| 11 | 4.58 | 6.85 | 33% |
| 12 | 4.73 | 6.85 | 31% |
| 13 | 4.85 | 6.89 | 30% |
| 14 | 4.98 | 6.90 | 28% |
| 15 | 5.14 | 6.91 | 26% |