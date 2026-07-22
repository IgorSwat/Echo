1. Rebuild sequence batching. Construct each sample as a complete variable-length sequence, place EOS immediately after its final real audio frame, then pad only after the complete assembled sequence.
2. Pass an actual key-padding mask. PAD positions must not be valid attention keys.
3. Use per-sample position IDs. A sample’s positions must not depend on batch peers.
4. Gather audio logits using true per-sample offsets. The current hidden[:, -audio_len:] assumes one shared audio length.
5. Ignore secondary heads at EOS. Use ignore_index for heads 1-15 on the EOS frame.
6. Make PAD genuinely neutral. Use padding_idx for text and prevent codec PAD rows from updating, although proper end-padding and masking are still required.
7. Condition higher codebooks. Use delayed codebook prediction or an intra-frame autoregressive head.
8. Normalize codec embedding sums. Divide by sqrt(16) or apply a learned normalized projection.