---
title: PatchTST — Patch Time Series Transformer
category: techniques
tags: [transformer, time-series, classification, deep-learning]
created: 2026-04-15
updated: 2026-04-15
---

# PatchTST — Patch Time Series Transformer

A transformer architecture for time series that splits the input into fixed-length patches (like ViT does for images), projects each patch to an embedding, and processes the sequence of patch embeddings with standard transformer encoder layers.

## Why It Works for Time Series

- **Patches capture local temporal patterns** (e.g., a 10-sample patch at 50Hz = 200ms motion primitive)
- **Attention over patches** captures long-range dependencies between motion primitives
- **Much more parameter-efficient** than per-timestep attention
- **Naturally handles variable-length sensor data** via different numbers of patches

## Architecture (as used in WEAR 2026)

```
Input: (batch, 50, 3)  # 1 second at 50Hz, 3 axes
  → Patch: (batch, 5, 30)  # 5 patches of length 10, flattened 10×3
  → Linear projection: (batch, 5, d_model)
  → + positional embedding + sensor embedding
  → Transformer encoder (n_layers × MultiHeadAttention + FFN)
  → Mean pool over patches: (batch, d_model)
  → Classification head: (batch, num_classes)
```

**Hyperparameters that worked:**
- `d_model=64`, `n_heads=4`, `n_layers=3`, `patch_len=10`
- `dropout=0.2`, label smoothing 0.1
- AdamW, lr=3e-4, batch 256, 30 epochs
- Runtime: ~37 min/fold on CPU (148 min total for 4-fold)

## When to Use

- Time series classification/regression with moderate sequence lengths
- When you need cross-domain generalization (e.g., cross-subject HAR)
- When handcrafted features underperform (complex temporal patterns)
- When you have enough data to train a transformer (10k+ samples)

## Gotchas

- **Patch length matters**: Too small = noisy, too large = loses temporal resolution. Start with ~10-20% of sequence length.
- **CPU-viable**: Small PatchTST (d_model≤128) trains in reasonable time on CPU — no GPU required for moderate datasets
- **Sensor embedding is critical for HAR**: Without it, the model doesn't know which body part the data comes from. See [[techniques/sensor-embedding]].

## Kaggle Tips

- Pair with [[techniques/lr-swap-augmentation]] and [[techniques/test-time-augmentation]] for HAR
- [[techniques/threshold-optimization]] on OOF predictions boosts macro F1
- Can run on Kaggle CPU when GPU has issues (see [[tools/kaggle-cpu-notebooks]])

## See Also
- [[techniques/sensor-embedding]]
- [[competitions/wear-hasca-2026]]
- [[patterns/cross-subject-generalization]]
