---
title: Running Deep Learning on Kaggle CPU
category: tools
tags: [kaggle, cpu, gpu, pytorch, runtime]
created: 2026-04-15
updated: 2026-04-15
---

# Running Deep Learning on Kaggle CPU

## The GPU Problem (as of April 2026)

Kaggle's P100 GPU has CUDA kernel incompatibility with the current PyTorch version in the Docker image. This causes runtime errors for many DL operations. Until Kaggle updates the image, all deep learning must run on CPU.

## Making CPU Training Viable

### Keep models small
- PatchTST with d_model=64, 3 layers trained 4 folds × 30 epochs in 148 minutes
- d_model=128 or 6 layers would roughly double this — still within 12h limit

### Batch size matters less on CPU
- CPU doesn't benefit from large batch sizes the same way GPU does
- Batch 256 worked fine for PatchTST on CPU

### Use efficient data loading
```python
# Don't use too many workers on Kaggle (only 4 CPU cores)
DataLoader(dataset, batch_size=256, num_workers=2, pin_memory=False)
```

### Time budget
- 12h session limit for CPU notebooks
- Budget: feature extraction (20-40 min) + training (2-5h) + inference (10-30 min) + buffer (1.5h)
- Always add checkpoint saving and time guards

### What fits in 12 hours on CPU

| Model | Config | 4-fold time | Fits? |
|-------|--------|-------------|-------|
| PatchTST (small) | d=64, L=3 | ~2.5h | Yes |
| PatchTST (medium) | d=128, L=4 | ~5h | Yes |
| LightGBM | 2000 trees | ~0.5h | Yes |
| XGBoost | 2000 trees | ~1h | Yes |
| CatBoost | 2000 trees, multiclass | ~12h | No |
| CNN-LSTM | moderate | ~3-4h | Yes |

## Platform Specs

| Resource | CPU Notebook |
|----------|-------------|
| RAM | ~30 GB |
| Cores | 4 |
| Session limit | 12h |
| Weekly quota | Unlimited |
| Disk | 20 GB |

## See Also
- [[mistakes/catboost-cpu-timeout]]
- [[techniques/patchtst]]
