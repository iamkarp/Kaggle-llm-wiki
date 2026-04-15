---
title: Wiki Index
updated: 2026-04-15
---

# Kaggle ML Wiki — Index

## Competitions
- [[competitions/wear-hasca-2026]] — 3rd WEAR Dataset Challenge, HAR with wearable sensors + video, macro F1. #1 on LB.

## Techniques
- [[techniques/patchtst]] — Patch Time Series Transformer for time series classification
- [[techniques/sensor-embedding]] — Learnable embeddings for sensor location in HAR
- [[techniques/lr-swap-augmentation]] — Left-right limb swap augmentation for wearable data
- [[techniques/test-time-augmentation]] — TTA with L-R flip averaging
- [[techniques/threshold-optimization]] — Per-class threshold tuning for macro F1
- [[techniques/gaussian-random-projection]] — Data-independent dimensionality reduction (alternative to PCA)
- [[techniques/gradient-boosted-trees]] — LightGBM, XGBoost, CatBoost for tabular features

## Patterns
- [[patterns/cross-subject-generalization]] — Why some methods fail when train/test subjects differ
- [[patterns/oof-vs-lb-divergence]] — When OOF scores don't predict LB scores
- [[patterns/multimodal-fusion]] — Combining inertial + video features

## Tools
- [[tools/kaggle-cpu-notebooks]] — Running DL on Kaggle CPU (GPU workarounds)
- [[tools/kaggle-cli]] — Kaggle CLI for notebook push, status, submission

## Mistakes
- [[mistakes/pca-on-pretrained-features]] — PCA on VideoMAE features overfits to training subjects
- [[mistakes/gbm-for-cross-subject-har]] — GBM with handcrafted features fails at cross-subject generalization
- [[mistakes/catboost-cpu-timeout]] — CatBoost on CPU exceeds Kaggle 12h limit
