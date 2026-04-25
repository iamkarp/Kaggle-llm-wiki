---
id: competition:wear-hasca-2026
type: competition
title: 3rd WEAR Dataset Challenge @ HASCA 2026
slug: wear-hasca-2026
aliases: []
tags:
- har
- time-series
- wearable
- multimodal
- macro-f1
status: active
created: 2026-04-15
updated: 2026-04-16
category: competitions
---

# 3rd WEAR Dataset Challenge @ HASCA 2026

**Platform:** Kaggle | **Metric:** Macro F1-score | **Result:** #1 (0.60855)

## Problem

Classify 1-second windows of wearable sensor data into 19 activity classes (0=null, 1-18=activities). The null class is ~40-45% of data.

**Key challenges:**
- **Cross-subject generalization**: Train on subjects 0-21, test on subjects 22-25
- **Single random sensor**: Each test window comes from one of 4 body locations (left/right arm/leg), but training has all 4
- **Multimodal**: Inertial accelerometer (50Hz, 3-axis) + pre-extracted VideoMAE features (768-dim, 30FPS)
- **Heavy class imbalance**: Null class dominates; macro F1 penalizes poor minority-class recall

## Data

| Split | Inertial | Video | Labels |
|-------|----------|-------|--------|
| Train | Per-subject CSVs, 50Hz, 4 sensors | Per-subject .npy, 768-dim, 30FPS | In CSV |
| Test | (12234, 50, 3) | (12234, 768, 15) | Predict |

Test metadata provides `id`, `sbj_id`, `sensor_location` for each window.

## What Worked

### Scaled PatchTST (V16) — 0.60855 LB (current best)
- [[techniques/patchtst]] scaled up: d_model=96, n_heads=4, n_layers=3, d_ff=192, dropout=0.15
- [[techniques/sensor-embedding]]: nn.Embedding(4, 16) projected to d_model
- [[techniques/lr-swap-augmentation]]: Negate x-axis, swap left↔right sensor IDs
- [[techniques/test-time-augmentation]]: Average predictions with L-R flipped input
- [[techniques/threshold-optimization]]: Per-class bias on OOF predictions
- Linear warmup (3 epochs) + cosine decay, 40 epochs, patience 10
- 4-fold grouped CV, batch 256, AdamW lr=3e-4, label smoothing 0.1
- Pure inertial — NO video features (they hurt generalization)
- **Runtime: 345 min on CPU** (1 seed, 70-88 min/fold)
- OOF: 0.5232 → 0.5310 adjusted, LB: 0.60855
- 239,891 parameters

### PatchTST with Sensor Embedding (V10) — 0.58575 LB (previous best)
- [[techniques/patchtst]] with d_model=64, n_heads=4, n_layers=3, patch_len=10
- [[techniques/sensor-embedding]]: nn.Embedding(4, 16) concatenated to patch embeddings
- [[techniques/lr-swap-augmentation]]: Negate x-axis, swap left↔right sensor IDs
- [[techniques/test-time-augmentation]]: Average predictions with L-R flipped input
- [[techniques/threshold-optimization]]: Per-class bias on OOF predictions
- 4-fold grouped CV by subject groups, 30 epochs, batch 256, AdamW lr=3e-4
- Label smoothing 0.1, additional augmentations: jitter, scaling, rotation, channel dropout
- **Runtime: 148 min on CPU**
- OOF: 0.5204, LB: 0.58575 (model generalizes better than OOF suggests)

## What Didn't Work

### GBM with Handcrafted Features — 0.18-0.24 LB
See [[mistakes/gbm-for-cross-subject-har]]. Multiple GBM variants all failed:
- V5: LGB+XGB+Cat + VideoMAE PCA → OOF 0.6261, LB 0.22323
- V6: LGB only → OOF 0.5922, LB 0.18589
- V13: LGB+XGB + VideoMAE RP → OOF 0.5901, LB 0.24321
- V14: GBM+PatchTST 50/50 blend → OOF 0.6348, LB 0.50182

GBM OOF scores are inflated because handcrafted features capture training-subject-specific patterns that don't transfer. Even blending GBM with PatchTST (V14) hurt — from 0.586 to 0.502.

### VideoMAE Features in PatchTST — 0.13211 LB
See [[mistakes/videomae-cross-subject]]. Even when feeding VideoMAE features to PatchTST (which generalizes well on its own), the video signal is toxic:
- V15: PatchTST + VideoMAE token → OOF 0.6087, LB 0.13211
- The model attended to subject-specific video patterns, collapsing to 61% null on test
- This is WORSE than GBM alone — video features are the single biggest trap in this competition

### PCA on VideoMAE Features
See [[mistakes/pca-on-pretrained-features]]. PCA trained on training subjects doesn't generalize.

### CatBoost on CPU
See [[mistakes/catboost-cpu-timeout]]. Exceeds 12h Kaggle limit.

### GPU Deep Learning on Kaggle P100
Kaggle P100 has CUDA kernel incompatibility with current PyTorch. All DL must run on CPU. See [[tools/kaggle-cpu-notebooks]].

## Validation Strategy

4-fold grouped cross-validation by subject:
- Fold 0: val subjects [0, 1, 2, 3, 4]
- Fold 1: val subjects [5, 6, 7, 8, 9, 10]
- Fold 2: val subjects [11, 12, 13, 14, 15]
- Fold 3: val subjects [16, 17, 18, 19, 20, 21]

## Key Lessons

1. **Cross-subject HAR needs learned representations**, not handcrafted features
2. **OOF can be misleading** when train/test distributions differ systematically
3. **Sensor embedding is critical** — tells the model which body part the data comes from
4. **L-R swap augmentation + TTA** effectively doubles training data for limb-worn sensors
5. **Threshold optimization** is free macro F1 improvement
6. **Don't blend weak generalizers** — even at optimal OOF weight, GBM contaminated PatchTST on test
7. **VideoMAE features are toxic** — they encode subject/environment-specific visual info; feeding them to any model (GBM or PatchTST) kills cross-subject generalization

## Competition Context

| Rank | Team | Score |
|------|------|-------|
| 1 | Jason Karpeles | 0.60855 |
| 2 | test_target-feature | 0.58422 |
| 3 | AttendAndDiscriminate (baseline) | 0.56091 |
| 4 | TinyHAR (baseline) | 0.54594 |
| 5 | DeepConvLSTM (baseline) | 0.53136 |

## See Also
- [[techniques/patchtst]]
- [[patterns/cross-subject-generalization]]
- [[patterns/oof-vs-lb-divergence]]

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[competitions/wear-hasca-2026|WEAR — Wearable Activity Recognition]]
- _uses_ → [[entities/lightgbm-catboost|LightGBM & CatBoost — Gradient Boosting Alternatives]]
- _uses_ → [[techniques/patchtst|PatchTST — Patch Time Series Transformer]]
- _caused_ → [[mistakes/catboost-cpu-timeout|Mistake: CatBoost CPU Timeout]]
- _caused_ → [[mistakes/gbm-for-cross-subject-har|Mistake: GBM for Cross-Subject HAR]]
- _caused_ → [[mistakes/pca-on-pretrained-features|Mistake: PCA on Pretrained Features]]
- _caused_ → [[mistakes/videomae-cross-subject|Mistake: VideoMAE Features Destroy Cross-Subject Generalization]]
- _evaluated_by_ → `metric:f1` (F1 Score)
- _hosted_by_ → `organization:kaggle` (Kaggle)
- _instance_of_ → `task_type:har` (Human Activity Recognition)
- _related_to_ → [[tools/kaggle-cpu-notebooks|Running Deep Learning on Kaggle CPU]]

**Incoming:**
- [[patterns/cross-subject-generalization|Cross-Subject Generalization]] _applied_in_ → here
- [[patterns/multimodal-fusion|Multimodal Fusion]] _applied_in_ → here
- [[patterns/oof-vs-lb-divergence|OOF vs LB Divergence]] _applied_in_ → here
- [[techniques/lr-swap-augmentation|Left-Right Swap Augmentation]] _succeeded_in_ → here
- [[techniques/lr-swap-augmentation|Left-Right Swap Augmentation]] _applied_in_ → here
- [[techniques/patchtst|PatchTST — Patch Time Series Transformer]] _succeeded_in_ → here
- [[techniques/patchtst|PatchTST — Patch Time Series Transformer]] _applied_in_ → here
- [[techniques/sensor-embedding|Sensor Location Embedding]] _succeeded_in_ → here
- [[techniques/sensor-embedding|Sensor Location Embedding]] _applied_in_ → here
- [[techniques/test-time-augmentation|Test-Time Augmentation (TTA)]] _succeeded_in_ → here
- [[techniques/threshold-optimization|Per-Class Threshold Optimization]] _succeeded_in_ → here
- [[techniques/threshold-optimization|Per-Class Threshold Optimization]] _applied_in_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
