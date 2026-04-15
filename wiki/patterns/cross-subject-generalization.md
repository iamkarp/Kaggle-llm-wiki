---
title: Cross-Subject Generalization
category: patterns
tags: [generalization, domain-shift, har, validation]
created: 2026-04-15
updated: 2026-04-15
---

# Cross-Subject Generalization

When training and test data come from different individuals, models must learn subject-invariant representations. This is a form of domain shift that breaks many standard approaches.

## The Problem

Each person performs activities differently — stride length, arm swing amplitude, speed, sensor mounting angle all vary. Features that distinguish "walking" from "running" for subject A may not work for subject B.

## What Works

1. **Learned representations** (transformers, CNNs) that discover invariant patterns across training subjects. In WEAR 2026, [[techniques/patchtst]] scored 0.586 on test subjects it never saw.

2. **Data augmentation** that simulates inter-subject variation: jitter, scaling, rotation, [[techniques/lr-swap-augmentation]].

3. **Grouped cross-validation** by subject — ensures validation mimics the actual train/test split. Never leak same-subject data into both train and val.

## What Fails

1. **Handcrafted statistical features + GBM**: These capture subject-specific patterns. OOF looks great (same subjects in train), LB is terrible (new subjects). See [[mistakes/gbm-for-cross-subject-har]].

2. **Subject-dependent dimensionality reduction** (PCA): Principal components reflect training subject characteristics. See [[mistakes/pca-on-pretrained-features]].

3. **Insufficient augmentation**: Without simulating variation, models memorize training subjects.

## Diagnostic: OOF vs LB Gap

A massive gap between OOF and LB scores is the signature of cross-subject overfitting. In WEAR 2026:

| Approach | OOF F1 | LB F1 | Gap |
|----------|--------|-------|-----|
| PatchTST (learned features) | 0.52 | 0.59 | +0.07 (generalizes well) |
| GBM (handcrafted features) | 0.59 | 0.24 | -0.35 (catastrophic) |

See [[patterns/oof-vs-lb-divergence]] for more on this diagnostic pattern.

## See Also
- [[patterns/oof-vs-lb-divergence]]
- [[mistakes/gbm-for-cross-subject-har]]
- [[techniques/patchtst]]
