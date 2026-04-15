---
title: "Mistake: GBM for Cross-Subject HAR"
category: mistakes
tags: [gbm, har, generalization, cross-subject]
created: 2026-04-15
updated: 2026-04-15
---

# Mistake: GBM with Handcrafted Features for Cross-Subject HAR

**What happened:** Trained LightGBM + XGBoost on 103 handcrafted time/frequency domain features from accelerometer data. OOF F1 was 0.59. LB F1 was 0.24.

**Why it's wrong:** Statistical features (mean, std, FFT magnitudes, zero-crossing rate) capture **how a specific person moves**, not **what activity is being performed**. Each person has a unique movement signature — stride frequency, acceleration amplitude, motion smoothness. GBM memorizes these training-subject patterns and fails on unseen subjects.

**Evidence from WEAR 2026:**

| Version | Features | OOF F1 | LB F1 |
|---------|----------|--------|-------|
| V6 | 103 inertial | 0.5922 | 0.18589 |
| V13 | 103 inertial + 50 VideoMAE RP | 0.5901 | 0.24321 |
| V14 | GBM + PatchTST 50/50 blend | 0.6348 | 0.50182 |
| V10 | PatchTST (learned features) | 0.5204 | 0.58575 |

Even adding VideoMAE features (V13) or blending with PatchTST (V14) couldn't save GBM. The GBM component actively hurt the ensemble — V14's blend scored 0.08 lower than PatchTST alone.

**The fix:** Use learned representations ([[techniques/patchtst]], CNNs, RNNs) that discover subject-invariant temporal patterns from raw data.

**General rule:** For cross-subject/cross-domain problems, prefer end-to-end learned features over handcrafted statistical features. GBM + handcrafted features is the default Kaggle approach for tabular data, but time series with domain shift is not truly tabular.

## Detection

If your GBM OOF is much higher than your neural net OOF, but the neural net scores higher on LB — your handcrafted features are overfitting to the training distribution.

## See Also
- [[patterns/cross-subject-generalization]]
- [[patterns/oof-vs-lb-divergence]]
- [[techniques/patchtst]]
