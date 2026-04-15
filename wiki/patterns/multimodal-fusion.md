---
title: Multimodal Fusion
category: patterns
tags: [multimodal, video, inertial, fusion]
created: 2026-04-15
updated: 2026-04-15
---

# Multimodal Fusion

Combining features from different data modalities (e.g., inertial sensor data + video features) to improve predictions.

## Fusion Strategies

### Early Fusion (Feature Concatenation)
Concatenate features from all modalities before feeding to the model.
- Simple, works with any model
- Requires compatible feature dimensions
- Used in WEAR 2026: concatenated 103 inertial features + 50 VideoMAE RP features

### Late Fusion (Prediction Averaging)
Train separate models per modality, average their predictions.
- Each model specializes in its modality
- More robust to missing modalities
- V14 WEAR: blended GBM (inertial+video features) with PatchTST (inertial only)

### Cross-Attention Fusion
Use transformer attention to let modalities attend to each other.
- Most powerful but most complex
- Not yet tried in WEAR — potential improvement direction

## WEAR 2026 Experience

VideoMAE features (768-dim pretrained video features) were strong signal:
- 83.5% of LightGBM feature importance came from 50 VideoMAE RP features
- But the GBM model still failed on LB due to cross-subject issues

**Untried opportunity**: Feed VideoMAE features directly into PatchTST as an additional modality token. This could combine PatchTST's good generalization with the strong video signal.

## See Also
- [[techniques/gaussian-random-projection]]
- [[competitions/wear-hasca-2026]]
