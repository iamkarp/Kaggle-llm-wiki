---
title: Gaussian Random Projection
category: techniques
tags: [dimensionality-reduction, feature-engineering, pretrained-features]
created: 2026-04-15
updated: 2026-04-15
---

# Gaussian Random Projection

A data-independent dimensionality reduction method that projects high-dimensional features to a lower-dimensional space using a random Gaussian matrix. Unlike PCA, the projection matrix doesn't depend on training data.

## Why Use It Instead of PCA

PCA learns the projection from training data. When training and test distributions differ (e.g., different subjects), PCA's learned directions may not be relevant for test data. See [[mistakes/pca-on-pretrained-features]].

Random projection preserves pairwise distances (Johnson-Lindenstrauss lemma) regardless of the data distribution. It trades optimality for robustness.

```python
from sklearn.random_projection import GaussianRandomProjection

rp = GaussianRandomProjection(n_components=50, random_state=42)
rp.fit(train_features)  # projection matrix is random, not data-dependent
train_reduced = rp.transform(train_features)
test_reduced = rp.transform(test_features)
```

## When to Use

- Reducing pretrained feature dimensions (e.g., 768-dim VideoMAE → 50-dim)
- When train/test distribution shift makes PCA unreliable
- When you need a quick baseline before trying more sophisticated approaches
- When computational cost of PCA is prohibitive

## Caveats

- Less compact than PCA (need more dimensions to preserve the same variance)
- Still helps GBM models — in WEAR 2026, VideoMAE RP features were 83.5% of LightGBM feature importance
- The underlying GBM still failed at cross-subject generalization (see [[mistakes/gbm-for-cross-subject-har]]), but the features themselves were informative

## See Also
- [[mistakes/pca-on-pretrained-features]]
- [[techniques/gradient-boosted-trees]]
