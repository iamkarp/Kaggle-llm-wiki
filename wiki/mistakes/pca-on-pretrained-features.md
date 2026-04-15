---
title: "Mistake: PCA on Pretrained Features"
category: mistakes
tags: [pca, dimensionality-reduction, overfitting, pretrained]
created: 2026-04-15
updated: 2026-04-15
---

# Mistake: PCA on Pretrained Features

**What happened:** Applied PCA to 768-dim VideoMAE features, reducing to ~100 components. PCA was fit on training subjects. LB score dropped to 0.22.

**Why it's wrong:** PCA learns principal components from the training data. When training and test come from different subjects, the principal directions of variation in training may encode subject identity rather than activity patterns. Test subjects' features project poorly onto these training-specific components.

**The fix:** Use [[techniques/gaussian-random-projection]] instead. The random projection matrix is data-independent, so it works equally well for any subject.

**Detection:** Compare OOF (high, because PCA components work for training subjects) vs LB (low, because they don't work for test subjects). A large gap is the warning sign.

**General rule:** When there's distribution shift between train and test, avoid data-dependent transformations (PCA, learned normalization, data-dependent feature selection). Prefer data-independent alternatives.

## Applies When
- Pretrained embeddings (VideoMAE, BERT, CLIP) need dimensionality reduction
- Train/test distribution shift exists
- Cross-subject, cross-domain, or temporal split evaluations

## See Also
- [[techniques/gaussian-random-projection]]
- [[patterns/cross-subject-generalization]]
