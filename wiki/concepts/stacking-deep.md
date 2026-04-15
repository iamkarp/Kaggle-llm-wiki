---
title: "Deep Stacking — Multi-Level Stacking Architectures"
tags: [stacking, multi-level, ensemble, oof, meta-learner, blending, tabular, kaggle, deepinsight, tabnet, auxiliary-targets, geometric-mean, otto, tsne]
date: 2026-04-15
source_count: 6
status: active
---

## What It Is
Deep stacking extends the standard 2-level stacking architecture (base models → meta-learner) to 3+ levels. Each level trains on the OOF predictions of the previous level. The Home Credit 1st-place solution (Tunguz team) used a 3-level architecture with 90+ base models. At each level, diversity and OOF correctness are paramount.

For the standard 2-level stacking pattern, see [[../concepts/ensembling-strategies]]. This page focuses on when and how to go deeper.

## Why Go Deeper?

Each stacking level can learn combinations the previous level couldn't express:
- **Level 1 → Level 2**: Meta-learner assigns weights to base models; learns which models to trust for which type of input
- **Level 2 → Level 3**: Learns non-linear combinations of level-2 stackers; each stacker may have different systematic biases that level 3 can correct

**Diminishing returns**: Going from L1→L2 is often a large gain. L2→L3 is smaller. L3→L4 is rarely worth the complexity unless you have 50+ diverse L2 models.

## Home Credit Architecture (3-Level)

```
RAW FEATURES + 1800 ENGINEERED FEATURES
    ↓
┌─────────────────────────────────────────────────┐
│ LEVEL 1: 90+ Base Models (OOF predictions)      │
│                                                  │
│ LightGBM × 30 (vary: params, features, seeds)   │
│ XGBoost × 20                                     │
│ CatBoost × 10                                    │
│ LogisticRegression × 5                           │
│ MLP (NN) × 5                                     │
│ DAE+NN × 5                                       │
│ Ridge × 10                                       │
│ Other (RF, ExtraTrees, KNN) × 5+                 │
└──────────────────────┬──────────────────────────┘
                       │ 90+ OOF columns
                       ↓
┌─────────────────────────────────────────────────┐
│ LEVEL 2: L1 Stackers                            │
│                                                  │
│ Ridge(L1_OOF)                                    │
│ LightGBM(L1_OOF)                                │
│ DAE+NN(L1_OOF)   ← non-linear combinations     │
└──────────────────────┬──────────────────────────┘
                       │ 3 OOF columns
                       ↓
┌─────────────────────────────────────────────────┐
│ LEVEL 3 / FINAL BLEND                           │
│ Weighted average of L2 outputs                  │
└─────────────────────────────────────────────────┘
```

## Building L1 Diversity (90+ Models)

90 base models sounds excessive but each one is a small variation — systematically covering the space:

| Source of Diversity | Example Variations |
|--------------------|-------------------|
| Algorithm | LGB, XGB, CatBoost, Ridge, MLP |
| Hyperparameters | `max_depth` 4/6/8, `learning_rate` 0.01/0.05 |
| Feature subsets | Full features, top-500, top-200, domain-grouped subsets |
| Random seeds | 5 seeds per algorithm/config combination |
| Target transforms | Raw target, log-transformed |
| Preprocessing | With/without RankGauss, different imputation |
| Data subsets | Different time windows, different feature groups |

**Key principle**: Diversity comes from orthogonal differences. Adding 10 LightGBM models that differ only by seed gives diminishing diversity gains. Adding LightGBM + Ridge + MLP gives high diversity even with just 3 models.

## L2 Stacker Selection

At level 2, the input is the N_models OOF columns from level 1. Use:

| L2 Model | When | Notes |
|----------|------|-------|
| **Ridge** | Default | Fast, stable, interpretable weights |
| **Logistic Regression** | Binary classification | Same advantages as Ridge |
| **LightGBM** | When L1 models have non-linear relationships | Can learn interaction between stackers |
| **DAE + NN** | When L1 models are numerous (90+) | Learns non-linear combinations without overfitting |
| **ElasticNet** | When many L1 models are correlated | L1 regularization drops redundant models |

**Adding original features to L2**: Sometimes helps when L1 models don't fully exploit all signals. Start without, add if L2 CV improves.

## OOF Discipline at Every Level

This cannot be relaxed. At every level:
- L1 models: train on K-1 folds, predict on the K-th fold
- L2 models: train on L1 OOF predictions, generate their own OOF predictions
- Final blend: no further fitting — just weighted average

**Fold consistency**: Use the same fold assignments throughout all levels. If L1 uses 5 folds with seed 42, L2 must use the same 5 folds.

```python
# Pre-compute fold assignments once
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignments = np.zeros(len(X_train), dtype=int)
for fold_i, (_, val_idx) in enumerate(kf.split(X_train, y_train)):
    fold_assignments[val_idx] = fold_i

# Reuse these assignments for every level
```

## Practical Deep Stacking Checklist

1. **Start with 2-level stacking** — verify it beats simple blending before adding levels
2. **Build L1 diversity first** — more diverse L1 is worth more than more stacking levels
3. **Check L1 OOF correlations** — prune correlated models before stacking (they add noise, not signal)
4. **Use Ridge at L2 first** — interpretable baseline; coefficients reveal model quality
5. **Add LightGBM or DAE+NN at L2** only if Ridge plateaus
6. **L3 is rarely necessary** — if you need it, you probably haven't maximized L1 diversity
7. **Monitor OOF AUC at each level** — if L2 OOF < L1 best OOF, something is wrong (leakage, overfitting)

## Blending vs. Stacking vs. Deep Stacking

| Method | Params | Overfitting Risk | Best For |
|--------|--------|-----------------|---------|
| Simple average | 0 | None | Quick baseline |
| Weighted blend | N | Low | Well-calibrated diverse models |
| 2-level stacking | ~ | Low-medium | Standard Kaggle competition |
| Deep stacking (3+) | ~~~ | Medium | 90+ base models; large datasets |

## When NOT to Deep Stack
- Dataset has < 5K rows (not enough data for OOF to be reliable at each level)
- You have < 10 diverse base models (not enough input for meaningful meta-learning)
- Competition has strict runtime limits (deep stacking is slow at inference)
- You're early in the competition (premature optimization — build better base models first)

---

## The Otto Blueprint: 33-Model L1 + Geometric Mean L3

Otto Group 1st place (Giba & Semenov, 499 votes, 2015) is the canonical 3-level stacking reference. Defined principles still used 10 years later.

### Level 1: 33 Diverse Models — The Complete Family

```
Random Forest       — multiple n_estimators, max_features
Logistic Regression — L1/L2 regularization, multiple C values
Extra Trees         — multiple depths
KNN                 — K = 1, 3, 5, 10, 25, 50, 100, 250, 500, 1000  ← multiple K!
libFM               — factorization machines
H2O Deep Learning   — distributed NN
Lasagne NN bags     — multiple seeds and architectures
XGBoost             — multiple depth/lr combinations
Sofia               — stochastic gradient SVM
T-SNE features      — 3D t-SNE dimensions fed to classifiers
KMeans clusters     — 50-cluster distance features
```

**The non-obvious choices**:
- **KNN at 10 different K values**: cheap to train, each K provides smooth local interpolation at a different scale. K=1 and K=1000 make very different errors — both worth keeping.
- **T-SNE features**: run t-SNE with n_components=3 on the joint train+test feature matrix; use the 3 dimensions as new numeric features. Captures global cluster structure. (Must be run transductively — see [[../concepts/deep-learning-tabular]] for transductive caveat.)
- **KMeans distance features**: 50-cluster KMeans, use `.transform()` to get distances to all cluster centers. Non-parametric density estimate of the feature space.

### The Diversity Principle (Codified by Otto)

> "Don't discard low-performance L1 models."

A model with weaker solo performance still contributes ensemble diversity if its errors are uncorrelated with other L1 models. The L2 meta-learner exploits this — it assigns small weights to weak-but-diverse models rather than zero. This principle was later independently confirmed in Home Credit (90+ models including simple Ridge variants).

### Level 2: Massively Bagged Stackers

```
XGBoost L2:   250 bags (different seeds)
NN L2:        600 bags (Lasagne, multiple architectures and seeds)
AdaBoost L2:  250 bags (ExtraTrees as base estimator)
```

With 297 L2 input features (33 models × 9 classes), high variance in NNs is expected. 600 bags stabilizes the NN L2 output. This is the multi-level equivalent of seed averaging for individual models.

### Level 3: Geometric Mean Blend

```
Final = 0.85 × (XGB_L2^0.65 × NN_L2^0.35) + 0.15 × ET_L2
```

The inner term is a **geometric mean** of XGB and NN L2 outputs:

```python
import numpy as np

def geometric_blend(preds_list, weights):
    """
    Geometric mean blend for probability outputs.
    preds_list: list of (n_samples, n_classes) arrays
    weights: exponent weights (should sum to 1)
    """
    log_blend = sum(w * np.log(p + 1e-10) for w, p in zip(weights, preds_list))
    blend = np.exp(log_blend)
    # Renormalize so class probabilities sum to 1 per sample
    return blend / blend.sum(axis=1, keepdims=True)

# Inner: geometric mean of XGB and NN
inner = geometric_blend([xgb_l2, nn_l2], weights=[0.65, 0.35])

# Outer: linear combination with ET for diversification
final = 0.85 * inner + 0.15 * et_l2
```

**Why geometric mean for multi-class probabilities**:
- For log-loss, geometric mean of well-calibrated models tends to produce sharper (less hedged) predictions than arithmetic mean
- Geometric mean amplifies agreement between models (both must be confident for the blend to be confident) — a natural uncertainty filter
- After geometric blend, renormalize to restore sum-to-1 constraint
- Works best when models are similarly calibrated; if one model is poorly calibrated, arithmetic blend is safer

## MoA 3-Stage Architecture: Auxiliary Targets as Meta-Features

A different flavor of 3-level stacking from MoA 1st place (Mark Peng): instead of stacking identical-objective models, use **auxiliary (non-scored) targets in Stage 1 to generate meta-features for Stage 2**.

```
Stage 1: 2-Heads ResNet
  Input:   875 raw features (gene expression + cell viability)
  Outputs: OOF predictions for
    ├── 206 scored targets
    └── 402 non-scored targets (not evaluated, but available)
  → Total: 608 OOF columns (meta-features for Stage 2)

Stage 2: Smaller NN on meta-features
  Input:   608 meta-features from Stage 1
  Output:  206 scored target predictions

Stage 3: Final 7-Model Weighted Blend
  Weights:
    ├── Stage 1 ResNet:       0.20
    ├── Stage 2 meta-NN:      0.20
    ├── EfficientNet-B3 (DeepInsight): 0.15
    ├── EfficientNet-B4 (DeepInsight): 0.15
    ├── TabNet:               0.10
    ├── LightGBM:             0.10
    └── MLP (vanilla):        0.10
```

**Key principle**: Predict everything available in Stage 1 — all auxiliary targets — and treat all Stage 1 outputs as meta-features. The 402 non-scored targets are correlated with the 206 scored targets (same drug mechanism domain). Stage 2 learns to translate the full mechanism-space representation into scored-target predictions.

**Generalizes to**: Any competition with auxiliary labels, related tasks, or weak supervision signals. If there's any supervisory signal beyond the primary target, use it in Stage 1.

## DeepInsight as a Stacking Component

DeepInsight (tabular → t-SNE image → CNN) is used at the final blend level, not in the stacking hierarchy, because:
- Image creation depends on the full training feature matrix (t-SNE fit)
- EfficientNet produces its own OOF predictions independently
- It contributes maximally to ensemble diversity (completely different compute graph)

```
DeepInsight pipeline (parallel to stacking):
  Raw features
  → t-SNE layout (fit once on train features)
  → Convert each sample to 64×64 image
  → EfficientNet-B3/B4 (pretrained ImageNet → fine-tune)
  → OOF predictions
  → Blended at final stage at 0.15+0.15 weight
```

The t-SNE layout is computed on the feature vectors (treating features as points in sample space), not on the samples. This means correlated features cluster together spatially — CNN filters learn correlation patterns in feature space.

## Sources
- [[../../raw/kaggle/solutions/home-credit-1st-tunguz.md]] — 3-level stacking with 90+ base models, DAE+NN stacker
- [[../../raw/kaggle/solutions/porto-seguro-1st-jahrer.md]] — simple averaging beats nonlinear stacking (contrasting insight)
- [[../../raw/kaggle/solutions/moa-1st-mark-peng.md]] — auxiliary targets as Stage 1 meta-features, DeepInsight, 7-model blend
- [[../../raw/kaggle/solutions/amex-default-14th-chris-deotte.md]] — nested 10-in-10 K-fold for leak-free stacking CV
- [[../../raw/kaggle/solutions/otto-group-1st-giba-semenov.md]] — 33-model L1 blueprint, geometric mean L3, diversity principle
- [[../../raw/kaggle/solutions/playground-s5e4-1st-chris-deotte.md]] — RAPIDS cuML minimal stacking playbook

## Related
- [[../concepts/ensembling-strategies]] — 2-level stacking foundation; fourth-root blending
- [[../concepts/denoising-autoencoders]] — DAE+NN as L2 stacker
- [[../concepts/validation-strategy]] — OOF discipline required at every level
- [[../concepts/nearest-neighbor-features]] — KNN target mean as a powerful L1 base feature
- [[../concepts/feature-selection]] — reduce features before building 90+ L1 models
- [[../concepts/deep-learning-tabular]] — DeepInsight, TabNet, VSN as L1 model components
- [[../concepts/categorical-embeddings]] — topic model embeddings as L1 input features (TalkingData)
