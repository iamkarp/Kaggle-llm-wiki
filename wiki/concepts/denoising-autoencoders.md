---
title: "Denoising Autoencoders — Tabular Representation Learning"
tags: [dae, autoencoder, representation-learning, swap-noise, rankgauss, neural-network, tabular]
date: 2026-04-15
source_count: 3
status: active
---

## What It Is
A denoising autoencoder (DAE) learns a compressed representation of input data by training a neural network to reconstruct clean inputs from corrupted versions. On tabular data, the learned bottleneck representation captures non-linear feature interactions that tree models miss — especially useful when features are noisy, correlated, or have many missing values.

The key question for tabular DAE is **what type of noise to apply**. The Porto Seguro 1st-place solution (Jahrer) established that **swap noise beats Gaussian noise** for tabular data.

## Swap Noise vs. Gaussian Noise

### Swap Noise (Use This)
With probability `p_swap` (~0.1), replace a feature value in sample `i` with the value of that feature from a randomly chosen other sample `j`.

```python
def apply_swap_noise(X, p_swap=0.1):
    X_noisy = X.copy()
    n, m = X.shape
    for j in range(m):
        mask = np.random.rand(n) < p_swap
        random_rows = np.random.randint(0, n, size=mask.sum())
        X_noisy[mask, j] = X[random_rows, j]
    return X_noisy
```

**Why swap noise wins**:
- Always produces realistic values (drawn from actual data distribution)
- Distribution-agnostic — works identically for continuous, ordinal, and categorical features
- Single hyperparameter (`p_swap`); 0.1 is a robust default
- Destroys correlations between features while preserving marginals

### Gaussian Noise (Don't Use for Tabular)
- Can push values outside the natural data range (e.g., negative counts, >1 probabilities)
- Requires per-feature variance calibration
- Poorly suited for categorical or binary features

## RankGauss Normalization

Before feeding tabular features into a DAE or NN, normalize with RankGauss:

1. Rank values in each column (average rank for ties)
2. Map ranks to (0, 1): `uniform = rank / (n + 1)`
3. Apply probit (inverse normal CDF): `gauss = ndtri(uniform)`

```python
from scipy.special import ndtri
import numpy as np

def rank_gauss(col):
    ranks = col.rank(method='average')
    uniform = ranks / (len(col) + 1)
    return ndtri(uniform)

# Apply per-feature
for col in feature_cols:
    df[col] = rank_gauss(df[col])
```

**Why RankGauss**:
- Produces a perfect Gaussian distribution regardless of input distribution
- Robust to outliers (they get mapped to ±3σ range, not ±∞)
- Neural networks train better with Gaussian-distributed inputs
- No distribution assumptions — works on any numeric feature

## DAE Architecture (Standard for Tabular)

```
Input (swap-corrupted, RankGauss-normalized)
→ Dense(512, activation='relu') → BatchNorm → Dropout(0.3)
→ Dense(256, activation='relu') → BatchNorm → Dropout(0.3)
→ Dense(128, activation='relu') → BatchNorm        [← bottleneck / learned representation]
→ Dense(256, activation='relu') → BatchNorm
→ Dense(512, activation='relu') → BatchNorm
→ Dense(n_features)                                 [reconstruction output]
```

**Training objective**: MSE between clean input and reconstruction from corrupted input.

**Using the representation**: After pre-training, extract the 128-dim bottleneck as new features. Feed these into the final supervised model (NN or tree).

## Supervised Autoencoder (Jane Street Variant)

A more powerful variant: train reconstruction loss and supervised loss jointly, end-to-end.

```
Input (130 features, RankGauss-normalized)
→ Encoder: Dense(256, Swish) → Dense(128, Swish) → Dense(64, Swish)
→ Bottleneck (64-dim)
    ├── Decoder:    Dense(128) → Dense(256) → Dense(130)  [reconstruction loss]
    └── Classifier: Dense(64)  → Dense(32)  → Dense(1)   [supervised loss]
```

**Total loss**: `λ_recon * MSE_reconstruction + λ_pred * BCE_prediction`

**Key advantage**: Forces the encoder to learn representations useful for BOTH reconstruction and prediction. **Prevents label leakage** in time-series: train the full model fresh per CV fold, never allowing validation-period data to influence the encoder.

## Three-Stage vs. End-to-End Training

| Approach | Steps | Best For |
|----------|-------|---------|
| **Sequential DAE** | (1) Pre-train DAE unsupervised; (2) Extract bottleneck; (3) Train supervised model | I.I.D. data; when unsupervised pre-training benefits from full dataset |
| **Supervised AE** | Train reconstruction + supervised jointly, per fold | Time-series; when encoder leakage is a concern |
| **DAE as stacker** | Train DAE on OOF meta-features from L1 models | Multi-level stacking (Home Credit approach) |

## DAE as Stacker

In deep stacking architectures (cf. Home Credit), DAE+NN is used at level 2 to combine L1 model OOF predictions non-linearly:

```
L1 OOF predictions (N_models columns)
→ Apply swap noise
→ DAE to learn non-linear combinations
→ Bottleneck → supervised head
→ L2 predictions
```

More expressive than Ridge at level 2, but requires careful regularization to avoid overfitting on small (N_models << N_samples) input.

## When To Use DAE
- Noisy tabular data with unclear feature interactions (insurance, finance, medical)
- High missingness rates where imputation introduces bias
- As a stacker in multi-level ensembles (Level 2+)
- When a pure tree model plateau can't be broken through feature engineering
- Time-series financial data (use supervised AE variant, per-fold)

## When NOT To Use
- Small datasets (<5K rows) — DAE needs enough data to learn meaningful structure
- Well-structured tabular data with clean features — trees may generalize better
- When compute is severely limited — DAEs are expensive to train

## Sources
- [[../../raw/kaggle/solutions/porto-seguro-1st-jahrer.md]] — swap noise, RankGauss, 6-model blend
- [[../../raw/kaggle/solutions/jane-street-1st-yirun-zhang.md]] — supervised AE, per-fold training, Swish
- [[../../raw/kaggle/solutions/home-credit-1st-tunguz.md]] — DAE+NN as level-2 stacker

## Related
- [[../concepts/ensembling-strategies]] — DAE blended with tree models
- [[../concepts/stacking-deep]] — DAE as level-2 stacker in deep architectures
- [[../concepts/validation-strategy]] — per-fold training requirement for time-series
- [[../concepts/feature-engineering-tabular]] — DAE bottleneck as engineered features
