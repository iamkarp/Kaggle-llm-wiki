---
title: "Nearest Neighbor Features — Non-Parametric Local Aggregation"
tags: [nearest-neighbor, knn, features, aggregation, tabular, time-series, non-parametric]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is
Nearest neighbor (NN) features aggregate the target values or engineered statistics of the K most similar training samples as new features for each observation. This provides a non-parametric local estimate — essentially answering "what happened in situations like this one?"

Two distinct uses in Kaggle:
1. **KNN target mean** (Home Credit): local target rate estimate — "what's the default rate among similar applicants?"
2. **NN aggregation features** (Optiver): local statistic aggregation — "what volatility do similar market states produce?"

## Why NN Features Are Powerful

Tree models partition the feature space with axis-aligned splits. KNN features provide smooth local interpolation that trees can't express efficiently. Together, they're highly complementary.

**KNN as non-parametric target encoding**: Unlike target encoding which aggregates by categorical group, KNN works in continuous feature space — no grouping required.

## KNN Target Mean (Classification)

Used as the top feature in Home Credit 1st place.

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

# IMPORTANT: normalize features first (KNN is distance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Fit NN index
nn = NearestNeighbors(n_neighbors=500, metric='euclidean', algorithm='ball_tree')
nn.fit(X_scaled)

# Get neighbors for each training sample (exclude self with [1:])
distances, indices = nn.kneighbors(X_scaled)

# OOF target mean of neighbors — always OOF to prevent leakage!
# (indices[:, 0] is the sample itself — skip it)
knn_target_mean = y_train[indices[:, 1:]].mean(axis=1)
knn_target_std = y_train[indices[:, 1:]].std(axis=1)
```

**Always use OOF**: If you use the same data for fitting the NN index and computing neighbor targets, you leak target information. Either:
- Use K-fold OOF (fit NN on training folds, compute on validation fold)
- Exclude self from neighbors (`kneighbors` returns self first — skip index 0)

For test data: fit the NN index on full training set, apply to test.

## NN Aggregation Features (Regression / Time-Series)

Used as 360 of 600 total features in Optiver 1st place (boost: 0.21 → 0.19 RMSPE).

```python
# For each observation, aggregate base features of K nearest neighbors
nn = NearestNeighbors(n_neighbors=50, metric='euclidean')
nn.fit(X_base_features)
distances, indices = nn.kneighbors(X_base_features)

# Aggregate target values
nn_target_mean = y[indices].mean(axis=1)
nn_target_std = y[indices].std(axis=1)

# Aggregate base features of neighbors (not just target)
for feat in key_feature_cols:
    nn_feat_mean = X_base_features[feat].values[indices].mean(axis=1)
    nn_feat_std = X_base_features[feat].values[indices].std(axis=1)
```

**Optiver-specific**: NN features were computed in multiple spaces:
- Cross-stock at same time_id (stocks in similar market state)
- Cross-time for same stock (this stock in similar historical states)
- Combined space: both dimensions together

## Choosing K

| Setting | Typical K | Rationale |
|---------|-----------|-----------|
| Small dataset (<10K) | 10–50 | Avoid including too much of the data |
| Medium dataset (10K–500K) | 100–500 | Stable local estimate |
| Large dataset (>500K) | 500–2000 | Can afford larger neighborhood |

K is a hyperparameter — tune on CV. Larger K = smoother estimate (less variance, more bias). Start with K=200–500 for tabular competitions.

## Distance Metric Selection

| Metric | When to Use |
|--------|-------------|
| Euclidean | Default for normalized numeric features |
| Cosine | High-dimensional sparse features (NLP) |
| Manhattan | Robust to outliers; slower to compute |
| Mahalanobis | Accounts for feature correlations (expensive) |

Always **normalize features** before Euclidean KNN — un-normalized features with large variance dominate the distance.

## Multiple K Values as Features

Rather than a single K, generate features for multiple values of K:

```python
for k in [10, 50, 100, 500]:
    distances, indices = nn.kneighbors(X_scaled, n_neighbors=k)
    df[f'knn_target_mean_k{k}'] = y[indices].mean(axis=1)
    df[f'knn_target_std_k{k}'] = y[indices].std(axis=1)
```

These capture local structure at different scales — very local (k=10) vs. regional (k=500).

## Cross-Dataset KNN Features
A powerful trick: compute KNN features where the NN index is built on a different table (not the main training table). Example from Home Credit: build KNN index on bureau history, compute nearest-neighbor features for each applicant in the main table. Captures similarity in credit history space.

## Performance Considerations
- KNN with `ball_tree` or `kd_tree` algorithm is much faster than brute force for medium datasets
- For large datasets (>1M rows): use approximate NN (FAISS, annoy, hnswlib)
- Precompute neighbor indices once and reuse for multiple aggregations

```python
import faiss
import numpy as np

X_float32 = X_scaled.astype('float32')
d = X_float32.shape[1]
index = faiss.IndexFlatL2(d)
index.add(X_float32)
distances, indices = index.search(X_float32, k=500)
```

## Sources
- [[../../raw/kaggle/solutions/home-credit-1st-tunguz.md]] — KNN target mean as top feature
- [[../../raw/kaggle/solutions/optiver-volatility-1st-nyanp.md]] — 360 NN aggregation features, 0.21→0.19 boost

## Related
- [[../concepts/target-encoding]] — categorical target encoding (KNN generalizes this to continuous space)
- [[../concepts/feature-engineering-tabular]] — where KNN features fit in the 5-stage process (Stage 5 / aggregations)
- [[../concepts/stacking-deep]] — KNN target mean is a powerful base feature for stacking
- [[../concepts/validation-strategy]] — OOF requirement for KNN target mean to avoid leakage
