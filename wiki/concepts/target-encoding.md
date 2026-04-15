---
title: "Target Encoding — Weighted Blend with OOF"
tags: [target-encoding, categorical, feature-engineering, leakage, oof, tabular]
date: 2026-04-14
source_count: 1
status: active
---

## What It Is
Target encoding replaces a categorical value with a numeric statistic derived from the target — typically the mean target value for that category. It handles high-cardinality categoricals (>50 unique values) that would explode dimensionality if one-hot encoded.

**The problem**: Naive mean encoding leaks the target into the feature. A model trained on mean-encoded training data will see exact target information during training and will overfit catastrophically.

**The solution**: Weighted blend + out-of-fold (OOF) encoding.

## The Weighted Blend Formula

```
encoded = (global_mean * k + class_mean * sqrt(n_class)) / (k + sqrt(n_class))
```

Where:
- `k = 6` — prior strength (hyperparameter; controls shrinkage for rare categories)
- `global_mean` — overall mean of target in training fold
- `class_mean` — mean target for this specific category in training fold
- `n_class` — count of rows with this category in training fold

**Intuition**: Categories with few rows (`n_class` small) shrink heavily toward `global_mean` — we don't trust sparse estimates. High-frequency categories trust `class_mean` more. The `sqrt(n_class)` term means the weight on `class_mean` grows sub-linearly — you need 4x the data to get 2x the trust.

**Why `k=6`**: Empirically robust default. Increase if categories are very sparse; decrease if the dataset is large and categories are dense.

## Implementation (OOF, Correct Way)

```python
import numpy as np
from sklearn.model_selection import KFold

def target_encode_oof(df_train, df_test, cat_col, target_col, k=6, n_splits=5, seed=42):
    global_mean = df_train[target_col].mean()
    encoded_train = np.zeros(len(df_train))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for tr_idx, val_idx in kf.split(df_train):
        fold_train = df_train.iloc[tr_idx]
        fold_val = df_train.iloc[val_idx]

        stats = fold_train.groupby(cat_col)[target_col].agg(['mean', 'count'])

        def encode(val):
            if val in stats.index:
                cm, n = stats.loc[val, 'mean'], stats.loc[val, 'count']
            else:
                cm, n = global_mean, 0
            return (global_mean * k + cm * np.sqrt(n)) / (k + np.sqrt(n))

        encoded_train[val_idx] = fold_val[cat_col].map(encode)

    # Test: use full training stats
    full_stats = df_train.groupby(cat_col)[target_col].agg(['mean', 'count'])
    def encode_test(val):
        if val in full_stats.index:
            cm, n = full_stats.loc[val, 'mean'], full_stats.loc[val, 'count']
        else:
            cm, n = global_mean, 0
        return (global_mean * k + cm * np.sqrt(n)) / (k + np.sqrt(n))

    encoded_test = df_test[cat_col].map(encode_test)

    return encoded_train, encoded_test
```

## When To Use It
- High-cardinality categoricals (>50 unique values): user IDs, zip codes, product IDs, store IDs
- When one-hot encoding would produce too many columns (>200)
- When CatBoost is not being used (CatBoost handles this internally)

## When NOT To Use It
- Low-cardinality (<50 unique): one-hot encode instead — more interpretable, less risk
- When using CatBoost: it handles this natively with ordered target statistics
- When target is not strongly related to the categorical (encoding provides noise, not signal)

## Gotchas
- **OOF is non-negotiable** — without it, you're leaking target information into training features
- **Unseen test categories**: new categories in test not in train get `global_mean` — this is correct and expected
- **Binary targets**: works as-is (mean = class probability). For regression: same formula applies
- **Multi-class**: encode separately for each class (one-vs-rest) or use rank-based encoding

## Anti-Pattern: Naive Mean Encoding
```python
# WRONG — DO NOT DO THIS
mean_map = train.groupby('category')['target'].mean()
train['encoded'] = train['category'].map(mean_map)  # leaks target!
```

## In Jason's Work
Not yet explicitly documented in existing competition pages, but the pattern applies to any tabular competition with high-cardinality features (store IDs, user IDs, product categories). Apply proactively on any `nunique() > 50` column.

## Sources
- [[../../raw/kaggle/kaggle-competition-playbook.md]] — §3.1 weighted blend formula and implementation

## Related
- [[../concepts/feature-engineering-tabular]] — where target encoding fits in the 5-stage pipeline
- [[../entities/lightgbm-catboost]] — CatBoost handles this natively
- [[../concepts/validation-strategy]] — OOF is also the basis for proper stacking
