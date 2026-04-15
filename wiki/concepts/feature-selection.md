---
title: "Feature Selection — Time Consistency, Forward Selection, Permutation Importance"
tags: [feature-selection, time-consistency, forward-selection, permutation-importance, leakage, tabular]
date: 2026-04-15
source_count: 3
status: active
---

## What It Is
Feature selection reduces a large feature set to a smaller, higher-quality set. The goal is to remove features that add noise, cause overfitting, introduce distribution shift, or slow training without contributing signal. Different competitions call for different selection strategies.

Three key methods from top Kaggle solutions:

1. **Time consistency selection** (IEEE Fraud, Chris Deotte) — for temporal distribution shift
2. **Forward selection with Ridge** (Home Credit, Tunguz) — for massive feature sets (1000+)
3. **Permutation importance** — model-agnostic and more reliable than native importance

---

## 1. Time Consistency Feature Selection

**Problem**: Your model trains on months 1–6 and is evaluated on months 7–12. Features that correlate with fraud in months 1–6 might not in months 7–12 (temporal drift). Standard CV on shuffled data won't catch this.

**Method** (from IEEE Fraud 1st place):
1. Split training data: train on months 1–5, evaluate on month 6 (skip-month hold-out)
2. Add features one at a time; keep only those that improve month-6 performance
3. Features that boost overall CV but hurt the temporal hold-out are temporally inconsistent — drop them

```python
# Build a temporal hold-out
train_early = train[train['month'] <= 5]
train_late = train[train['month'] == 6]

baseline_auc = eval_auc(train_model(train_early, X_cols_baseline), train_late)

# Test each candidate feature
for new_feature in candidate_features:
    cols = X_cols_baseline + [new_feature]
    model = train_model(train_early, cols)
    auc = eval_auc(model, train_late)
    if auc > baseline_auc + threshold:
        X_cols_baseline.append(new_feature)
        baseline_auc = auc
```

**Use this when**: Train/test are from different time periods. Any competition with a temporal gap (financial, sales, fraud, churn).

**Why it works**: Temporally consistent features reflect structural patterns (customer behavior archetypes, merchant category effects) rather than transient correlations that happen to exist in the training window.

---

## 2. Forward Feature Selection with Ridge

**Problem**: You have 1800 features. Feeding all of them into LightGBM will overfit, slow training, and reduce interpretability. Which 200 actually matter?

**Method** (from Home Credit 1st place):
1. Start with empty feature set
2. Train Ridge regression with cross-validation on all candidate features (score each individually first)
3. Add the feature that maximally improves CV AUC
4. Repeat; stop when adding features no longer helps
5. Use the selected ~200–300 features for all downstream models

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import numpy as np

selected = []
remaining = list(all_features)
best_score = -np.inf

while remaining:
    scores = []
    for feat in remaining:
        candidate = selected + [feat]
        model = Ridge(alpha=1.0)
        score = cross_val_score(model, X[candidate], y, cv=5, scoring='roc_auc').mean()
        scores.append((score, feat))
    
    best_new_score, best_feat = max(scores)
    if best_new_score > best_score + 1e-4:  # meaningful improvement threshold
        selected.append(best_feat)
        remaining.remove(best_feat)
        best_score = best_new_score
    else:
        break  # no more improvement

print(f"Selected {len(selected)} features")
```

**Why Ridge for selection** (not LightGBM):
- Fast to cross-validate (minutes vs. hours for 1000+ features × 5 folds)
- Stable — doesn't overfit badly during selection
- Gives reliable signal on marginal feature contribution
- Selected features then benefit the more powerful LightGBM/XGBoost downstream

**Alternative (faster)**: Two-stage:
1. Use LightGBM importance to eliminate bottom 50% of features
2. Use Ridge forward selection on the survivors

---

## 3. Permutation Importance

More reliable than native LightGBM/XGBoost importance (which is biased toward high-cardinality features).

**Method**: Train a model. For each feature, shuffle its values and measure the drop in CV score. The drop = feature importance.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10,        # shuffle each feature 10 times; average the drop
    random_state=42,
    scoring='roc_auc'
)

importance_df = pd.DataFrame({
    'feature': X_val.columns,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

# Drop features with negative importance (actively hurting)
to_drop = importance_df[importance_df['importance_mean'] < 0]['feature'].tolist()
```

**When native importance is misleading**:
- High-cardinality features (user IDs, product IDs) get artificially high native importance
- Features used early in trees appear more important than late-used features
- Correlated features split importance between them (both appear unimportant)

Permutation importance is unbiased by all of the above.

---

## 4. NaN Pattern Grouping + PCA (IEEE Fraud Variant)

For datasets with many features sharing NaN patterns (same source system):
1. Group features by their exact missingness pattern
2. Apply PCA within each group — reduces redundancy within correlated feature blocks
3. Use PCA components instead of raw features

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

# Group V-columns by NaN pattern
v_cols = [c for c in train.columns if c.startswith('V')]
nan_patterns = {}
for col in v_cols:
    pattern = tuple(train[col].isna().values)
    nan_patterns.setdefault(pattern, []).append(col)

# PCA within each group
pca_features = []
for pattern, cols in nan_patterns.items():
    if len(cols) < 2:
        pca_features.append(train[cols])
        continue
    n_components = min(len(cols), 5)
    pca = PCA(n_components=n_components)
    X_group = train[cols].fillna(train[cols].median())
    components = pca.fit_transform(X_group)
    component_df = pd.DataFrame(components, columns=[f'pca_{cols[0]}_g{i}' for i in range(n_components)])
    pca_features.append(component_df)
```

---

## Decision Tree for Method Selection

```
Is the test set from a different time period than train?
  YES → Use Time Consistency Selection first
  NO  ↓
Do you have > 500 features?
  YES → Forward Selection with Ridge (reduce to 200-300)
  NO  ↓
Do native importances look suspicious (high-cardinality bias)?
  YES → Permutation Importance
  NO  → Native importance is fine as a starting point
```

## Anti-Patterns

| Anti-Pattern | Fix |
|-------------|-----|
| Selecting features by correlation to target | Biased toward leaky features; use stepwise CV |
| Using LightGBM native importance alone | Biased toward high-cardinality; add permutation importance |
| Forward selection with LightGBM (slow) | Use Ridge for selection; use LightGBM for final model |
| Dropping features after seeing LB score | Overfits to LB; use held-out temporal validation |

## Sources
- [[../../raw/kaggle/solutions/ieee-fraud-1st-chris-deotte.md]] — time consistency selection, NaN PCA grouping
- [[../../raw/kaggle/solutions/home-credit-1st-tunguz.md]] — forward selection with Ridge, 1600→240 features
- [[../../raw/kaggle/kaggle-competition-playbook.md]] — §2 stepwise interaction search anti-pattern

## Related
- [[../concepts/validation-strategy]] — temporal hold-out construction
- [[../concepts/feature-engineering-tabular]] — feature selection follows feature engineering
- [[../concepts/xgboost-ensembles]] — native importances (with caveats)
- [[../concepts/stacking-deep]] — feature selection precedes base model training in stacking pipelines
