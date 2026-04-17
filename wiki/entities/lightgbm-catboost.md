---
title: "LightGBM & CatBoost — Gradient Boosting Alternatives"
tags: [lightgbm, catboost, gradient-boosting, tabular, framework, tool]
date: 2026-04-14
source_count: 2
status: active
---

## LightGBM

### What It Is
Microsoft's gradient boosting framework. Grows trees leaf-wise (rather than depth-wise like XGBoost), which is faster and often achieves better performance on large datasets. Slightly more prone to overfitting on small datasets.

### Typical Use in Jason's Work
- **March Mania v5 hybrid**: One component of the hybrid ensemble alongside XGBoost depths 5–8 and LogReg
- **Mega ensemble** (`mega_ensemble.py`): Used as one ensemble member

### Key Differences from XGBoost
- Faster training on large datasets (leaf-wise growth)
- `num_leaves` is the key depth parameter (not `max_depth`)
- Native categorical handling via `categorical_feature` parameter
- Can be more prone to overfitting — use `min_child_samples` to regularize

### Installation
```bash
pip install lightgbm
import lightgbm as lgb
```

---

## CatBoost

### What It Is
Yandex's gradient boosting framework. Primary differentiator: exceptional native handling of categorical features without manual encoding. Ordered boosting prevents target leakage during tree construction.

### Typical Use in Jason's Work
- **Wine competition**: `wine_catboost.py` — CatBoost used for categorical wine attributes
- **Mega ensemble**: One component among many

### Key Advantages
- `cat_features` parameter accepts raw categorical columns — no encoding needed
- Often best-performing on datasets with many categoricals
- Built-in overfitting detection
- Slower to train than LightGBM but often worth it for categoricals

### Installation
```bash
pip install catboost
from catboost import CatBoostClassifier
```

---

## Choosing Between XGBoost / LightGBM / CatBoost
| Condition | Recommend |
|-----------|-----------|
| Small dataset (<10K rows) | XGBoost (most regularization options) |
| Large dataset (>100K rows) | LightGBM (fastest) |
| Many categorical features | CatBoost (native handling) |
| Need reproducible behavior | XGBoost (most stable defaults) |
| Ensemble diversity | Use all three |

In practice: **train all three and ensemble** — they make complementary errors.

## Sources
- [[../../raw/kaggle/v6-ensemble-documentation]] — LightGBM in v5 hybrid

## Related
- [[../entities/xgboost]] — primary framework in Jason's work
- [[../concepts/xgboost-ensembles]] — ensembling patterns
- [[../strategies/march-mania-v6-ensemble]] — LightGBM used in v5 component
