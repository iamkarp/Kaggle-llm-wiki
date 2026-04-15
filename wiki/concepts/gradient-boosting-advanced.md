---
title: "Gradient Boosting — Advanced Configuration Tricks"
tags: [xgboost, lightgbm, catboost, hyperparameter, optuna, gpu, rapids, 2024, 2025]
date: 2026-04-15
source_count: 3
status: active
---

## What It Is

Advanced XGBoost/LightGBM/CatBoost configuration patterns that go beyond the basics. These are the tricks that win competitions once the baseline is solid. Compiled from 2024–2025 winning solutions and NVIDIA Grandmaster Pro Tips.

## XGBoost Advanced Parameters

### `booster=dart`
Dropout regularization applied to trees: at each boosting round, a random subset of trees is dropped and regrown. Better on noisy tabular data.
- Typically +0.1–0.3% improvement over `gbtree` on noisy competitions.
- **Slower at inference** (can't use early stopping). Compensate: fix `n_estimators` by cross-validation first with `gbtree`, then retrain with `dart`.

### `grow_policy=lossguide` + `max_leaves`
Leaf-wise growth (similar to LightGBM). Often faster/better than default `depthwise` on complex data.
```python
params = {"grow_policy": "lossguide", "max_leaves": 127, "max_depth": 0}
```

### `monotone_constraints`
Enforce monotonic relationships where domain knowledge exists. E.g., price → demand must be monotonically negative.
```python
# +1 = monotone increasing, -1 = decreasing, 0 = unconstrained
params = {"monotone_constraints": {"age": 1, "price": -1, "feature_x": 0}}
```
Reduces overfitting in small datasets by incorporating domain knowledge.

### `interaction_constraints`
Prevent specific feature pairs from co-appearing in a split path. Useful when feature groups are semantically independent (e.g., don't allow features from user profile and features from transaction to interact directly).

### GPU Training
```python
params = {"tree_method": "hist", "device": "cuda"}  # XGBoost 2.0+
# or older API:
params = {"tree_method": "gpu_hist"}
```
10–20× speedup on A100/V100.

### OOF-based `n_estimators` for Full Retraining
1. Find optimal `n_estimators` via `early_stopping_rounds=50` on a validation fold.
2. Multiply by 1.05–1.25 to compensate for more training data.
3. Retrain on full data with that fixed `n_estimators`.
This is safer than using early stopping on full data (no validation set to stop on).

## LightGBM Advanced Parameters

### `boosting_type=goss`
Gradient-based One-Side Sampling. Only uses high-gradient instances + random sample of low-gradient ones.
- **Fastest training** for large datasets (100K+ rows).
- Slight accuracy trade-off vs default `gbdt`. Worth it if speed is the bottleneck.

### `boosting_type=dart`
Random tree dropping (same idea as XGBoost DART). Can improve generalization.

### `num_leaves` — The Most Important Parameter
- Rule of thumb: `2^max_depth * 0.6` — gives a leaf-wise equivalent.
- Typical competition range: 63–255 for medium datasets, up to 511+ for large.
- **Don't just use `max_depth`** — LightGBM is leaf-wise; `num_leaves` controls complexity.

### `min_child_samples`
Critical anti-overfit regularizer for noisy tabular data. Range: 20–100. Higher = more regularization.

### `path_smooth`
Smoothing factor for leaf values. Underused but effective for noisy targets. Try values: 0.0–5.0.

### `feature_fraction_bynode`
Resample columns at each node (not just each tree). Adds more randomness than `feature_fraction` alone. Often improves generalization.

### `linear_tree=True`
Fits linear models at leaf nodes instead of constants. Very effective for near-linear relationships or when feature interactions are additive.

## CatBoost Advanced Parameters

### `cat_features` list — Always Use It
Pass raw categoricals without encoding. CatBoost's ordered target statistics outperform manual label encoding.
```python
cat_features = ['category_A', 'category_B', 'status_code']
model = CatBoostClassifier(cat_features=cat_features, ...)
```

### `border_count`
Number of split points for numerical features. Default 254; increase to 1024 for financial/medical data where precision matters.

### Ordered boosting (default)
Avoids target leakage from in-sample target encoding. Best for small datasets. When data is large and compute is a bottleneck, `boosting_type='Plain'` is faster.

### GPU
```python
model = CatBoostClassifier(task_type='GPU', devices='0')
```
Up to 15× speedup on V100.

## Optuna HPO Recipe

Standard competition HPO:

```python
import optuna
import lightgbm as lgb
import numpy as np

def objective(trial):
    params = {
        "objective": "binary",
        "num_leaves": trial.suggest_int("num_leaves", 20, 300),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }
    cv_scores = []
    for tr_idx, val_idx in folds:
        model = lgb.LGBMClassifier(**params)
        model.fit(X[tr_idx], y[tr_idx],
                  eval_set=[(X[val_idx], y[val_idx])], verbose=False)
        cv_scores.append(evaluate(model, X[val_idx], y[val_idx]))
    return np.mean(cv_scores)

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler())
study.optimize(objective, n_trials=100, callbacks=[
    optuna.pruners.MedianPruner()
])
```

**Optuna 4.7.0+ note:** New `GPSampler` is more sample-efficient than `TPESampler` for expensive objectives. Try it when each trial takes >1 minute.

## Key Insight: FE vs HPO Effort

From Home Credit 2024 1st place: **Feature engineering improvements correlate to LB far more reliably than hyperparameter improvements.** CV gains from HPO can be noise; CV gains from new features tend to hold on LB.

Rule of thumb: spend 80% of time on features, 20% on HPO.

## Gotchas

- **CatBoost and LGBM different feature sets:** Some categoricals that boost CatBoost can hurt LightGBM. Maintain separate feature sets per model.
- **DART early stopping:** Can't use `early_stopping_rounds` with `dart` booster (dropped trees create non-monotonic loss curves). Fix n_estimators first with `gbdt`, then switch to `dart`.
- **GPU memory:** CatBoost with `border_count=1024` requires significantly more GPU memory than default. Monitor with `nvidia-smi`.

## In Jason's Work
See [[../entities/xgboost]] for Jason's core XGBoost parameter defaults. The patterns here extend those defaults for competition-specific optimizations.

## Sources
- [[../../raw/kaggle/modern-tabular-dl-techniques.md]] — comprehensive parameter reference
- [[../../raw/kaggle/2024-2025-winning-solutions-tabular.md]] — applied examples from winning solutions
- [Chris Deotte NVIDIA blog](https://developer.nvidia.com/blog/author/cdeotte/) — advanced parameter tutorials

## Related
- [[../entities/xgboost]] — Jason's XGBoost baseline config
- [[../entities/lightgbm-catboost]] — LGB/CAT baseline config
- [[../concepts/ensembling-strategies]] — ensembling multiple GBM configurations
- [[../concepts/feature-engineering-tabular]] — FE is more impactful than HPO
- [[../concepts/tabpfn-tabm]] — when neural nets outperform these GBDTs
