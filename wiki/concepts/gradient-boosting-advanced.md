---
id: concept:gradient-boosting-advanced
type: concept
title: Gradient Boosting — Advanced Configuration Tricks
slug: gradient-boosting-advanced
aliases: []
tags:
- xgboost
- lightgbm
- catboost
- hyperparameter
- optuna
- gpu
- rapids
- 2024
- 2025
status: active
date: 2026-04-15
source_count: 3
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

## Loss Function Parameter Scaling

### CatBoost Huber Delta — Must Match Target Scale

The Huber loss transitions from MSE (quadratic) to MAE (linear) at `|residual| > delta`. If delta is much smaller than typical residuals, **the loss gradient becomes constant** — the model can't improve and stops at 0 iterations.

```python
# BAD: delta=1.0 when target std=138 → model stops at 0 iterations
CatBoostRegressor(loss_function='Huber:delta=1.0', ...)  # BROKEN

# GOOD: delta proportional to target scale
target_std = y_train.std()
delta = target_std * 0.5  # e.g., ~70 for target std=138
CatBoostRegressor(loss_function=f'Huber:delta={delta}', ...)
```

**Rule of thumb:** `delta = 0.3 × target_std` to `1.0 × target_std`. Smaller delta = more robustness to outliers but slower convergence. For LightGBM, the equivalent is the `alpha` parameter in `huber` objective.

### LightGBM Fair Loss Alpha

Similar scaling applies. `fair_c` (alpha in the fair loss) should be proportional to target scale:

```python
# For target with std=138
params = {'objective': 'fair', 'fair_c': 50.0}  # not the default 1.0
```

## Learning Rate + Early Stopping Interaction

Low learning rate requires more patience. If `learning_rate` is lowered without increasing `early_stopping_rounds`, models stop before learning:

| Learning Rate | Patience Needed | Typical Best Iteration |
|---|---|---|
| 0.1 | 50 | 100-300 |
| 0.05 | 50-100 | 200-500 |
| 0.02 | 100-200 | 500-1500 |
| 0.01 | 200+ | 1000-3000 |

**Practical rule:** `patience ≈ 50 / learning_rate`. A model with LR=0.02 and patience=50 will often stop too early during temporary validation loss plateaus.

### Full-Data Retrain: Enforce Minimum Iterations

When retraining on full data (no validation set), early stopping is unavailable. Use `n_estimators = 1.25 × avg_best_cv_iteration`, but **enforce a minimum**:

```python
avg_best = np.mean([m.best_iteration_ for m in cv_models])
n_estimators = max(int(avg_best * 1.25), 200)  # minimum 200
```

Without the floor, models that stopped at 0-15 iterations in CV (due to regime-shift validation noise) produce near-constant predictions.

## Gotchas

- **CatBoost and LGBM different feature sets:** Some categoricals that boost CatBoost can hurt LightGBM. Maintain separate feature sets per model.
- **DART early stopping:** Can't use `early_stopping_rounds` with `dart` booster (dropped trees create non-monotonic loss curves). Fix n_estimators first with `gbdt`, then switch to `dart`.
- **GPU memory:** CatBoost with `border_count=1024` requires significantly more GPU memory than default. Monitor with `nvidia-smi`.
- **Huber/Fair/Quantile delta/alpha scaling:** These loss parameters must scale with target magnitude. Default values (often 1.0) are catastrophic for targets with std >> 10. Always set relative to `y_train.std()`.
- **Ridge solver for stacking:** When using Ridge as a meta-learner on scaled features, features with zero IQR produce `inf` after RobustScaler. Fix: `np.clip(np.nan_to_num(X, nan=0, posinf=0, neginf=0), -10, 10)` + `solver='svd'` (not default Cholesky, which overflows).

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

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[entities/xgboost|XGBoost — eXtreme Gradient Boosting]]
- _applied_in_ → [[competitions/playground-s5-s6|Kaggle Playground Series S5 & S6 Winning Patterns (2025-2026)]]
- _applied_in_ → [[competitions/stock-return-prediction|Predict 1-Year US Stock Returns from Fundamentals]]
- _compared_to_ → [[concepts/tabpfn-tabm|TabPFN & TabM — New SOTA Tabular Foundation Models]]
- _cites_ → `source:2024-2025-winning-solutions-tabular` (2024–2025 Winning Solutions: Tabular/Financial/Insurance Competitions)
- _cites_ → `source:modern-tabular-dl-techniques` (Modern Deep Learning & Advanced Techniques for Tabular Kaggle (2023–2025))
- _works_with_ → [[concepts/ensembling-strategies|Ensembling Strategies — Fourth-Root Blend, Stacking, Diversity]]
- _works_with_ → [[concepts/feature-engineering-tabular|Feature Engineering — Tabular Data Patterns]]
- _works_with_ → [[entities/lightgbm-catboost|LightGBM & CatBoost — Gradient Boosting Alternatives]]

**Incoming:**
- [[tools/autoresearch|AutoResearch — Autonomous Agent Experimentation for ML Contests]] _requires_ → here
- [[concepts/feature-selection-advanced|Advanced Feature Selection Techniques]] _works_with_ → here
- [[concepts/memory-optimization|Memory Optimization & Large Dataset Handling]] _works_with_ → here
- [[concepts/shap-feature-engineering|SHAP as a Feature Engineering Discovery Tool]] _works_with_ → here
- [[concepts/target-encoding-advanced|Advanced Target Encoding Techniques]] _works_with_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
