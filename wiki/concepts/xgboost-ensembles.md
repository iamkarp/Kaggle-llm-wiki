---
title: "XGBoost Ensembles — Weighted Averaging of Gradient Boosted Trees"
tags: [xgboost, ensemble, gradient-boosting, hyperparameters, tabular]
date: 2026-04-14
source_count: 2
status: active
---

## What It Is
XGBoost (eXtreme Gradient Boosting) is a gradient-boosted decision tree library. Ensembling multiple XGBoost models with different hyperparameters (especially `max_depth`) reduces variance while preserving the expressive power of deep trees.

The core idea: a single deep XGBoost model may overfit to quirks in training data. Multiple models at different depths, averaged together, smooth out those quirks without losing the non-linear patterns each captures.

## When To Use It
- Tabular data competitions where features include both strong signals and noise
- When a single model achieves good CV but worse LB (variance problem)
- When you have enough compute to train 3–10 models independently
- As a first ensembling step before stacking

## Hyperparameters (Key Ones)

| Parameter | Role | Typical Range | Jason's Usage |
|-----------|------|--------------|---------------|
| `n_estimators` | Number of trees | 100–1000 | 500–600 |
| `max_depth` | Tree depth (variance control) | 3–10 | 5–9 |
| `learning_rate` | Step size (regularization) | 0.005–0.1 | 0.01–0.015 |
| `subsample` | Row subsampling per tree | 0.6–1.0 | typically 0.8 |
| `colsample_bytree` | Feature subsampling | 0.6–1.0 | typically 0.8 |
| `min_child_weight` | Leaf regularization | 1–10 | default |
| `gamma` | Min split loss | 0–5 | default |

## Gotchas
- **Deep trees overfit** on small datasets — `max_depth=9` needs careful CV
- **Learning rate and n_estimators are coupled** — lower LR needs more trees (use early stopping)
- **Feature importance can be misleading** on correlated features — use permutation importance for interpretability
- **Probability calibration**: XGBoost probabilities are generally reasonable but can be miscalibrated on imbalanced data

## In Jason's Work

### March Mania v6 Ensemble
Three XGBoost variants at depths 5–9, combined with fixed weights:
- v2.9 (depth=9, 600 trees, lr=0.01) — highest expressiveness
- v2.8 (depth=8, 500 trees, lr=0.015) — moderate regularization
- v5 includes depths 5–8 as sub-components

Key finding: deeper trees (9) + lower learning rate (0.01) outperformed shallower/faster combos for basketball prediction. Full-season feature signals favored capacity.

### Wine Competition
`wine_catboost.py` — CatBoost used instead of XGBoost for categorical handling; similar gradient boosting family.

### Mega Ensemble
`mega_ensemble.py` uses XGBoost as one component alongside LightGBM, CatBoost, RF, LogReg, SVM, MLP.

## Sources
- [[../../raw/kaggle/v6-ensemble-documentation.md]] — depth/lr choices for March Mania
## Related
- [[../strategies/march-mania-v6-ensemble]] — primary usage example
- [[../entities/xgboost]] — framework entity page
- [[../entities/lightgbm-catboost]] — sister frameworks
- [[../concepts/calibration]] — probability calibration for tree models
- [[../concepts/feature-engineering-tabular]] — features fed into these models
