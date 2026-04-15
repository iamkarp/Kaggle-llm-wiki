---
title: "XGBoost — eXtreme Gradient Boosting"
tags: [xgboost, gradient-boosting, tabular, framework, tool]
date: 2026-04-14
source_count: 1
status: active
---

## What It Is
XGBoost is a gradient-boosted decision tree library. It builds an ensemble of decision trees sequentially, each correcting the errors of the previous. Known for winning tabular data competitions on Kaggle from 2014–2020+.

Key advantages: fast training with parallelization, native NaN handling, built-in regularization (L1/L2), competitive with LightGBM on most tabular tasks.

## Typical Use in Jason's Work
- **March Mania**: Primary model family for all components (v2.8, v2.9, parts of v5)
- **Mega ensemble**: One component among many in `mega_ensemble.py`
- Outputs raw probabilities (usually reasonably calibrated on balanced data)

## Key Parameters Used
| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_estimators` | 500–600 | Higher with lower learning rate |
| `max_depth` | 5–9 | 9 for "Elite" component; 5-8 for diversity |
| `learning_rate` | 0.01–0.015 | Lower = more trees needed but better generalization |
| `objective` | `binary:logistic` | For classification/probability output |
| `eval_metric` | `logloss` | Monitors training progress |

## Performance Notes
- Depth 9 + lr 0.01 + 600 trees outperformed shallower configs for March Mania
- Feature importance API is useful but misleading with correlated features
- Early stopping (`early_stopping_rounds=50`) prevents over-training

## Installation
```bash
pip install xgboost
import xgboost as xgb
```

## Related
- [[../concepts/xgboost-ensembles]] — patterns for ensembling XGBoost models
- [[../entities/lightgbm]] — close competitor; faster, comparable performance
- [[../entities/catboost]] — better native categorical handling
- [[../strategies/march-mania-v6-ensemble]] — primary usage example
