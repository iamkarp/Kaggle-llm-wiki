---
title: "Imbalanced Data Techniques for Kaggle"
tags: [imbalanced, smote, focal-loss, threshold-optimization, downsampling, tabular]
date: 2026-04-15
source_count: 7
status: active
---

## Summary

Top Kaggle solutions rarely use SMOTE on tabular data. The dominant pattern is: built-in GBDT weighting + stratified CV + threshold optimization + diverse sampling. SMOTE applied before cross-validation causes data leakage that inflates CV by +0.16 AUC.

## What It Is

A collection of competition-validated techniques for training models when one class is far more frequent than another. The correct technique depends on imbalance severity, model type, and evaluation metric.

## Key Facts / Details

### The Dominant Pattern (GBDTs on Tabular)

1. `scale_pos_weight` / `class_weight` built-in GBDT parameters
2. `StratifiedKFold` — non-negotiable
3. Threshold optimization on OOF predictions
4. Downsampling sandwich (for extreme imbalance >20:1)

### SMOTE — When It Helps vs Hurts

**SMOTE before split = DATA LEAKAGE. Always.**

| Scenario | Verdict |
|---|---|
| Large tabular GBDT problems | AVOID — adds noise |
| SMOTE before train/test split | ALWAYS WRONG — inflates CV +0.16 AUC |
| Neural nets on continuous features | OK |
| Medical image classification | OK |
| Small datasets (<1K minority) | OK |

**Correct SMOTE implementation:**
```python
from imblearn.pipeline import Pipeline
pipe = Pipeline([('smote', SMOTE()), ('clf', model)])
cv_score = cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
```

### scale_pos_weight / class_weight (Primary Approach)

```python
# XGBoost: count(negative) / count(positive)
# e.g., 97:3 imbalance → scale_pos_weight = 32
xgb.XGBClassifier(scale_pos_weight=32)

# LightGBM (never use both)
lgb.LGBMClassifier(is_unbalance=True)           # auto
lgb.LGBMClassifier(scale_pos_weight=32)         # manual (preferred)

# CatBoost
CatBoostClassifier(class_weights=[1, imbalance_ratio])
```

### Focal Loss (Medium-High Impact)

```python
# Original params (RetinaNet 2017): alpha=0.25, gamma=2.0
# Competition tuning:
# gamma: 1.0 (mild 10:1) to 3.0 (severe 100:1+)
# alpha: 0.5-0.75 for moderate imbalance
```

M5 Forecasting: focal loss documented as "major part of every top solution."

### Threshold Optimization (High ROI)

```python
from sklearn.metrics import fbeta_score
import numpy as np

oof_probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
thresholds = np.arange(0.01, 0.99, 0.01)
scores = [fbeta_score(y, oof_probs > t, beta=2) for t in thresholds]
optimal_threshold = thresholds[np.argmax(scores)]
```

In fraud/rare-event competitions with F2 metric: optimal threshold often 0.1–0.3 (not 0.5).

**sklearn 1.3+:** `TunedThresholdClassifierCV` automates this.

### Downsampling Sandwich (Porto Seguro Pattern, 26:1 imbalance)

```python
n_bags = 10  # approximately imbalance ratio
bag_preds = []
minority_X = X[y == 1]
majority_X = X[y == 0]

bag_size = len(majority_X) // n_bags
for i in range(n_bags):
    bag_majority_X = majority_X[i*bag_size:(i+1)*bag_size]
    X_bag = pd.concat([minority_X, bag_majority_X])
    y_bag = pd.concat([pd.Series(np.ones(len(minority_X))),
                       pd.Series(np.zeros(len(bag_majority_X)))])
    model.fit(X_bag, y_bag)
    bag_preds.append(model.predict_proba(X_test)[:, 1])

final_preds = np.mean(bag_preds, axis=0)
```

Used in: Porto Seguro 2017, Home Credit Default Risk 2018, AmEx Default 2022.

### Probability Calibration After Weighting

Training with modified class weights biases predicted probabilities. Calibrate when:
- Metric is log-loss
- Stacking models (miscalibrated probs corrupt the stacker)

```python
def correct_undersampling_proba(p_undersampled, beta):
    """beta = majority_size / minority_size in undersampled training set"""
    return p_undersampled / (p_undersampled + (1 - p_undersampled) / beta)
```

### Competition Impact Table

| Technique | Impact | Best For |
|---|---|---|
| scale_pos_weight / class_weight | High | GBDT tabular |
| Downsampling sandwich | High | Extreme imbalance >20:1 |
| Threshold optimization (OOF sweep) | High | F1/F2/recall metrics |
| Focal loss (GBDT custom objective) | Medium-High | Neural nets, RSNA winners |
| StratifiedKFold | High | All imbalanced problems |
| SMOTE in pipeline | Medium | NLP, image, small tabular only |
| Platt calibration post-undersampling | Medium | Stacked models, log-loss |
| **SMOTE before split** | **NEGATIVE** | **Never** |

## Gotchas

- Never apply SMOTE before train/test split — inflates CV AUC by ~0.16 (ArXiv 2412.07437)
- `is_unbalance=True` and `scale_pos_weight` in LightGBM should never both be set — they conflict
- PR-AUC is better than ROC-AUC for imbalanced: ROC-AUC can remain high when minority class performance is poor
- For multi-label: use `MultilabelStratifiedKFold` from `iterative-stratification` package

## Sources

- `raw/kaggle/imbalanced-data-techniques.md` *(not yet ingested)* — comprehensive technique reference
- [Imbalance-XGBoost paper](https://arxiv.org/pdf/1908.01672)
- [LightGBM Focal Loss (Max Halford)](https://maxhalford.github.io/blog/lightgbm-focal-loss/)
- [SMOTE leakage arXiv 2412.07437](https://arxiv.org/html/2412.07437v1)

## Related

- [[negative-downsampling]] — downsampling with prior correction, 5-bag averaging
- [[validation-strategy]] — StratifiedKFold, adversarial validation
- [[ensembling-strategies]] — ensemble approaches for imbalanced problems
- [[post-processing]] — threshold optimization, probability calibration
