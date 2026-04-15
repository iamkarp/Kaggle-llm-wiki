---
title: "Advanced Feature Selection Techniques"
tags: [feature-selection, null-importance, boruta, lofo, adversarial-validation, shap, tabular]
date: 2026-04-15
source_count: 5
status: active
---

## Summary

The Grandmaster pattern: generate 10,000+ features, then apply a rigorous multi-stage selection pipeline. Null importance (target permutation) is the gold standard for distinguishing signal from noise. SHAP-based methods (BorutaShap, ShapRFECV) outperform gain-based importance which is biased toward high-cardinality features.

## What It Is

A suite of competition-validated feature selection techniques, from fast correlation pruning to computationally intensive null importance and Boruta-SHAP. Used in sequence as a pipeline, not as alternatives.

## Key Facts / Details

### Recommended Competition Pipeline

| Stage | Method | Goal | Speed |
|---|---|---|---|
| 1 | Correlation pruning (>0.95) | Remove duplicates | Fast |
| 2 | Mutual information | Remove zero-signal | Fast |
| 3 | Adversarial validation | Drop distribution-drift features | Medium |
| 4 | Null importance (50 runs) | Statistical signal vs. noise test | Slow |
| 5 | LOFO / Boruta-SHAP | Confirm final subset | Slow |
| 6 | Forward selection (optional) | Fine-tune minimal set | Very slow |

### 1. Null Importance (Gold Standard)

Introduced by Olivier Grellier for Home Credit Default Risk. Used in IEEE-CIS Fraud (400+ V-columns).

```python
import lightgbm as lgb
import numpy as np
import pandas as pd

def get_feature_importances(data, target, shuffle=False, seed=None):
    y = target.sample(frac=1.0, random_state=seed).reset_index(drop=True) if shuffle else target.copy()
    dtrain = lgb.Dataset(data, label=y, free_raw_data=False, silent=True)
    params = {
        'objective': 'binary', 'boosting_type': 'rf',
        'subsample': 0.623, 'colsample_bytree': 0.7,
        'num_leaves': 127, 'max_depth': 8,
        'seed': seed, 'bagging_freq': 1, 'n_jobs': 4
    }
    clf = lgb.train(params=params, train_set=dtrain, num_boost_round=200, verbose_eval=False)
    return pd.DataFrame({'feature': data.columns,
                        'importance': clf.feature_importance(importance_type='gain')})

actual_imp = get_feature_importances(X, y, shuffle=False)

null_imps = [get_feature_importances(X, y, shuffle=True, seed=i) for i in range(50)]
null_imps_df = pd.concat(null_imps)

def score_feature(feature):
    actual = actual_imp.loc[actual_imp.feature == feature, 'importance'].values[0]
    null_dist = null_imps_df.loc[null_imps_df.feature == feature, 'importance'].values
    return 100.0 * (null_dist < actual).sum() / len(null_dist)

selected = [f for f, s in {f: score_feature(f) for f in X.columns}.items() if s >= 80]
```

**Why it beats standard importance:** Gain importance rewards features that correlate with training noise. Null distribution quantifies exactly how much importance a useless feature gets by chance.

### 2. BorutaShap (Best Hybrid)

Boruta's shadow-feature methodology + SHAP values. Outperforms plain Boruta.

```python
from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier

selector = BorutaShap(model=RandomForestClassifier(), importance_measure='shap', classification=True)
selector.fit(X=X, y=y, n_trials=100, random_state=0)
selected_features = selector.Subset().columns.tolist()
```

### 3. LOFO (Leave One Feature Out)

Measures drop in OOF CV score when each feature is removed. Detects overfitting features that look important on training data.

```python
from lofo import LOFOImportance, Dataset, plot_importance

dataset = Dataset(df=train_df, target='target', features=feature_cols)
lofo_imp = LOFOImportance(dataset, cv=4, scoring='roc_auc')
importance_df = lofo_imp.get_importance()
to_drop = importance_df[importance_df['importance_mean'] < 0]['feature'].tolist()
```

### 4. Adversarial Validation for Feature Selection

Drop features that distinguish train from test — they hurt LB generalization.

```python
import lightgbm as lgb

train_adv = train_df[feature_cols].copy(); train_adv['is_test'] = 0
test_adv  = test_df[feature_cols].copy();  test_adv['is_test'] = 1
combined  = pd.concat([train_adv, test_adv]).reset_index(drop=True)

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31)
scores = cross_val_score(model, combined[feature_cols], combined['is_test'], cv=5, scoring='roc_auc')

if scores.mean() > 0.6:
    model.fit(combined[feature_cols], combined['is_test'])
    adv_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    features_to_investigate = adv_importance.head(10).index.tolist()
```

### 5. Mutual Information (Fast Filter)

```python
from sklearn.feature_selection import mutual_info_classif

mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
# Stage 1: keep top 500-1000 features from 10K
```

### 6. Correlation Pruning (Fastest)

```python
def correlation_pruning(df, target, threshold=0.95):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    target_corr = df.corrwith(target).abs()
    to_drop = set()
    for col in upper.columns:
        for partner in upper.index[upper[col] > threshold].tolist():
            if col not in to_drop and partner not in to_drop:
                to_drop.add(col if target_corr[col] < target_corr[partner] else partner)
    return [c for c in df.columns if c not in to_drop]
```

### 7. ShapRFECV (ING Bank probatus)

Published result: eliminated 60/110 features → slight AUC improvement + significant complexity reduction.

```python
from probatus.feature_elimination import ShapRFECV
from lightgbm import LGBMClassifier

shap_rfecv = ShapRFECV(LGBMClassifier(n_estimators=100), step=0.2, cv=5,
                        scoring='roc_auc', n_iter=10, random_state=42)
shap_rfecv.fit_compute(X_train, y_train)
selected = shap_rfecv.get_reduced_features_set(num_features=30)
```

## When To Use It

- Whenever you have more than 50 features
- After generating GroupBy aggregations (which creates hundreds of features)
- Before hyperparameter tuning (smaller feature set = faster experimentation)
- Always use SHAP-based selection over gain-based selection for high-cardinality features

## Gotchas

- Standard gain importance is biased toward high-cardinality features → use SHAP
- Never skip feature selection when features > 200 (even strong models overfit)
- Null importance takes 50+ model runs — cache results
- LOFO can produce negative importance for features that cause overfitting
- Never use VIF alone for tree models (VIF is for linear models)

## In Jason's Work

ShapRFECV and null importance are the recommended methods for tabular competitions with many engineered features. Run correlation pruning and MI first to reduce to manageable set, then null importance for final validation.

## Sources

- [[../raw/kaggle/advanced-feature-selection.md]] — complete technique reference with code
- [Null importance notebook (ogrellier)](https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances)
- [BorutaShap tutorial](https://www.kaggle.com/code/lucamassaron/tutorial-feature-selection-with-boruta-shap)
- [LOFO importance](https://github.com/aerdem4/lofo-importance)
- [ShapRFECV (probatus)](https://medium.com/ing-blog/open-sourcing-shaprfecv-improved-feature-selection-powered-by-shap-994fe7861560)

## Related

- [[concepts/shap-feature-engineering]] — SHAP as feature engineering discovery tool
- [[concepts/feature-engineering-tabular]] — generating the features to then select from
- [[concepts/validation-strategy]] — adversarial validation for distribution shift
- [[concepts/gradient-boosting-advanced]] — GBDT models used for selection
