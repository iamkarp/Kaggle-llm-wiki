---
title: Gradient Boosted Trees (LightGBM, XGBoost, CatBoost)
category: techniques
tags: [gbm, tabular, lightgbm, xgboost, catboost, ensemble]
created: 2026-04-15
updated: 2026-04-15
---

# Gradient Boosted Trees

The default choice for tabular data on Kaggle. LightGBM, XGBoost, and CatBoost each have different inductive biases, making them good candidates for ensembling.

## Typical Setup for Kaggle

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    objective='multiclass', num_class=19,
    num_leaves=127, learning_rate=0.05,
    n_estimators=2000, class_weight='balanced',
    colsample_bytree=0.7, subsample=0.7,
    reg_alpha=0.1, reg_lambda=1.0,
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100)])
```

## Ensemble Strategy

- Train LightGBM + XGBoost (+ CatBoost if time permits)
- Search optimal blend weights on OOF: `w_lgb * P_lgb + w_xgb * P_xgb`
- Different tree methods provide diversity (leaf-wise vs level-wise vs symmetric)

## When GBM Fails

**Cross-subject/cross-domain generalization**: GBM relies on handcrafted features that may encode subject-specific patterns. In WEAR 2026, GBM achieved OOF F1 ~0.59 but LB only ~0.24. See [[mistakes/gbm-for-cross-subject-har]].

The problem is not GBM itself but the features: statistical summaries (mean, std, FFT) of accelerometer data capture individual movement characteristics that don't transfer across people.

## Platform Constraints

- LightGBM + XGBoost: CPU-efficient, fit within Kaggle 12h limit
- CatBoost on CPU: Very slow for multi-class. 4-fold CV exceeded 12h in WEAR. See [[mistakes/catboost-cpu-timeout]]
- All three support GPU, but Kaggle GPU has its own issues (see [[tools/kaggle-cpu-notebooks]])

## See Also
- [[mistakes/gbm-for-cross-subject-har]]
- [[mistakes/catboost-cpu-timeout]]
- [[patterns/cross-subject-generalization]]
