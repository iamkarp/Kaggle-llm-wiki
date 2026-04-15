# Advanced Feature Selection Techniques for Kaggle Competitions

Compiled from Home Credit/IEEE-CIS competition notebooks, Kaggle GMs, research papers. April 2026.

---

## The Grandmaster Pattern

Chris Deotte (Feb 2025 1st place): Generate 10,000+ features with cuDF-pandas, then apply aggressive selection pipelines. Key insight: **generating many features first, then selecting rigorously, consistently outperforms careful manual feature engineering.**

---

## 1. Null Importance (Target Permutation Importances) — The Gold Standard

Introduced by Olivier Grellier (`ogrellier`) for Home Credit Default Risk. Used extensively in IEEE-CIS Fraud Detection (400+ V-columns).

**Concept:** Train model on real target → get actual importances. Shuffle target N times (50–100) and retrain each time → build null distribution. Keep features where actual importance clearly exceeds null distribution.

**Why it beats standard importance:** Standard gain-based importance rewards features that correlate with training noise. The null distribution quantifies exactly how much importance a useless feature accumulates by chance.

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
    imp = pd.DataFrame({'feature': data.columns,
                        'importance': clf.feature_importance(importance_type='gain')})
    return imp

actual_imp = get_feature_importances(X, y, shuffle=False)

null_imps = []
for i in range(50):
    null_imps.append(get_feature_importances(X, y, shuffle=True, seed=i))
null_imps_df = pd.concat(null_imps)

def score_feature(feature):
    actual = actual_imp.loc[actual_imp.feature == feature, 'importance'].values[0]
    null_dist = null_imps_df.loc[null_imps_df.feature == feature, 'importance'].values
    return 100.0 * (null_dist < actual).sum() / len(null_dist)

feature_scores = {f: score_feature(f) for f in X.columns}
selected_features = [f for f, s in feature_scores.items() if s >= 80]  # actual beats 80%+ of null runs
```

Package: https://github.com/kingychiu/target-permutation-importances
Reference: https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances

---

## 2. BorutaShap — Best Hybrid Method

Combines Boruta's shadow-feature methodology with SHAP values. Recommended over plain Boruta.

```python
from BorutaShap import BorutaShap
from sklearn.ensemble import RandomForestClassifier

selector = BorutaShap(model=RandomForestClassifier(), importance_measure='shap', classification=True)
selector.fit(X=X, y=y, n_trials=100, random_state=0)
selector.plot(which_features='all')
selected_features = selector.Subset().columns.tolist()
```

Package: https://github.com/Ekeany/Boruta-Shap
Tutorial: https://www.kaggle.com/code/lucamassaron/tutorial-feature-selection-with-boruta-shap

**Plain Boruta** (when you don't have SHAP available):
```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
selector.fit(X.values, y.values)
selected = X.columns[selector.support_].tolist()
tentative = X.columns[selector.support_weak_].tolist()
```

---

## 3. LOFO (Leave One Feature Out) Importance

Measures drop in OOF CV score when each feature is removed. Detects features that cause overfitting even if they look important on training data.

```python
from lofo import LOFOImportance, Dataset, plot_importance
import lightgbm as lgb

dataset = Dataset(df=train_df, target='target', features=feature_cols)
lofo_imp = LOFOImportance(dataset, cv=4, scoring='roc_auc')
importance_df = lofo_imp.get_importance()
plot_importance(importance_df, figsize=(12, 8))

to_drop = importance_df[importance_df['importance_mean'] < 0]['feature'].tolist()
```

Package: https://github.com/aerdem4/lofo-importance
Used in Ubiquant competition: https://www.kaggle.com/code/aerdem4/ubiquant-lofo-feature-importance

---

## 4. Adversarial Validation for Feature Selection

Drop features that distinguish train from test — they hurt LB generalization.

```python
import lightgbm as lgb
import pandas as pd
import numpy as np

train_adv = train_df[feature_cols].copy(); train_adv['is_test'] = 0
test_adv  = test_df[feature_cols].copy();  test_adv['is_test'] = 1
combined  = pd.concat([train_adv, test_adv]).reset_index(drop=True)

X_adv = combined[feature_cols]
y_adv = combined['is_test']

model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31)
scores = cross_val_score(model, X_adv, y_adv, cv=5, scoring='roc_auc')
print(f"Adversarial AUC: {scores.mean():.4f}")  # Near 0.5 = good

if scores.mean() > 0.6:
    model.fit(X_adv, y_adv)
    adv_importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    features_to_investigate = adv_importance.head(10).index.tolist()
    # Selectively drop features that have high importance AND low target correlation
```

---

## 5. Mutual Information — Fast Filter Stage

Non-parametric, captures nonlinear dependencies. Best used as Stage 1 for rapid pruning.

```python
from sklearn.feature_selection import mutual_info_classif, SelectKBest

mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=0)
mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# For regression
from sklearn.feature_selection import mutual_info_regression
mi_scores = mutual_info_regression(X, y, random_state=0)
```

**High-dim workflow (10k features):** MI as Stage 1 (fast, keeps top 500–1000) → null importance or Boruta-SHAP as Stage 2.

---

## 6. Variance Inflation Factor (VIF) — Multicollinearity

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

def iterative_vif_selection(df, threshold=10.0):
    features = df.columns.tolist()
    while True:
        vif_data = pd.DataFrame()
        vif_data['feature'] = features
        vif_data['VIF'] = [variance_inflation_factor(df[features].values, i) 
                           for i in range(len(features))]
        max_vif = vif_data['VIF'].max()
        if max_vif <= threshold:
            break
        worst = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
        features.remove(worst)
    return features
```

VIF > 10 = serious multicollinearity; VIF > 5 = warrants investigation.

---

## 7. Correlation-Based Pruning — Fastest Method

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

---

## 8. SFFS — Sequential Floating Forward Selection

```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from lightgbm import LGBMClassifier

sfs = SFS(LGBMClassifier(n_estimators=200),
          k_features=(5, 50),
          forward=True,
          floating=True,     # SFFS: allows backward steps to escape local optima
          scoring='roc_auc',
          cv=5,
          n_jobs=-1)
sfs.fit(X, y)
selected_features = list(sfs.k_feature_names_)
```

Also available in sklearn 1.0+: `sklearn.feature_selection.SequentialFeatureSelector`.

---

## 9. Stability Selection (Bootstrap + LASSO)

```python
from stability_selection import StabilitySelection
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

base_estimator = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='liblinear', C=0.1))
])
selector = StabilitySelection(base_estimator=base_estimator,
                               lambda_name='model__C',
                               lambda_grid=np.logspace(-5, -1, 50),
                               threshold=0.6,
                               n_bootstrap_iterations=100)
selector.fit(X.values, y.values)
```

Package: https://github.com/scikit-learn-contrib/stability-selection
Provides theoretical false-discovery-rate control.

---

## Recommended Competition Pipeline

| Stage | Method | Goal | Speed |
|---|---|---|---|
| 1 | Correlation pruning (>0.95) | Remove duplicates | Fast |
| 2 | Mutual information | Remove zero-signal | Fast |
| 3 | Adversarial validation | Drop distribution-drift features | Medium |
| 4 | Null importance (50 runs) | Statistical signal vs. noise test | Slow |
| 5 | LOFO / Boruta-SHAP | Confirm final subset | Slow |
| 6 | Forward selection (optional) | Fine-tune minimal set | Very slow |

---

Sources:
- Null importance notebook: https://www.kaggle.com/code/ogrellier/feature-selection-with-null-importances
- BorutaShap tutorial: https://www.kaggle.com/code/lucamassaron/tutorial-feature-selection-with-boruta-shap
- LOFO importance: https://github.com/aerdem4/lofo-importance
- target-permutation-importances package: https://github.com/kingychiu/target-permutation-importances
- SHAP-select arXiv 2024: https://arxiv.org/html/2410.06815v1
