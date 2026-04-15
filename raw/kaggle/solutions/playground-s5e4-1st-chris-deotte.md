# Playground Series S5E4 — 1st Place Solution
**Author**: Chris Deotte | **Year**: 2025 | **Votes**: 233

---

## Competition
Kaggle Playground Series tabular competition. Synthetic dataset generated from a real-world dataset. Binary classification. Demonstrates Chris Deotte's modern stacking playbook in its most accessible form — smaller scale than Home Credit but same structural principles.

## The Modern Deotte Stacking Playbook (Simplified Form)

This competition serves as the clearest illustration of Deotte's 3-level stack with RAPIDS cuML — the same architecture used in production-grade solutions, but implementable in a playground competition timeframe.

### Level 1: Diverse Base Models via RAPIDS cuML

RAPIDS cuML provides GPU-accelerated sklearn-compatible implementations. On a single GPU, training 50+ models that would take hours with CPU sklearn runs in minutes:

```python
import cuml
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.linear_model import LogisticRegression as cuLR
from cuml.neighbors import KNeighborsClassifier as cuKNN
from cuml.svm import SVC as cuSVC
import lightgbm as lgb
import xgboost as xgb

# All cuML models have sklearn-compatible fit/predict_proba API
l1_models = {
    'rf_100':    cuRFC(n_estimators=100, max_depth=8),
    'rf_300':    cuRFC(n_estimators=300, max_depth=10),
    'rf_500':    cuRFC(n_estimators=500, max_depth=12),
    'lr_l1':     cuLR(C=0.1, penalty='l1', solver='qn'),
    'lr_l2':     cuLR(C=1.0, penalty='l2'),
    'lr_l2_c10': cuLR(C=10.0, penalty='l2'),
    'knn_5':     cuKNN(n_neighbors=5),
    'knn_15':    cuKNN(n_neighbors=15),
    'knn_51':    cuKNN(n_neighbors=51),
    'lgb_deep':  lgb.LGBMClassifier(num_leaves=127, learning_rate=0.05),
    'lgb_wide':  lgb.LGBMClassifier(num_leaves=31, n_estimators=500),
    'xgb_d6':   xgb.XGBClassifier(max_depth=6, learning_rate=0.05),
    'xgb_d8':   xgb.XGBClassifier(max_depth=8, learning_rate=0.03),
    # ... more variants
}
```

### OOF Generation (Standard Pattern)

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
n_models = len(l1_models)

oof_train = np.zeros((len(X_train), n_models))
oof_test = np.zeros((len(X_test), n_models))

for i, (name, model) in enumerate(l1_models.items()):
    test_fold_preds = []
    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        model.fit(X_train[tr_idx], y_train[tr_idx])
        oof_train[val_idx, i] = model.predict_proba(X_train[val_idx])[:, 1]
        test_fold_preds.append(model.predict_proba(X_test)[:, 1])
    oof_test[:, i] = np.mean(test_fold_preds, axis=0)
```

### Level 2: Ridge Meta-Learner

```python
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV

# Ridge at L2: fast, interpretable, doesn't overfit on ~n_models input features
ridge = CalibratedClassifierCV(RidgeClassifier(alpha=1.0), cv=5, method='sigmoid')
ridge.fit(oof_train, y_train)

l2_oof = ridge.predict_proba(oof_train)[:, 1]  # L2 OOF
l2_test = ridge.predict_proba(oof_test)[:, 1]  # L2 test predictions
```

### Level 3: Blend

```python
# Optionally add a strong L1 model directly in the final blend
lgb_best_l1_oof = oof_train[:, l1_models_list.index('lgb_deep')]
lgb_best_l1_test = oof_test[:, l1_models_list.index('lgb_deep')]

# Blend L2 stacker with best L1 model
final_preds = 0.7 * l2_test + 0.3 * lgb_best_l1_test
```

This 3-level structure (diverse L1 → Ridge L2 → blend L3) is the minimal version of the 90-model Home Credit architecture.

## Why RAPIDS cuML Enables Fast Stacking

Training 50 models × 10 folds = 500 model fits. With sklearn on CPU:
- Random Forest (500 trees): ~30 sec/fit → 500 × 30s = ~4 hours
- KNN: ~5 sec/fit → manageable
- Total: potentially 6+ hours for L1 alone

With RAPIDS cuML on GPU:
- Random Forest (500 trees): ~1 sec/fit → 500 × 1s = ~8 minutes
- KNN: ~0.1 sec/fit
- Total: ~15 minutes for L1

This speed difference determines whether a stacking approach is viable in a competition timeframe.

## Playground Series Specifics

Kaggle Playground competitions use synthetic data generated from a real-world dataset. This creates two useful strategies:

### Strategy 1: Use Original Dataset
If the original dataset is public (often disclosed in the data tab), include it in training alongside the synthetic data:
```python
# Often improves performance significantly
original_data = pd.read_csv('original_dataset.csv')
train_combined = pd.concat([synthetic_train, original_data])
```

### Strategy 2: Detect and Exploit Duplicates
Synthetic generation sometimes creates near-duplicate rows between train and test. Feature engineering that detects these can boost score:
```python
# Nearest-neighbor distance features
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=1)
nn.fit(X_train)
distances, _ = nn.kneighbors(X_test)
X_test['min_train_dist'] = distances[:, 0]
```

## Key Takeaways
1. RAPIDS cuML reduces 50-model L1 training from hours to minutes — makes stacking viable in playground timeframes
2. The 3-level structure (diverse GPU-accelerated L1 → Ridge L2 → blend) is Deotte's minimal stacking template
3. For playground competitions: always check if original dataset is available and include it
4. 10-fold CV at L1 produces stable OOF even with modest dataset sizes
5. Ridge L2 with calibration: fast, stable, and interpretable meta-weights
