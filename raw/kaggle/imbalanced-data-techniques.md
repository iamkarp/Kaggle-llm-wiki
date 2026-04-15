# Imbalanced Data Techniques for Kaggle Competitions

Compiled from competition winning solutions, Grandmaster writeups, and research papers. April 2026.

---

## Core Finding

Top Kaggle solutions RARELY use SMOTE on tabular data. The dominant pattern is:
1. Built-in GBDT weighting parameters + 
2. Stratified CV + 
3. Threshold optimization + 
4. Ensemble with diverse sampling

SMOTE is primarily used in medical imaging/NLP pipelines.

---

## 1. SMOTE — When It Helps vs Hurts

**Why SMOTE often hurts in tabular competitions:**
- Applied BEFORE cross-validation creates data leakage — inflates AUC by +0.16 artificially (ArXiv 2412.07437). Inflated CV crashes on private LB.
- Interpolating in feature space creates unrealistic combinations in high-dimensional tabular data with mixed categorical/continuous features.
- Tree models are not distance-based — synthetic interpolated points add noise more than signal.

**When SMOTE variants DO help:**
- Neural networks on purely continuous features
- Medical image classification
- Text classification
- Extremely small datasets (<1,000 minority class examples)

**Proper implementation — INSIDE pipeline, NEVER before split:**
```python
from imblearn.pipeline import Pipeline
pipe = Pipeline([('smote', SMOTE()), ('clf', model)])
cv_score = cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
```

**Variants:** ADASYN, Borderline-SMOTE — similar performance to vanilla SMOTE on tabular. K-Means SMOTE (interpolates only in sparse areas) is most principled variant.

---

## 2. Focal Loss — Settings That Win Competitions

**Original params (RetinaNet 2017):** alpha=0.25, gamma=2.0

**Competition tuning range:**
- gamma: 1.0–3.0 (1 for mild 10:1 imbalance, 2–3 for severe 100:1+)
- alpha: 0.5–0.75 for moderate imbalance

**For LightGBM custom objective** (Max Halford, used in multiple competitions):
```python
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a, g = alpha, gamma
    y_true = dtrain.get_label()
    p = 1 / (1 + np.exp(-y_pred))
    # Return gradient and hessian
    # See: https://maxhalford.github.io/blog/lightgbm-focal-loss/
```

**Focal loss for XGBoost:** Imbalance-XGBoost package (ArXiv 1908.01672). Provides analytically derived first and second-order derivatives. Showed improved F1 and MCC over vanilla XGBoost on imbalanced datasets.

**M5 Forecasting competition:** Focal loss documented as "a major part of every top solution."

---

## 3. Threshold Optimization — The Highest ROI Fix

Competition consensus: Use PR-AUC for evaluation; optimize threshold separately.

**Why PR-AUC > ROC-AUC for imbalanced:** ROC-AUC can remain high even when minority class performance is poor (large TN count inflates curve).

**Protocol:**
```python
from sklearn.metrics import f1_score, fbeta_score
import numpy as np

# 1. Get OOF predictions
oof_probs = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]

# 2. Sweep thresholds
thresholds = np.arange(0.01, 0.99, 0.01)
scores = [fbeta_score(y, oof_probs > t, beta=2) for t in thresholds]  # F2 example
optimal_threshold = thresholds[np.argmax(scores)]

# 3. Apply to test predictions
test_preds = model.predict_proba(X_test)[:, 1] > optimal_threshold
```

**Key insight:** In fraud/rare-event competitions where metric is F2 (recall-weighted), optimal threshold is often 0.1–0.3, far below default 0.5.

**sklearn 1.3+:** `TunedThresholdClassifierCV` automates this with proper CV.

---

## 4. Cost-Sensitive Learning — The #1 Practical Approach for GBDTs

```python
# XGBoost scale_pos_weight
# Formula: count(negative) / count(positive)
# e.g., 97:3 imbalance → scale_pos_weight = 97/3 ≈ 32
xgb.XGBClassifier(scale_pos_weight=32)  # Starting point; tune 0.5x–2x ratio

# LightGBM (one or the other, never both)
lgb.LGBMClassifier(is_unbalance=True)           # auto-computes weights
lgb.LGBMClassifier(scale_pos_weight=32)         # manual (preferred by top competitors)

# CatBoost
CatBoostClassifier(class_weights=[1, imbalance_ratio])
```

---

## 5. Downsampling Sandwich (Porto Seguro Pattern)

Discovered in Porto Seguro 2017 (26:1 imbalance). Widely copied since:

1. Compute imbalance ratio R (majority:minority)
2. Split majority class into R equal partitions
3. Train R models, each on [full minority + 1 majority partition]
4. Average R models' predictions

This "bags" the majority class without losing information. Used in Home Credit Default Risk (2018), AMEX Default Prediction (2022).

```python
n_bags = 10  # approximately equal to imbalance ratio
bag_preds = []
minority_X = X[y == 1]
minority_y = y[y == 1]
majority_X = X[y == 0]
majority_y = y[y == 0]

bag_size = len(majority_X) // n_bags
for i in range(n_bags):
    bag_majority_X = majority_X[i*bag_size:(i+1)*bag_size]
    bag_majority_y = majority_y[i*bag_size:(i+1)*bag_size]
    X_bag = pd.concat([minority_X, bag_majority_X])
    y_bag = pd.concat([minority_y, bag_majority_y])
    model.fit(X_bag, y_bag)
    bag_preds.append(model.predict_proba(X_test)[:, 1])

final_preds = np.mean(bag_preds, axis=0)
```

---

## 6. Probability Calibration for Imbalanced Classes

Training with modified class weights biases predicted probabilities — they no longer represent true posteriors. Calibrate when:
- Metric is log-loss
- Stacking models (miscalibrated probs corrupt the stacker)

**Undersampling correction formula:**
```python
def correct_undersampling_proba(p_undersampled, beta):
    """beta = majority_size / minority_size in undersampled training set"""
    return p_undersampled / (p_undersampled + (1 - p_undersampled) / beta)
```

**Platt Scaling:** Preferred when classifier is biased toward majority class. Best for smaller calibration sets.

---

## 7. CV Stratification Rules

**Non-negotiable:** Always use `StratifiedKFold` for imbalanced classification. Standard KFold can produce folds with zero minority class samples.

**SMOTE in CV (correct pattern):**
```python
# WRONG — leakage
X_res, y_res = SMOTE().fit_resample(X, y)
cv_score = cross_val_score(model, X_res, y_res)

# RIGHT — no leakage
pipe = Pipeline([('smote', SMOTE()), ('model', clf)])
cv_score = cross_val_score(pipe, X, y, cv=StratifiedKFold(5))
```

**For multi-label:** `MultilabelStratifiedKFold` from `iterative-stratification` package.

**5-fold vs 10-fold:** 10-fold preferred when minority class is very small — ensures enough minority examples per fold.

---

## Summary: Competition Tier by Technique

| Technique | Competition Impact | Best For | Avoid When |
|---|---|---|---|
| scale_pos_weight / class_weight | **High** — consistently in top solutions | GBDT on tabular | Log-loss metric (miscalibrates) |
| Downsampling sandwich | **High** — Porto Seguro, Home Credit, AMEX | Extreme imbalance (>20:1) | Small datasets |
| Threshold optimization (OOF sweep) | **High** — almost universal | F1/F2/recall metrics | AUC or log-loss |
| Focal Loss (GBDT custom objective) | **Medium-High** — M5, RSNA winners | Neural nets, custom GBDT | Pure trees without tuning |
| StratifiedKFold | **High** — standard practice | All imbalanced problems | Never skip |
| SMOTE in pipeline | **Medium** — neural nets only | NLP, image, small tabular | Large tabular GBDT problems |
| Platt calibration post-undersampling | **Medium** | Stacked models, log-loss | ROC-AUC-only metrics |
| SMOTE before split | **NEGATIVE** — causes leakage | Never | Always |

---

Sources:
- Imbalance-XGBoost paper: https://arxiv.org/pdf/1908.01672
- LightGBM Focal Loss (Max Halford): https://maxhalford.github.io/blog/lightgbm-focal-loss/
- Leakage from SMOTE before CV: https://arxiv.org/html/2412.07437v1
- RSNA 2019 ICH 1st place: https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection
- Porto Seguro 18th place solution: https://jeddy92.github.io/seguro/
- Platt scaling limitations paper: https://arxiv.org/pdf/2410.18144
- NVIDIA fraud detection guide: https://developer.nvidia.com/blog/leveraging-machine-learning-to-detect-fraud-tips-to-developing-a-winning-kaggle-solution/
