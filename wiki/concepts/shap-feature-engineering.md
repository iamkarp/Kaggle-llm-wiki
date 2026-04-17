---
title: "SHAP as a Feature Engineering Discovery Tool"
tags: [shap, feature-engineering, interpretability, interactions, drift-detection, tabular]
date: 2026-04-15
source_count: 6
status: active
---

## Summary

SHAP is not just for explaining models — it's an active feature engineering tool. SHAP dependency plots reveal what transformations the model wants (log, logit, bins). SHAP interaction values identify the highest-value feature pairs to engineer explicitly. Error analysis via SHAP exposes where and why the model systematically fails.

## What It Is

A methodology for using Shapley values to drive feature engineering decisions rather than just post-hoc explanation. The NVIDIA Grandmasters Playbook explicitly lists error analysis via SHAP among its 7 battle-tested techniques.

## Key Facts / Details

### 1. SHAP Dependency Plots → Transformation Discovery

The shape of the SHAP dependency curve tells you what transformation the model is approximating:

| Curve shape | Model is approximating | Action |
|---|---|---|
| Logarithmic curve | `log(x)` | Add `np.log1p(x)` as explicit feature |
| S-shaped (sigmoid-like) | Logit transform | Add `scipy.special.logit(x)` |
| Flat then steep slope | Threshold at inflection | Bin feature around that value |
| Large vertical spread | Interaction with another variable | Color = interaction partner → engineer product/ratio |

**Key rule:** Vertical dispersion in a SHAP dependency plot = interaction effect. Zero dispersion = purely additive. When you see dispersion, look at the coloring variable — that's your top interaction pair.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Dependency plot for feature 'age' (auto-selects best interaction coloring)
shap.dependence_plot("age", shap_values, X_val)
# Or explicit interaction
shap.dependence_plot("age", shap_values, X_val, interaction_index="income")
```

### 2. SHAP Interaction Values → Top Pairs to Engineer

```python
shap_interaction_values = explainer.shap_interaction_values(X)
# Shape: (n_samples, n_features, n_features)

mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
np.fill_diagonal(mean_abs_interactions, 0)  # zero out self-interactions

pairs = []
for i in range(mean_abs_interactions.shape[0]):
    for j in range(i+1, mean_abs_interactions.shape[1]):
        pairs.append((mean_abs_interactions[i,j], X.columns[i], X.columns[j]))
pairs.sort(reverse=True)
top_interactions = pairs[:10]

# Engineer explicit features for top pairs
for _, feat_a, feat_b in top_interactions:
    df[f'{feat_a}_x_{feat_b}'] = df[feat_a] * df[feat_b]
    df[f'{feat_a}_div_{feat_b}'] = df[feat_a] / (df[feat_b] + 1e-6)
```

**Measured result:** Cancer prediction — SHAP-guided interaction FE improved accuracy from 0.8794 → 0.8968.

### 3. Error Analysis with SHAP → Targeted Feature Discovery

```python
oof_preds = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
residuals = np.abs(y - oof_preds)

high_error_mask = residuals > np.percentile(residuals, 80)
low_error_mask = residuals < np.percentile(residuals, 20)

shap_values = explainer.shap_values(X)
high_error_shap = np.abs(shap_values[high_error_mask]).mean(axis=0)
low_error_shap = np.abs(shap_values[low_error_mask]).mean(axis=0)

divergence = high_error_shap - low_error_shap
divergence_series = pd.Series(divergence, index=X.columns).sort_values(ascending=False)
# Features with high divergence → these need transformed versions or interaction terms
```

**Chris Deotte's approach:** Train Stage 2 model on Stage 1's residuals as target — forces second model to focus on what first model got wrong.

### 4. SHAP-Space Drift Detection

```python
from scipy import stats

shap_train = explainer.shap_values(X_train)
shap_test = explainer.shap_values(X_test)

for i, feature in enumerate(X_train.columns):
    ks_stat, p_value = stats.ks_2samp(shap_train[:, i], shap_test[:, i])
    if ks_stat > 0.1:
        print(f"SHAP drift: {feature} (KS={ks_stat:.3f})")
```

**Why SHAP-space drift > raw feature drift:** A feature with large raw shift but near-zero SHAP values is irrelevant. A feature with small raw shift but large SHAP shift is critical.

**When drift detected:** Re-engineer that feature to be more stable, use pseudo-labeling to recalibrate, or down-weight during inference.

### 5. ShapRFECV (Recursive Feature Elimination)

See [[feature-selection-advanced]] for full implementation.

Published result: eliminated 60/110 features → slight AUC improvement + significant complexity reduction.

### 6. SHAP Clustering for Subgroup Discovery

```python
from sklearn.cluster import KMeans

shap_values_array = explainer.shap_values(X)
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(shap_values_array)

for cluster in range(5):
    cluster_mask = cluster_labels == cluster
    cluster_shap = shap_values_array[cluster_mask].mean(axis=0)
    top_features = pd.Series(cluster_shap, index=X.columns).abs().sort_values(ascending=False).head(5)
    print(f"Cluster {cluster}: {top_features.index.tolist()}")
    # → Subpopulations may need cluster-conditional features
```

### Competition-Ready Workflow

```
1. Train CV model → OOF predictions + OOF SHAP values

2. SHAP dependency plots for top 20 features → identify:
   - Nonlinear curve shape → add explicit transformation
   - Vertical dispersion → top interaction pair → add product/ratio feature

3. SHAP interaction matrix → extract top 10 off-diagonal pairs → engineer interactions

4. High-residual sample analysis:
   - Filter top 20% error samples
   - Compare SHAP vs. low-error samples
   - Divergent features → engineer targeted feature

5. SHAP beeswarm/summary → prune bottom 20% by mean |SHAP| → retrain

6. If test data available → SHAP-space KS test for drift:
   - Flag drifted features → pseudo-labeling or feature recalibration
```

## When To Use It

- After training first baseline model (need SHAP values to analyze)
- When stuck on plateau after initial feature engineering
- When model has high error on specific subpopulations
- Before submission to check if test distribution differs from train

## Gotchas

- SHAP interaction values (`shap_interaction_values`) only work with `TreeExplainer`
- Computing SHAP on full training set is slow — use a representative sample (10K rows)
- SHAP dependency plot vertical spread can look like interaction even for numeric precision artifacts — verify with domain knowledge
- High-cardinality features can dominate SHAP interaction matrix — check normalized by feature variance

## Sources

- `raw/kaggle/shap-interpretability-feature-engineering.md` *(not yet ingested)* — full workflow reference
- [SHAP feature engineering notebook](https://www.kaggle.com/code/wrosinski/shap-feature-importance-with-feature-engineering)
- [SHAP interaction values (XGBoost)](https://shap.readthedocs.io/en/latest/)
- [ShapRFECV (probatus ING)](https://medium.com/ing-blog/open-sourcing-shaprfecv-improved-feature-selection-powered-by-shap-994fe7861560)
- [SHAP for drift detection](https://towardsdatascience.com/shap-for-drift-detection-effective-data-shift-monitoring-c7fb9590adb0/)

## Related

- [[feature-selection-advanced]] — ShapRFECV, BorutaShap, null importance
- [[feature-engineering-tabular]] — what features to engineer
- [[validation-strategy]] — adversarial validation for drift
- [[gradient-boosting-advanced]] — tree models used with SHAP
