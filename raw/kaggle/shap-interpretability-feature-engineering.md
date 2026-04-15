# SHAP and Interpretability as a Kaggle Score Improvement Tool

Compiled from NVIDIA Grandmasters Playbook, SHAP documentation, competition notebooks, research papers. April 2026.

---

## Core Principle

SHAP is not just for explaining models — it's an active feature engineering discovery tool. The NVIDIA Grandmasters Playbook explicitly lists error analysis via SHAP among its 7 battle-tested techniques: "Analyzing model errors can be crucial to get ideas of new features to add."

---

## 1. SHAP Dependency Plots → Nonlinear Feature Transformations

A SHAP dependency (scatter) plot shows feature raw value (x-axis) vs SHAP contribution (y-axis). The **shape of that curve tells you what transformation the model wants**:

| Curve shape | Model is approximating | Action |
|---|---|---|
| Logarithmic curve | `log(x)` | Add `np.log1p(x)` as explicit feature |
| S-shaped (sigmoid-like) | Logit transform | Add `scipy.special.logit(x)` or bounded transform |
| Flat then steep slope | A threshold at the inflection point | Bin feature around that value |
| Large vertical spread | Interaction with another variable | Color variable = the interaction partner → engineer product/ratio |

**Key rule:** Vertical dispersion in a SHAP dependency plot = interaction effect. Zero dispersion = purely additive. When you see dispersion, look at the coloring variable — that's your top interaction pair to explicitly engineer.

```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Dependency plot for feature 'age' (auto-selects best interaction coloring)
shap.dependence_plot("age", shap_values, X_val)
# Or with explicit interaction
shap.dependence_plot("age", shap_values, X_val, interaction_index="income")
```

---

## 2. SHAP Interaction Values as New Features

```python
# Only works with TreeExplainer
shap_interaction_values = explainer.shap_interaction_values(X)
# Returns shape: (n_samples, n_features, n_features)
# shap_interaction_values[i, j, k] = interaction contribution of feature j and k for sample i

# Find top interaction pairs
mean_abs_interactions = np.abs(shap_interaction_values).mean(axis=0)
np.fill_diagonal(mean_abs_interactions, 0)  # zero out self-interactions

# Get top-10 off-diagonal pairs
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

**Measured result:** In a cancer prediction study using SHAP-guided feature engineering:
- Top 15 features selected via SHAP
- Interaction-based features constructed ("chronic severity" etc.)
- SHAP-based feature weighting applied
- Accuracy improved from 0.8794 → 0.8968

---

## 3. Error Analysis with SHAP → Targeted Feature Discovery

```python
# 1. Get OOF predictions + residuals
oof_preds = cross_val_predict(model, X, y, cv=5, method='predict_proba')[:, 1]
residuals = np.abs(y - oof_preds)  # higher = model was more wrong

# 2. Identify high-error samples (top 20%)
high_error_mask = residuals > np.percentile(residuals, 80)
low_error_mask = residuals < np.percentile(residuals, 20)

# 3. Compare SHAP distributions between high/low error groups
shap_values = explainer.shap_values(X)
high_error_shap = np.abs(shap_values[high_error_mask]).mean(axis=0)
low_error_shap = np.abs(shap_values[low_error_mask]).mean(axis=0)

divergence = high_error_shap - low_error_shap
divergence_series = pd.Series(divergence, index=X.columns).sort_values(ascending=False)

# Features with high divergence = where model systematically fails
print("Features driving high-error samples:")
print(divergence_series.head(10))
# → These features need transformed versions, interaction terms, or subgroup-specific features
```

**Chris Deotte's approach:** Train Stage 2 model on Stage 1's residuals as target. Forces second model to focus on what first model got wrong — structured error-targeted feature discovery.

---

## 4. SHAP for Test-Time Distribution Shift Detection

```python
import shap
from scipy import stats

# Train SHAP baseline on training data
explainer = shap.TreeExplainer(model)
shap_train = explainer.shap_values(X_train)
shap_test = explainer.shap_values(X_test)

# Compare SHAP distributions feature by feature
for i, feature in enumerate(X_train.columns):
    ks_stat, p_value = stats.ks_2samp(shap_train[:, i], shap_test[:, i])
    if ks_stat > 0.1:  # significant drift in SHAP space
        print(f"SHAP drift detected: {feature} (KS={ks_stat:.3f}, p={p_value:.3e})")
```

**Why SHAP-space drift > raw feature drift:** A feature with large raw distribution shift but near-zero SHAP values is irrelevant. A feature with small raw shift but large SHAP shift is critical.

**Competition use:** If you detect drift in a high-SHAP feature, you know your model will degrade specifically on that feature. Options: re-engineer it to be more stable, use pseudo-labeling to recalibrate, or down-weight it during inference.

---

## 5. ShapRFECV (ING Bank probatus package)

Recursive feature elimination using SHAP importance with cross-validation.

**Why it outperforms standard RFECV:**
- Standard gain importance is biased toward high-cardinality features
- Permutation importance is unreliable with correlated features
- SHAP correctly distributes credit among correlated features

**Published result:** ShapRFECV eliminated 60/110 features → slight AUC improvement + significant complexity reduction. Both CV Validation and Test AUC improved vs sklearn RFECV baseline.

```python
# pip install probatus
from probatus.feature_elimination import ShapRFECV
from lightgbm import LGBMClassifier

model = LGBMClassifier(n_estimators=100)
shap_rfecv = ShapRFECV(model, step=0.2, cv=5, scoring='roc_auc',
                        n_iter=10, random_state=42)
shap_rfecv.fit_compute(X_train, y_train)
selected = shap_rfecv.get_reduced_features_set(num_features=30)
```

---

## 6. Subgroup Analysis via SHAP Clustering

```python
# Cluster samples by their SHAP profiles (not raw features)
from sklearn.cluster import KMeans

shap_values_array = explainer.shap_values(X)
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(shap_values_array)

# For each cluster, find which features are most extreme
for cluster in range(5):
    cluster_mask = cluster_labels == cluster
    cluster_shap = shap_values_array[cluster_mask].mean(axis=0)
    top_features = pd.Series(cluster_shap, index=X.columns).abs().sort_values(ascending=False).head(5)
    print(f"Cluster {cluster}: {top_features.index.tolist()}")
    # → These subpopulations may need cluster-conditional features
```

---

## Practical Workflow (Competition-Ready)

```
1. Train CV model → OOF predictions + OOF SHAP values

2. SHAP dependency plots for top 20 features → identify:
   - Nonlinear curve shape → add explicit transformation
   - Vertical dispersion → top interaction pair → add product/ratio feature

3. SHAP interaction matrix → extract top 10 off-diagonal pairs → engineer interactions

4. High-residual sample analysis:
   - Filter to top 20% error samples
   - Compare SHAP vs. low-error samples
   - Divergent features → investigate subgroup → engineer targeted feature

5. SHAP beeswarm/summary → prune bottom 20% by mean |SHAP| → retrain

6. If test data available → SHAP-space KS test for drift:
   - Flag drifted features → consider pseudo-labeling or feature recalibration
```

---

Sources:
- NVIDIA Grandmasters Playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- SHAP feature engineering notebook: https://www.kaggle.com/code/wrosinski/shap-feature-importance-with-feature-engineering
- SHAP interaction example: https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Basic%20SHAP%20Interaction%20Value%20Example%20in%20XGBoost.html
- ShapRFECV (probatus): https://medium.com/ing-blog/open-sourcing-shaprfecv-improved-feature-selection-powered-by-shap-994fe7861560
- SHAP for drift detection: https://towardsdatascience.com/shap-for-drift-detection-effective-data-shift-monitoring-c7fb9590adb0/
- Christoph Molnar IML book: https://christophm.github.io/interpretable-ml-book/shap.html
