# Kaggle Competition Playbook

Jason's end-to-end playbook for structured Kaggle competitions. Covers the full workflow from problem framing through final submission selection.

---

## 1. Problem Framing

Before touching data:

- **Task type**: Classification (binary/multi), regression, ranking, multi-label, ordinal?
- **Metric**: What is the eval metric exactly? Is it differentiable? (LogLoss → tree-friendly. MAP@K → need special handling. Weighted F1 → class balance matters.)
- **CV strategy**: Time-series split (no shuffle!), stratified k-fold, group k-fold? Match the LB split as closely as possible.
- **Leak check**: Look at file metadata, row ordering, ID patterns, test target distributions if leaked. Leaks are fair game on Kaggle.
- **Target distribution**: Skewed? Imbalanced? Log-transform for regression. Stratify for classification.

---

## 2. EDA

- **Target distribution**: Plot histogram. Skew, outliers, zeros, multimodality.
- **Feature distributions**: plot_kde for numeric, value_counts for categorical.
- **Missingness**: `df.isnull().mean()` sorted. MCAR vs. MAR vs. MNAR?
- **Adversarial validation**: Train classifier (LightGBM) to distinguish train vs. test rows. AUC > 0.6 → distribution shift. Important features of the adversarial model are the drifting features — drop or transform them.
- **Correlation analysis**: Heatmap for numerics. But do NOT use correlation to pick interaction features (see Anti-Patterns).
- **Target leakage check**: Any feature that has suspiciously high importance or near-perfect correlation with target in train but not test?

---

## 3. Variable Typing

Assign types before any feature engineering:

| Type | Handling |
|------|----------|
| **Categorical (low-cardinality, ≤50 unique)** | One-hot or label encode |
| **Categorical (high-cardinality, >50 unique)** | Target encoding (see §3.1) |
| **Ordinal** | Map to integers preserving order |
| **Numeric (continuous)** | Keep as-is for trees; scale for linear |
| **Numeric (counts)** | Consider log1p transform |
| **ID / hash** | Drop unless used for leak or target encoding |
| **Datetime** | Extract components (see §5.3) |
| **Free text** | Three strategies (see §6) |

### 3.1 Target Encoding (Weighted Blend)

For high-cardinality categoricals, avoid simple mean encoding (it leaks). Use weighted blend with global prior:

```
encoded = (global_mean * k + class_mean * sqrt(n_class)) / (k + sqrt(n_class))
```

Where:
- `k = 6` — prior strength (controls shrinkage toward global mean for rare categories)
- `class_mean` — mean target value for this category in training data
- `n_class` — number of training rows with this category value
- `global_mean` — overall training target mean

Low-frequency categories (small `n_class`) shrink toward `global_mean`. High-frequency categories (large `n_class`) trust `class_mean`.

**Critical**: Always apply with OOF (out-of-fold). Never fit on the full training set and apply to itself — this is the target encoding leakage anti-pattern.

```python
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
global_mean = train[target].mean()
k = 6

encoded_train = np.zeros(len(train))
for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train)):
    fold_stats = train.iloc[tr_idx].groupby(cat_col)[target].agg(['mean', 'count'])
    def encode_row(row):
        if row[cat_col] in fold_stats.index:
            cm = fold_stats.loc[row[cat_col], 'mean']
            n = fold_stats.loc[row[cat_col], 'count']
        else:
            cm, n = global_mean, 0
        return (global_mean * k + cm * np.sqrt(n)) / (k + np.sqrt(n))
    encoded_train[val_idx] = train.iloc[val_idx].apply(encode_row, axis=1)

# For test: use full training stats
test_stats = train.groupby(cat_col)[target].agg(['mean', 'count'])
# apply same formula to test rows
```

---

## 4. Missing Values

| Type | Strategy |
|------|----------|
| **Categorical** | Treat NaN as its own category (`"__MISSING__"` or `-1`). Never use mode imputation — mode encoding destroys the missingness signal. |
| **Numeric** | Impute with median (robust to outliers). Add `feature_was_missing` binary indicator column. |
| **High-missingness (>70%)** | Consider dropping the feature OR keeping as-is + indicator. |
| **Test-only missingness** | If a feature is missing in test but not train, it may be the most important feature for adversarial validation. |

---

## 5. Feature Engineering — 5 Stages

Apply in order. Each stage builds on the previous.

### Stage 1: Hand-Crafted Domain Features
Features from subject-matter knowledge. In basketball: Elo, Four Factors. In finance: returns, ratios. In NLP: sentence length, punctuation count. **These are always the highest-leverage features.**

### Stage 2: Stepwise Interaction Search
Don't exhaustively generate all pairwise interactions — too many, and most are noise. Instead:

1. Train baseline model, get feature importances.
2. Take top-N features (e.g., top 10).
3. Generate pairwise products and ratios for just those top-N.
4. Add interactions one at a time, keeping only those that improve CV.

**Anti-pattern**: Picking interactions by correlation to target — biased toward leaky or spurious features.

### Stage 3: Date/Time Features
From any datetime column, extract:
- Year, month, day-of-week, day-of-year, week-of-year
- Is weekend, is holiday (use `holidays` library)
- Days since epoch (for trend capture)
- Time since last event (for event-based data)

### Stage 4: Target Transforms
For regression:
- `log1p(target)` for right-skewed targets (remember to `expm1` predictions at inference)
- Box-Cox or Yeo-Johnson for general normalization
- For count targets: Poisson regression objective in LightGBM

For classification with imbalance:
- Don't oversample blindly; try class weights first
- `scale_pos_weight` in XGBoost, `is_unbalance` in LightGBM

### Stage 5: Group Aggregations
For each categorical group variable, compute aggregations of numeric features:

```python
agg_features = train.groupby(group_col)[num_col].agg(['mean', 'std', 'min', 'max', 'median'])
agg_features.columns = [f'{num_col}_{group_col}_{stat}' for stat in ['mean','std','min','max','median']]
train = train.merge(agg_features, on=group_col, how='left')
```

Key aggregations: mean, std, min, max, median, count, skew. The `std` and `skew` within a group often capture heterogeneity that mean alone misses.

---

## 6. Text / Unstructured Data

Three strategies, roughly in order of effort:

### Strategy A: Embeddings (Fastest Path)
Use a pretrained sentence encoder (sentence-transformers, OpenAI embeddings, etc.) to get dense vectors.

**Reduce dimensionality before feeding to trees**: PCA to 16–64 dims. Raw high-dim embeddings (768d, 1536d) hurt tree models — they become noise-dominated features.

```python
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=32, random_state=42)
X_reduced = svd.fit_transform(embedding_matrix)
```

### Strategy B: TF-IDF (Fast + Interpretable)
Good when vocabulary overlap between train and test is high.

**Check train/test term overlap first**:
```python
train_terms = set(vectorizer.fit(train_text).vocabulary_.keys())
test_terms = set(t for doc in test_text for t in doc.split())
overlap = len(train_terms & test_terms) / len(train_terms | test_terms)
print(f"Term overlap: {overlap:.2%}")
```

If overlap < 40%, TF-IDF will produce mostly zero vectors for test — use embeddings instead.

TF-IDF best practices:
- `max_features=50000`, `ngram_range=(1,2)`, `sublinear_tf=True`
- Apply SVD to reduce: 32–128 components
- Use `min_df=2` to remove hapax legomena

### Strategy C: LLM-Guided Regex Extraction (Highest Precision)
Use Claude/GPT to inspect a sample of text and identify extractable patterns. Then write regex to extract structured features at scale.

Example: "Extract: (1) whether text mentions a dollar amount, (2) sentiment (pos/neg/neutral), (3) presence of urgency words."

This is more work but produces highly interpretable features that trees can use efficiently. Best for competition data where text has consistent semi-structured patterns (medical notes, product descriptions, legal docs).

---

## 7. Models

The standard trio for tabular competitions:

| Model | Strength | Key Tuning Params |
|-------|----------|-------------------|
| **LightGBM** | Fastest, best on large data | `num_leaves`, `learning_rate`, `feature_fraction`, `bagging_fraction` |
| **XGBoost** | Most regularization control | `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` |
| **CatBoost** | Best on high-cardinality cats | `iterations`, `depth`, `learning_rate`, `cat_features` |

### Optuna Tuning
Use Optuna for hyperparameter search. Key practices:
- Use CV score as objective, not train score
- Set `n_trials=50–100` for reasonable coverage
- Use `pruner=MedianPruner()` to kill bad trials early
- Always include `early_stopping_rounds` in the inner training loop

```python
import optuna
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 16, 256),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    cv_score = cross_val_lgbm(params, X_train, y_train)
    return cv_score
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
```

### Early Stopping
Always use `early_stopping_rounds` (typically 50–100). This means `n_estimators` is effectively determined by training, not the hyperparameter. Set a high upper bound (e.g., 10000) and let early stopping decide.

---

## 8. Ensembling

### Level 1: Weighted Average (Fourth-Root Blend)

The standard weighted average blends raw scores linearly. For Kaggle, a **rank-based power blend** (fourth-root weighting) often works better:

```python
# Rank each model's predictions (on held-out / test set)
for i, preds in enumerate(model_preds):
    model_preds[i] = preds.rank(pct=True)  # or scipy.stats.rankdata

# Weight by fourth root of CV score improvement over baseline
# If model i has CV = 0.85, baseline = 0.80: weight_i = (0.85 - 0.80)^(1/4)
weights = [(cv_i - baseline) ** 0.25 for cv_i in cv_scores]
weights = np.array(weights) / sum(weights)

ensemble = sum(w * p for w, p in zip(weights, model_preds))
```

**Why fourth-root**: Compresses weight differences — a model 2x better than another gets ~1.19x the weight, not 2x. Prevents one dominant model from drowning out diversity.

### Level 2: Stacking
Train a meta-learner on out-of-fold level-1 predictions:

```python
# Level 1: generate OOF predictions from each model
oof_preds = np.column_stack([model.oof_predictions for model in level1_models])
test_preds = np.column_stack([model.test_predictions for model in level1_models])

# Level 2: fit meta-learner
from sklearn.linear_model import Ridge, LogisticRegression
meta = Ridge(alpha=1.0)  # or LogisticRegression for classification
meta.fit(oof_preds, y_train)
final_preds = meta.predict(test_preds)
```

Ridge at level 2: simple, doesn't overfit, interpretable (meta-weights tell you model quality). Add raw features to level-2 input only if stacking CV improvement stalls.

### Submission Selection
Don't just submit best CV. Pick 2 submissions:
1. **Best CV** (safest)
2. **Best CV among diverse models** — ensure models disagree on edge cases

Diversity metric: correlation of OOF predictions. If two models correlate > 0.98, they're essentially the same. Prefer lower-correlated ensemble members.

---

## 9. Validation Strategy

### CV Design Principles
- Match train/test split structure exactly (time, groups, stratification)
- Use at least 5 folds; for small datasets, 10-fold or LOO
- Never shuffle time-series data
- For group data: `GroupKFold` — all rows of a group must be in the same fold

### Tracking the CV-to-LB Gap
Log every submission:
```
| Model | CV Score | LB Score | Gap | Notes |
|-------|----------|----------|-----|-------|
| LGB baseline | 0.8420 | 0.8391 | 0.0029 | overfitting signal |
```

If gap grows → overfitting to CV (check for leakage in features). If gap shrinks or inverts → distribution shift (run adversarial validation again).

**Rule**: When CV and LB disagree, trust CV for model development. Trust LB only for final submission selection (and only if you have enough submissions to detect signal vs. noise).

### Final Submission Selection
- Submit ~5 diverse models before the deadline
- Pick 2 final submissions: (1) best CV, (2) best ensemble of diverse models
- Avoid picking based on single LB submission — high variance with few submissions

---

## 10. Anti-Patterns

| Anti-Pattern | Why It's Bad | What to Do Instead |
|-------------|-------------|-------------------|
| Correlation-based interaction picking | Biased toward spurious/leaky features | Stepwise CV-based interaction search |
| Mode imputation for categoricals | Destroys missingness signal | Treat NaN as own category (`"__MISSING__"`) |
| Target encoding without OOF | Leaks target into features | Always use K-fold OOF target encoding |
| Raw high-dim embeddings into trees | Too many noisy features; tree splits waste capacity | PCA/SVD to 16–64 dims first |
| Submitting to LB to validate features | Wastes submission slots; high variance signal | Use local CV for feature validation; LB only for final selection |
| Tuning on LB score | Overfits to LB split | Trust CV; use LB as sanity check only |
| Single-model final submission | Leaves ensemble gains on table | Always ensemble ≥ 3 models |
| Ignoring adversarial validation | Misses distribution shift that kills LB score | Run adv val as first EDA step on any new competition |
