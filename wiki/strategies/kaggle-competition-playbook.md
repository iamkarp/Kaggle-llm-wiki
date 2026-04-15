---
title: "Kaggle Competition Playbook — End-to-End Workflow"
tags: [kaggle, playbook, workflow, eda, feature-engineering, ensembling, validation, tabular]
date: 2026-04-14
source_count: 1
status: active
---

## Summary
Jason's comprehensive playbook for Kaggle tabular ML competitions, covering the full workflow from problem framing through final submission selection. Read this at the start of every new competition. The raw source is `raw/kaggle/kaggle-competition-playbook.md` — refer to it for full code examples.

This page is the navigation hub. Each major section links to a dedicated concept page for depth.

---

## Phase 0: Setup Checklist
Before touching data:
- [ ] Read competition description fully — metric, data description, evaluation notes
- [ ] Check for known leaks in the discussion forum
- [ ] Set up local CV framework that matches LB split
- [ ] Create submission log table (CV / LB / Gap / Notes)
- [ ] Run adversarial validation on raw features

---

## Phase 1: Problem Framing

| Decision | Options |
|----------|---------|
| Task type | Binary classification / Multi-class / Regression / Ranking / Multi-label |
| Metric | LogLoss, AUC, RMSE, MAE, MAP@K, Weighted F1, MSE — know the formula |
| CV strategy | Stratified KFold / TimeSeriesSplit / GroupKFold |
| Target distribution | Check skew, imbalance, outliers |
| Leak check | File metadata, row order, ID patterns, test target leakage |

**Metric drives everything**: LogLoss penalizes confident wrong predictions → calibration matters. MAE → median regression (LightGBM `objective='mape'`). MAP@K → need to optimize ranking, not probabilities.

---

## Phase 2: EDA

- Target distribution: histogram, skew, class imbalance
- Feature distributions: KDE for numeric, value_counts for categorical
- Missingness: `df.isnull().mean()` — pattern (MCAR vs. MAR vs. MNAR?)
- **Adversarial validation**: Train LightGBM to distinguish train vs. test. AUC > 0.6 → distribution shift → drop/transform drifting features
- Target correlation: high correlation is suspicious (possible leak)

→ Full detail: [[../concepts/validation-strategy#adversarial-validation]]

---

## Phase 3: Variable Typing

Assign before engineering:
- **Low-cardinality categorical** (≤50 unique): one-hot encode
- **High-cardinality categorical** (>50 unique): target encoding with OOF weighted blend
- **Ordinal**: integer-map preserving order
- **Datetime**: extract components
- **Free text**: embeddings or TF-IDF or LLM-guided regex

### Target Encoding Formula
```
encoded = (global_mean * 6 + class_mean * sqrt(n_class)) / (6 + sqrt(n_class))
```
Always apply OOF — never fit on full training set and apply to itself.

→ Full detail: [[../concepts/target-encoding]]

---

## Phase 4: Missing Values

| Column type | Treatment |
|------------|-----------|
| Categorical | `NaN → "__MISSING__"` category (preserves missingness signal) |
| Numeric | Median imputation + `feature_was_missing` binary indicator |
| High-missingness (>70%) | Drop or keep with indicator; inspect why it's missing |

**Anti-pattern**: mode imputation for categoricals destroys the missingness signal.

---

## Phase 5: Feature Engineering — 5 Stages

Apply in order:

| Stage | Action |
|-------|--------|
| 1. Hand-crafted domain | Subject-matter features — always highest leverage |
| 2. Stepwise interactions | Top-N features by importance → pairwise products/ratios → keep by CV |
| 3. Date/time | year, month, dow, doy, week, is_weekend, is_holiday, days_since_epoch |
| 4. Target transforms | `log1p` for skewed regression targets; class weights for imbalance |
| 5. Group aggregations | mean/std/min/max/median of numeric per categorical group |

**Anti-pattern**: Exhaustive pairwise interactions — too many, mostly noise. Stepwise CV-based selection only.

→ Full detail: [[../concepts/feature-engineering-tabular]]

---

## Phase 6: Text / Unstructured Data

Three strategies (pick based on context):

| Strategy | When | Effort |
|----------|------|--------|
| **Embeddings + PCA-32** | Semantic similarity; diverse vocabulary | Low |
| **TF-IDF + SVD-64** | High train/test term overlap (>40% Jaccard) | Low |
| **LLM-guided regex** | Semi-structured patterns; need interpretability | High |

**Always reduce embeddings first** — raw 768d/1536d embeddings hurt trees.
**Always check train/test overlap** before TF-IDF.

→ Full detail: [[../concepts/text-feature-engineering]]

---

## Phase 7: Models

Standard trio:
| Model | Key Params | Best For |
|-------|-----------|---------|
| **LightGBM** | `num_leaves`, `feature_fraction`, `bagging_fraction` | Large data, speed |
| **XGBoost** | `max_depth`, `subsample`, `colsample_bytree` | Control + regularization |
| **CatBoost** | `depth`, `cat_features` | High-cardinality categoricals |

Use **Optuna** with `MedianPruner` for hyperparameter search. Always use `early_stopping_rounds` (50–100) — let training length be determined by early stopping, not `n_estimators`.

→ Framework details: [[../entities/xgboost]], [[../entities/lightgbm-catboost]]

---

## Phase 8: Ensembling

**Level 1 — Weighted blend**:
- Rank-normalize predictions (handles scale differences)
- Fourth-root weight by CV improvement over baseline
- Alternative: hill-climbing on OOF predictions

**Level 2 — Stacking**:
- Generate OOF predictions from level-1 models
- Train Ridge (regression) or LogisticRegression (classification) on OOF matrix
- Apply to test: average level-1 test predictions per fold → feed to meta-learner

**Submission selection**:
- Pick 2 final submissions: (1) best CV, (2) best diverse ensemble
- Diversity metric: OOF prediction correlation (want < 0.95 between picks)

→ Full detail: [[../concepts/ensembling-strategies]]

---

## Phase 9: Validation & Submission Management

- Trust CV over LB for model development
- Log every submission: CV / LB / Gap / Notes
- Gap growing → check for leakage or CV/LB split mismatch
- Final pick: best CV + most diverse — don't pick based on single lucky LB score
- **Gate every submission behind review** — don't burn slots on untested code

→ Full detail: [[../concepts/validation-strategy]]

---

## Phase 10: Anti-Patterns Cheatsheet

| Anti-Pattern | Fix |
|-------------|-----|
| Correlation-based interaction picking | Stepwise CV search |
| Mode imputation for categoricals | NaN as own category |
| Target encoding without OOF | Always OOF |
| Raw high-dim embeddings into trees | PCA/SVD to 16–64 dims |
| Submitting to LB to validate features | Use CV; LB for final selection only |
| Tuning on LB score | Trust CV |
| Single-model final submission | Ensemble ≥3 models |
| Ignoring adversarial validation | Run it as first EDA step |

---

## Sources
- [[../../raw/kaggle/kaggle-competition-playbook.md]] — primary source (full code examples in each section)

## Related
- [[../concepts/target-encoding]] — §3 weighted blend formula
- [[../concepts/feature-engineering-tabular]] — §5 five-stage process
- [[../concepts/text-feature-engineering]] — §6 text strategies
- [[../concepts/ensembling-strategies]] — §8 fourth-root blend and stacking
- [[../concepts/validation-strategy]] — §9 CV design and gap tracking
- [[../competitions/march-mania-2026]] — applied example (tabular sports prediction)
- [[../competitions/autopilot-vqa-2026]] — applied example (vision + text pipeline)
