# Kaggle Past Solutions — Tabular Classics

Source: ndres.me/kaggle-past-solutions catalog
Ingested: 2026-04-16

---

## Rossmann Store Sales — 1st Place (Gert Jacobusse)

**Competition:** Predict daily sales for 1,115 Rossmann drug stores. Tabular time-series.
**Writeup:** http://blog.kaggle.com/2015/12/21/rossmann-store-sales-winners-interview-1st-place-gert/

### Approach
- 20+ XGBoost models ensembled with simple averaging
- Heavy feature engineering focus: store-level statistics, promotional patterns, competition distance/time features
- Time-based validation: held out last 6 weeks as validation set (critical for time-series)
- Log-transform of target variable to handle skewed sales distribution

### Key Techniques
1. **Store-level aggregation features**: mean/median/std of past sales per store, day-of-week effects
2. **Promotional features**: days since/until promotion, promotion duration, interaction with day-of-week
3. **Competition features**: months since competitor opened, distance to competitor
4. **State holidays**: encoded with lead/lag indicators (before/after holiday effects)
5. **XGBoost with RMSPE objective**: custom evaluation metric matching competition metric

### How to Reuse
- For any retail/store forecasting: encode store-level historical patterns as features
- Time-based validation split is essential for time-series competitions
- Log-transform target when sales/counts are right-skewed
- Multiple XGBoost models with different feature subsets > single model with all features

---

## Rossmann Store Sales — 3rd Place Entity Embeddings (Cheng Guo & Felix Berkhahn)

**Code:** https://github.com/entron/entity-embedding-rossmann
**Paper:** "Entity Embeddings of Categorical Variables" (arXiv:1604.06737)

### Approach
Pioneered using **neural network entity embeddings** for categorical variables in tabular data. Trained a simple feedforward NN on the Rossmann task, then extracted the learned embeddings for categorical features as inputs to other models (XGBoost, RF).

### Architecture
```
Categorical inputs → Embedding layers → Concatenate with numeric → Dense(1000) → Dense(500) → Dense(1)
```

### Embedding Dimensions
| Feature | Cardinality | Embedding Dim |
|---------|-------------|---------------|
| Store | 1115 | 10 |
| DayOfWeek | 7 | 6 |
| Year | 3 | 2 |
| Month | 12 | 6 |
| Day | 31 | 10 |
| StateHoliday | 4 | 3 |
| CompetitionMonthsOpen | 25 | 2 |
| Promo2SinceWeek | 25 | 1 |
| State | 12 | 6 |

### Key Insight
The paper showed entity embeddings capture meaningful semantic structure. For example, German states cluster geographically in embedding space — East German states group together despite this never being an explicit feature.

### How to Reuse
- **Rule of thumb**: embedding_dim ≈ min(50, cardinality // 2)
- Train NN end-to-end on target, then extract embeddings as features for tree models
- Works best for high-cardinality categoricals (stores, users, products)
- Standard approach now in fastai tabular, PyTorch tabular pipelines

---

## Crowdflower Search Results Relevance — 1st Place (Chenglong Chen)

**Competition:** Rate the relevance of search results (1-4 scale). Text matching + regression.
**Writeup:** http://blog.kaggle.com/2015/07/27/crowdflower-winners-interview-1st-place-chenglong-chen/
**Code:** https://github.com/ChenglongChen/Kaggle_CrowdFlower

### Approach
- NLP feature engineering pipeline → 35-model median ensemble
- **Key innovation**: distribution-based decoding trick that improved QWK by 0.17

### Feature Engineering Pipeline
1. **Text preprocessing**: stemming, stopword removal, spell correction
2. **Distance features**: cosine similarity, Jaccard, edit distance, TF-IDF between query and result
3. **Counting features**: word overlap count, bigram overlap, unique word ratios
4. **Statistical features**: query length, result length, word count ratios
5. **SVD features**: truncated SVD on TF-IDF matrices for latent semantic similarity

### Distribution-Based Decoding
Instead of rounding regression output to nearest integer (1-4):
1. Compute the target distribution in training set
2. Use percentile-based thresholds that match the training distribution
3. This single trick improved QWK from ~0.50 to ~0.67

### Ensemble Strategy
- 35 models: XGBoost, Ridge, SVR, RandomForest, ExtraTreerees, KNN
- Median ensemble (more robust to outliers than mean for ordinal targets)
- Stacking with 5-fold OOF predictions

### How to Reuse
- For any ordinal regression competition (QWK metric): use distribution-based decoding
- Text matching: compute multiple distance metrics between query/document pairs
- Median ensemble when predictions are on a bounded ordinal scale

---

## Homesite Quote Conversion — 1st Place (KazAnova / Faron / Clobber)

**Competition:** Predict whether a customer will purchase an insurance quote. Binary classification.
**Writeup:** http://blog.kaggle.com/2016/04/08/homesite-quote-conversion-winners-write-up-1st-place-kazanova-faron-clobber/

### Approach
- **StackNet**: KazAnova's 3-level stacking methodology
- 100+ base models at Level 1 → meta-learners at Level 2 → final blender at Level 3
- This competition popularized industrial-scale stacking for Kaggle

### StackNet Architecture
```
Level 1 (100+ models):
  - XGBoost (multiple configs: depth 4-12, lr 0.01-0.3)
  - LightGBM, RandomForest, ExtraTrees
  - Ridge, Logistic Regression (with polynomial features)
  - LibSVM, KNN (multiple K values)
  - Neural networks (1-3 hidden layers)
  
Level 2 (10-15 models):
  - XGBoost on L1 OOF predictions
  - NN on L1 OOF predictions
  - Ridge/Logistic on L1 OOF predictions

Level 3 (final):
  - Simple weighted average or Ridge
```

### Key Protocol: K-Fold OOF
1. Split data into K folds
2. For each base model: train on K-1 folds, predict on held-out fold
3. Concatenate held-out predictions → OOF feature for next level
4. Train Level 2 models on OOF features from Level 1
5. Repeat for Level 3

### How to Reuse
- **StackNet** is available as open-source: https://github.com/kaz-Anova/StackNet
- Key principle: diversity > individual model quality at L1
- Include both strong (XGBoost) and weak (KNN, Ridge) models — weak models add information
- Always use OOF predictions, never in-fold predictions (prevents leakage)
- Level 3 should be simple (weighted average or Ridge) to prevent overfitting

---

## Cross-Cutting Patterns

| Technique | Rossmann | Crowdflower | Homesite |
|-----------|----------|-------------|----------|
| Feature engineering depth | Very high | Very high | Moderate |
| Ensemble size | 20+ | 35 | 100+ |
| Stacking levels | 1 | 2 | 3 |
| Custom metric optimization | RMSPE | QWK decoding | AUC |
| Validation strategy | Time-based | 5-fold | 5-fold |
| Key innovation | Store features | Distribution decoding | StackNet |
