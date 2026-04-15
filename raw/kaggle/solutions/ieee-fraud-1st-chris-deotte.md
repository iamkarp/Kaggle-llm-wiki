# IEEE-CIS Fraud Detection — 1st Place Solution
**Author**: Chris Deotte | **Votes**: 396

---

## Competition
Detect fraudulent transactions in credit card data. Binary classification, AUC metric. Train ~590K rows, test ~506K rows. Time-split (train months 1–6, test months 7–12+). Key challenge: severe distribution shift between train and test.

## Core Insight: Classify Clients, Not Transactions

**The fundamental reframe**: Fraud is a property of a credit card / client, not an individual transaction. A fraudulent card will have many fraudulent transactions; a legitimate card will have none (or very few).

**Implication**: If you can identify which card a transaction belongs to, you can aggregate across all transactions for that card → much stronger signal.

This led to the central engineering task: **UID discovery** — finding a unique identifier for each credit card.

## UID Discovery

No explicit card identifier was in the data. Chris discovered the UID through two methods:

### Method 1: Script-Based UID
By analyzing columns `card1`, `card2`, `card3`, `card4`, `card5`, `card6`, `addr1`, `addr2`:
```python
# Combine card features to create a pseudo-UID
train['uid'] = train['card1'].astype(str) + '_' + train['card2'].astype(str) + \
               '_' + train['card3'].astype(str) + '_' + train['card4'].astype(str) + \
               '_' + train['addr1'].astype(str) + '_' + train['addr2'].astype(str)
# This correctly groups transactions belonging to the same card in 70-80% of cases
```

### Method 2: ML-Based UID Refinement
Train a model to predict whether two transactions come from the same card. Features: delta of transaction time, delta of amounts, match on card/addr columns. Output: a more refined grouping.

The UID was then used as a group key for:
- GroupKFold validation (keep all transactions of one card together)
- Client-level aggregation features

## Time Consistency Feature Selection

**Key insight**: The test set is 6+ months after training. Features that are "temporally consistent" (stable over time) are more reliable than features that happen to correlate with fraud only during the training period.

**Method**:
1. Split training data: train on months 1–5, evaluate on month 6 (a "time gap" hold-out)
2. For each feature, train a model on months 1–5 and measure AUC on month 6
3. Features where adding them hurts month-6 AUC (even if they help overall CV) are dropped — they overfit to temporal patterns in training

This is a form of **temporal feature selection**: select features that are stable across time, not just predictive in the training window.

```python
# Example: compare feature importance stability
# Train on first 5 months, eval on month 6
lgb_early = train_lgb(X_train[months_1_5], y[months_1_5])
auc_late = eval_auc(lgb_early, X_train[month_6], y[month_6])
# Features with high importance in months_1_5 but low auc_late → drop
```

## PCA on V-Column Groups (NaN Pattern Grouping)

The dataset had 339 V-columns (V1–V339) with complex NaN patterns. Rather than imputing blindly:

1. **Group by NaN pattern**: columns with the same missingness pattern are related (they come from the same source system or event type)
2. **PCA within each group**: reduce each NaN-pattern group to its principal components
3. This reduces 339 V-columns to ~50 PCA components while preserving most variance

```python
# Group columns by their NaN pattern (mask)
def get_nan_pattern(df, col):
    return tuple(df[col].isna().astype(int).tolist())

nan_patterns = {}
for col in v_cols:
    pattern = get_nan_pattern(train, col)
    nan_patterns.setdefault(pattern, []).append(col)

# PCA within each group
from sklearn.decomposition import PCA
for pattern, cols in nan_patterns.items():
    pca = PCA(n_components=min(len(cols), 5))
    train_pca = pca.fit_transform(train[cols].fillna(0))
    # add as new features
```

## Multiple Validation Strategies

Chris used several CV approaches to combat the time-split challenge:

| Strategy | Description | Purpose |
|----------|-------------|---------|
| **Standard StratifiedKFold** | 5-fold ignoring time | Baseline; overestimates performance |
| **Skip-month holdout** | Train months 1–4, validate month 6 (skip 5) | Simulates real time gap |
| **GroupKFold by UID** | Keep all transactions of one card together | Prevents card-level leakage |
| **Month-based split** | Sequential time splits | Measures temporal degradation |

The skip-month holdout (train 1–4, skip 5, validate 6) was most predictive of private LB because it replicated the distribution shift pattern.

## Feature Engineering Highlights
- **Frequency encoding**: How often does each value appear? Rare values often indicate fraudulent behavior
- **Normalized transaction amount**: Amount relative to mean/std for that card, day-of-week, etc.
- **Lag features**: Time since last transaction on this card, time since last transaction at this merchant
- **V-column PCA**: ~50 PCA components replacing 339 raw V-columns

## Key Takeaways
1. Reframing the problem (classify cards, not transactions) was the core insight
2. UID discovery via feature combination is a powerful technique for entity resolution
3. Time consistency feature selection prevents overfitting to temporal patterns in training
4. NaN patterns contain information — group and PCA within patterns rather than imputing blindly
5. Multiple validation strategies (skip-month, GroupKFold) are needed when distribution shift is severe
6. Always build validation to match the structure of the actual train/test split
