---
title: "Time Series Cross-Validation — Walk-Forward, Purged CV, Gap Strategy"
tags: [time-series, cross-validation, walk-forward, purged-cv, embargo, financial, forecasting, kaggle]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is

Cross-validation strategies specifically designed for time-ordered data. Standard k-fold is **incorrect** for time series because it creates future leakage (future data appears in training folds). Using wrong CV for a time-series competition is one of the most common fatal errors.

## When To Use It

Any dataset with a temporal dimension where the test set is in the future relative to training data.

## Walk-Forward (Expanding Window) — Gold Standard

Train on [0, t], validate on [t+1, t+h]. Expand training window each fold. This simulates real forecasting conditions.

```python
from sklearn.model_selection import TimeSeriesSplit

# Sklearn implementation
tss = TimeSeriesSplit(n_splits=5)
for tr_idx, val_idx in tss.split(df):
    X_tr, X_val = df.iloc[tr_idx], df.iloc[val_idx]
    # train < validation in time — guaranteed

# Custom walk-forward with explicit horizons
def walk_forward_cv(df, date_col, n_splits=5, val_days=28):
    dates = sorted(df[date_col].unique())
    fold_size = len(dates) // (n_splits + 1)
    for i in range(n_splits):
        cutoff = dates[fold_size * (i + 1)]
        end = dates[min(fold_size * (i + 1) + val_days, len(dates) - 1)]
        train_mask = df[date_col] < cutoff
        val_mask = (df[date_col] >= cutoff) & (df[date_col] < end)
        yield df[train_mask], df[val_mask]
```

**Why it's the gold standard:** It forces you to simulate real production conditions. The model never sees the future during validation.

## Sliding Window Validation

Train on a fixed-size N-month window, validate on next M months. Training window slides forward each fold.

```python
def sliding_window_cv(df, date_col, train_months=24, val_months=3):
    all_months = sorted(df[date_col].dt.to_period('M').unique())
    for i in range(len(all_months) - train_months - val_months + 1):
        train_months_range = all_months[i:i + train_months]
        val_months_range = all_months[i + train_months:i + train_months + val_months]
        train_mask = df[date_col].dt.to_period('M').isin(train_months_range)
        val_mask = df[date_col].dt.to_period('M').isin(val_months_range)
        yield df[train_mask], df[val_mask]
```

**When to use over walk-forward:** When old data is less relevant (concept drift, structural breaks, regulatory changes). A 2024 model shouldn't weight 2010 data equally.

## Purged Cross-Validation — Critical for Financial Data

Standard walk-forward still has leakage when training labels temporally overlap with validation labels (e.g., 5-day return at t=100 overlaps with 5-day return at t=98).

**Purging:** Remove training observations whose labels overlap in time with validation labels.

**Embargo period:** After each validation fold, add an embargo period (N periods) where training is also forbidden — because validation-adjacent training rows may be correlated through features.

```python
# Using mlfinlab
from mlfinlab.cross_validation import PurgedKFold

pkcv = PurgedKFold(
    n_splits=5,
    samples_info_sets=event_times,  # series mapping each row to its label end-time
    pct_embargo=0.01  # 1% of total samples as embargo buffer
)
for tr_idx, val_idx in pkcv.split(X, y):
    ...

# Using skfolio (newer)
from skfolio.model_selection import CombinatorialPurgedCV

cpcv = CombinatorialPurgedCV(n_splits=5, n_test_folds=2, purged_ratio=0.05)
```

**Why CPCV (Combinatorial Purged CV) over standard purged:** CPCV uses multiple test fold combinations, giving better statistical estimates of overfitting. Outperforms on Deflated Sharpe Ratio and Probability of Backtest Overfitting metrics.

**Reference:** Advances in Financial Machine Learning by Marcos López de Prado (the definitive source on this technique).

## Key Rule: Match CV to Test Structure

The most important principle:
- If test = future 28 days → each validation fold should validate on 28-day windows
- If test has a gap between public and private LB → build a gap into CV splits
- If test is on entirely new entities → use GroupKFold, not time-split

### When NOT to Use Time-Based CV

**Cross-sectional data with year groups** (e.g., annual stock fundamentals): Each row is a stock-year, not a time step. Expanding window CV (train ≤2019, val 2020) exposes models to regime shifts that dominate validation loss — the model can't learn because validation loss spikes from market regime mismatch, not poor feature learning.

**Use `GroupKFold(groups=year)` instead.** This mixes years across folds, giving each fold a representative sample of market regimes. Result: stable validation loss, meaningful early stopping, models that actually train (1000+ iterations vs 0-15 with expanding window).

This applies whenever: (a) data is cross-sectional by period, (b) the target distribution shifts dramatically between periods, and (c) models use early stopping.

```python
# Competition example: test is 4 weeks after training ends
# Build CV to mimic this exactly:
def competition_cv(df, train_end_date, test_start_date, val_window_days=28):
    gap_days = (test_start_date - train_end_date).days
    # Each fold: train on data before cutoff, skip gap_days, validate on val_window
    ...
```

## Multiple Validation Windows — Anti-Overfit Check

Top competitors validate across 3–5 different time windows and look for consistency:

```python
windows = [
    ('2021-01-01', '2022-01-01', '2022-04-01'),
    ('2022-01-01', '2023-01-01', '2023-04-01'),
    ('2022-07-01', '2023-07-01', '2023-10-01'),
]
scores = []
for train_start, train_end, val_end in windows:
    # Train/validate for each window
    score = evaluate_model(df, train_start, train_end, val_end)
    scores.append(score)

# A solution that only scores well on ONE window is likely overfit
print(f"CV mean: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

## Post-Cutoff CV

From Playground S5E12 1st place: Use data **after the public LB cutoff date** as a separate validation set. This data is "held out" from both training and the public LB, making it an independent validation signal.

```python
# Competition has public LB using rows before cutoff_date
# Use rows AFTER cutoff_date (not in public LB) as your most reliable CV signal
post_cutoff_mask = df['date'] > public_lb_cutoff_date
X_post_cv = df[post_cutoff_mask][features]
y_post_cv = df[post_cutoff_mask][target]
# Evaluate trained models on this set as final selection criterion
```

## Hyperparameters

- **n_splits:** 5 standard, 3 minimum for large datasets where each fold is long
- **val_window:** Match to competition test period length
- **embargo_pct:** 0.01–0.05 for financial data (larger when label overlap period is longer)
- **gap:** Days between train end and validation start (match to test gap)

## Gotchas

| Anti-Pattern | Why | Fix |
|---|---|---|
| Standard KFold on time series | Future data leaks into training folds | Use TimeSeriesSplit |
| No embargo with financial labels | Overlapping returns create serial correlation | Add embargo period |
| Single validation window | May overfit to a specific market regime | Use 3-5 windows |
| Ignoring public/private LB gap | CV doesn't match private evaluation | Build gap into CV splits |
| GroupKFold when temporal is the right split | Misses the key train/test boundary | Use time split for forecasting |

## In Jason's Work
Not explicitly applied (March Mania uses season-level splits which are inherently temporal). Directly relevant for the NFP straddle strategy if backtesting is formalized. For any future financial/retail forecasting competition.

## Sources
- [[../../raw/kaggle/timeseries-nlp-techniques.md]] — comprehensive time series CV reference
- [[../../raw/kaggle/2024-2025-winning-solutions-tabular.md]] — post-cutoff CV from S5E12 1st place
- [Purged CV Kaggle notebook](https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna)
- Advances in Financial Machine Learning (de Prado) — canonical reference

## Related
- [[../concepts/time-series-features]] — what features to build for time series
- [[../concepts/validation-strategy]] — general validation principles
- [[../concepts/online-learning]] — retraining during test phase
- [[../strategies/nfp-straddle-forex]] — live trading application
