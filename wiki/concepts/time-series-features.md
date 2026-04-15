---
title: "Time Series Feature Engineering — Lags, Rolling Stats, Fourier"
tags: [time-series, feature-engineering, lags, rolling-stats, fourier, forecasting, kaggle]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is

Feature engineering specific to time series tabular data. Rolling statistics, lag features, and cyclical encodings are the distinguishing features of top-2 solutions in multiple retail forecasting competitions (per arXiv analysis of Kaggle forecasting competitions). This is the single highest-leverage area in time series competitions.

## When To Use It

Any competition where the training data has a temporal dimension (timestamp, date, sequence order). Also applicable to any time-ordered tabular data even without explicit forecasting (financial, clinical, activity data).

## Lag Features

Replace a raw feature value at time t with its value at time t-k (k periods ago).

```python
import pandas as pd

# Basic lags at meaningful intervals
for lag in [1, 7, 14, 28, 56, 364]:
    df[f'sales_lag_{lag}'] = df.groupby('item_id')['sales'].shift(lag)

# CRITICAL: Always .shift() before any windowing operation
# This ensures the current time step's value is NEVER included in the feature
```

**Domain-relevant intervals:**
- Daily data: lag-1, lag-7 (weekly), lag-28 (monthly), lag-364 (annual)
- Hourly data: lag-1, lag-24 (daily), lag-168 (weekly)
- Financial/trading: lag-1, lag-5 (weekly), lag-21 (monthly)

**Entity-grouped lags (M5 approach):**
```python
# Group by every entity combination
for group_col in ['store_id', 'item_id', 'store_class', 'store_dept',
                  ('store_id', 'class'), ('store_id', 'dept')]:
    for lag in [7, 28]:
        col_name = f'lag_{lag}_by_{group_col}'
        df[col_name] = df.groupby(group_col)['sales'].shift(lag)
```

M5 (Walmart) 1st place used lags grouped by store, item, store-class, store-department, and ALL pairwise combinations.

## Rolling Window Statistics

Compute statistics over a trailing window. **Always shift first.**

```python
# Standard rolling stats
for window in [7, 14, 28, 56, 90, 180]:
    shifted = df.groupby('item_id')['sales'].shift(1)  # shift first!
    df[f'rolling_mean_{window}'] = shifted.rolling(window).mean()
    df[f'rolling_std_{window}']  = shifted.rolling(window).std()
    df[f'rolling_q10_{window}']  = shifted.rolling(window).quantile(0.1)
    df[f'rolling_q90_{window}']  = shifted.rolling(window).quantile(0.9)

# Exponentially weighted moving average (recent trends with decay)
for alpha in [0.3, 0.5, 0.7]:
    shifted = df.groupby('item_id')['sales'].shift(1)
    df[f'ewma_{alpha}'] = shifted.ewm(alpha=alpha).mean()
```

**Key insight:** Rolling quantiles (10th/90th percentile) encode volatility asymmetry — more information than std alone.

**EWMA advantage:** Automatically down-weights old observations, captures recent trend changes without manual window selection.

## Fourier / Cyclical Features

Encode periodic calendar features as sine/cosine pairs to avoid ordinality artifacts.

```python
import numpy as np

# Day of week (period = 7)
df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Month (period = 12)
df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)

# Week of year (period = 52)
df['sin_woy'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
df['cos_woy'] = np.cos(2 * np.pi * df['week_of_year'] / 52)

# Multiple Fourier harmonics for complex seasonality
for k in [1, 2, 3, 4]:
    df[f'sin_dow_h{k}'] = np.sin(2 * np.pi * k * df['day_of_week'] / 7)
    df[f'cos_dow_h{k}'] = np.cos(2 * np.pi * k * df['day_of_week'] / 7)
```

**Why sine/cosine over ordinal encoding:** An ordinal encoder treats day 6 (Saturday) and day 0 (Sunday) as maximally far apart; sine/cosine correctly treats them as adjacent in the weekly cycle.

**Multiple harmonics:** For non-sinusoidal seasonality (e.g., retail sales with spikes at certain days), adding higher harmonics (k=2,3,4) captures the shape better.

## Calendar and Exogenous Features

```python
# Holiday indicators
import holidays
us_holidays = holidays.US()
df['is_holiday'] = df['date'].apply(lambda d: 1 if d in us_holidays else 0)
df['days_to_next_holiday'] = ...  # distance features
df['days_since_last_holiday'] = ...

# Event flags (from domain knowledge)
df['is_black_friday'] = (df['month'] == 11) & (df['day_of_week'] == 4) & (df['week_of_month'] == 4)
df['is_super_bowl_week'] = ...

# For retail: price, promotions, markdowns
df['price_ratio_to_category_avg'] = df['price'] / df.groupby('category')['price'].transform('mean')
df['is_promoted'] = (df['markdown_event'] > 0).astype(int)
```

## One-Model-Per-Horizon Strategy

For GBDT time series: train a separate model for each forecast horizon instead of a single multi-output model.

```python
horizons = [1, 7, 14, 28]
models = {}
for h in horizons:
    # Target: sales at time t+h
    df[f'target_h{h}'] = df.groupby('item_id')['sales'].shift(-h)
    X_train_h = df.dropna(subset=[f'target_h{h}'])
    
    model_h = lgb.LGBMRegressor(**params)
    model_h.fit(X_train_h[features], X_train_h[f'target_h{h}'])
    models[h] = model_h
```

**Why:** Each horizon has different predictive features. The 1-day ahead forecast cares most about yesterday; the 28-day ahead forecast needs longer-lag context. A single model must compromise.

## Leakage Risks

| Pattern | Risk Level | Fix |
|---------|-----------|-----|
| Rolling window without `.shift(1)` | CRITICAL | Always shift before windowing |
| Cumulative stats including current row | HIGH | `expanding().sum().shift(1)` |
| Target encoding without OOF | HIGH | Always use within-fold TE |
| Future event flags (holiday next week) | LOW (intentional) | OK to include for known future dates |

## Hyperparameters / Choices

- **Lag intervals:** Start with domain-obvious ones (weekly, monthly, annual for daily data). Then add half-periods and double-periods.
- **Rolling window sizes:** 7, 14, 28, 90, 180 covers most weekly/monthly/quarterly patterns.
- **EWMA alpha:** 0.3 (slow decay) vs 0.7 (fast decay). Try both; ensemble provides diversity.

## Gotchas

- **NaN explosion:** Lags and rolling stats create NaNs at the beginning of each entity's series. Fill with entity mean, global mean, or forward-fill depending on the domain.
- **Memory:** On wide entity × feature combinations, rolling stats can generate millions of columns. Use feature importance to prune after generation.
- **Global vs local models:** Build rolling stats globally (all entities), not per-entity (too noisy for entities with sparse history).

## In Jason's Work
Not yet applied (March Mania uses season-level sports data, not daily time series). Relevant for the M5-style or NFP forex trading context. The NFP straddle strategy uses lagged news event data.

## Sources
- [[../../raw/kaggle/timeseries-nlp-techniques.md]] — comprehensive time series techniques reference
- [[../../raw/kaggle/solutions/m5-forecasting-1st-yeonjun.md]] — M5 lag feature blueprint with 220 LightGBM models

## Related
- [[../concepts/time-series-cv]] — cross-validation strategy for time series
- [[../concepts/feature-engineering-tabular]] — general tabular FE (lags extend this)
- [[../entities/lightgbm-catboost]] — GBDT parameters for time series
- [[../strategies/nfp-straddle-forex]] — live trading application
