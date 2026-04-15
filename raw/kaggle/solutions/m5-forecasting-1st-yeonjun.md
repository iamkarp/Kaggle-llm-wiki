# M5 Forecasting — 1st Place Solution
**Author**: Yeonjun In | **Votes**: 261

---

## Competition
Forecast Walmart daily unit sales for ~30,500 products across 10 stores and 3 states, at 12 hierarchy levels, for 28 days ahead. WRMSSE metric (Weighted Root Mean Squared Scaled Error). Time series, hierarchical structure, ~1,913 days of history.

## Core Approach: LightGBM with Recursive Horizon Features

Single model: LightGBM trained with recursive features — predictions from shorter forecast horizons are used as input features for longer horizons.

### Why Recursive Features?
For a 28-day forecast horizon, direct forecasting (predict all 28 days simultaneously) requires the model to extrapolate far from training data. Recursive forecasting — predict day 1, use it as a feature to predict day 2, etc. — lets each prediction build on recent predicted context.

**Yeonjun's approach**: Rather than fully recursive (predict one day at a time, 28 steps), use horizon-grouped recursion:
- Predict days 1–7 first (near horizon)
- Use day 1–7 predictions as features to predict days 8–14
- Use day 1–14 predictions as features to predict days 15–28

```python
def predict_recursive(model, X_base, horizons=[(1,7), (8,14), (15,28)]):
    all_preds = {}
    
    for h_start, h_end in horizons:
        # Build features including previous horizon predictions
        X_horizon = X_base.copy()
        for prev_start, prev_end in horizons:
            if prev_end < h_start:  # only use earlier predictions
                for d in range(prev_start, prev_end + 1):
                    if d in all_preds:
                        X_horizon[f'pred_day_{d}'] = all_preds[d]
        
        # Predict this horizon
        for day in range(h_start, h_end + 1):
            X_day = build_day_features(X_horizon, day)
            all_preds[day] = model.predict(X_day)
    
    return all_preds
```

## Hierarchical Time Series Structure

M5 has a 12-level hierarchy: total → state → store → category → department → item. Sales at each level must be coherent (sum of lower levels = higher level).

### Reconciliation Approach
Rather than training 12 separate models, train a single item-level LightGBM and aggregate predictions up the hierarchy. This bottom-up reconciliation:
- Ensures perfect coherence at all levels
- Avoids the complexity of 12 separate models
- Leverages the rich item-level features (store, dept, category, event effects)

## Feature Engineering

### Lag Features (Core)
```python
lag_days = [7, 14, 21, 28, 35, 42, 49, 56, 364, 371, 378]
for lag in lag_days:
    df[f'sales_lag_{lag}'] = df.groupby('id')['sales'].shift(lag)
```

Lags chosen to capture:
- Weekly seasonality (multiples of 7)
- Year-ago comparison (364, 371, 378 — same weekday last year)

### Rolling Window Statistics
```python
roll_windows = [7, 14, 30, 60, 180]
for w in roll_windows:
    df[f'sales_roll_mean_{w}'] = (
        df.groupby('id')['sales']
        .transform(lambda x: x.shift(1).rolling(w).mean())
    )
    df[f'sales_roll_std_{w}'] = (
        df.groupby('id')['sales']
        .transform(lambda x: x.shift(1).rolling(w).std())
    )
```

### Calendar Features
- Day of week, week of year, month, year
- SNAP (Supplemental Nutrition Assistance Program) event flags by state
- National holidays, sporting events, Black Friday, Christmas proximity

### Price Features
- Current price, price change from last week
- Price relative to item's historical mean
- Price relative to store average for that category

### Categorical Embeddings
Item, store, state, category, department as label-encoded integers. LightGBM handles these natively.

## LightGBM Configuration

```python
lgb_params = {
    'objective': 'tweedie',          # Tweedie distribution for count/sales data
    'tweedie_variance_power': 1.1,   # Between Poisson (1.0) and Gamma (2.0)
    'metric': 'rmse',
    'num_leaves': 2**11 - 1,         # 2047 — large for 30K+ time series
    'min_data_in_leaf': 2**8 - 1,    # 255
    'learning_rate': 0.03,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'n_estimators': 1400,
    'boost_from_average': False,
}
```

**Tweedie objective**: Sales data has many zeros (items out of stock or not purchased). Tweedie distribution handles this better than Gaussian — it's a compound Poisson-Gamma distribution that naturally models zero-inflated counts.

## Key Takeaways
1. Recursive horizon features (predict near horizons first, use as features for far horizons) outperform direct multi-output forecasting
2. Bottom-up hierarchical reconciliation: train at item level, aggregate up
3. Lag features at multiples of 7 (weekly seasonality) and 364/371/378 (year-ago same weekday) are essential
4. Tweedie objective handles zero-inflated sales better than RMSE/MAE
5. `num_leaves=2047` — don't be afraid of large leaf counts for 30K+ entities with rich history
6. Rolling window stats at 7/14/30/60/180 days capture short and long-term trend simultaneously
