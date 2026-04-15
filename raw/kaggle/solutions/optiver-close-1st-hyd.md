# Optiver Trading at the Close — 1st Place Solution
**Author**: hyd | **Year**: 2024 | **Votes**: 333

---

## Competition
Predict the closing auction price movement for Nasdaq-listed stocks. Regression, MAE metric. Test data arrives in rolling batches during the competition — the model must handle online inference and can be retrained on newly revealed test data.

## Model Ensemble: CatBoost + GRU + Transformer

Final blend weights:
- **CatBoost**: 50% — primary tree model, most stable
- **GRU** (Gated Recurrent Unit): 30% — sequence model for temporal patterns
- **Transformer**: 20% — attention over time steps

Weights chosen by OOF score on training data. CatBoost dominates because it handles the tabular structure well; GRU and Transformer capture sequential patterns that trees miss.

## Online Learning: Retrain Every 12 Days

The competition provided test data incrementally (one day at a time). Rather than using a single frozen model trained on the public training set, hyd retrained every 12 days on accumulated data:

```
Training set:       All historical data available
Day 1–12:           Use frozen model
Day 13:             Retrain on original_train + days_1-12
Day 25:             Retrain on original_train + days_1-24
...
```

**Why 12-day intervals**: Retraining daily is expensive; 12 days balances freshness vs. compute. The model adapts to regime changes in market behavior that occur over the test period.

**Memory-efficient implementation**: Load data day-by-day from HDF5 files rather than loading all data into RAM:
```python
# Day-by-day loading from h5 files
for day in training_days:
    day_data = pd.read_hdf(f'data_{day}.h5', key='df')
    # Process and accumulate features incrementally
    # Avoids memory explosion from loading full history at once
```

## Feature Engineering: 300 Features

Total: ~300 features selected by CatBoost importance from a larger pool.

### seconds_in_bucket Grouping
The key temporal feature: group observations by their position in the 10-second auction buckets. The closing auction has a specific micro-structure — each second within the bucket has characteristic order flow patterns.

```python
# Group features by seconds position in bucket
for seconds in train['seconds_in_bucket'].unique():
    bucket_subset = train[train['seconds_in_bucket'] == seconds]
    # Compute statistics for this specific second
    stats = bucket_subset.groupby('stock_id')[feature_cols].agg(['mean', 'std'])
```

### Rank Features Within Time Buckets
Rather than using raw feature values, rank features within each time bucket. Ranking removes cross-stock scale differences and captures relative positioning:

```python
# Rank features within each time bucket (datetime group)
train['feature_rank'] = train.groupby(['date_id', 'seconds_in_bucket'])['feature'].rank(pct=True)
```

This is analogous to how traders think: not "this stock has bid_size=1000" but "this stock has bid_size in the top 20% for this second."

## Weighted-Mean Post-Processing

Standard post-processing subtracts the simple mean of predictions. hyd discovered that **weighted mean subtraction** works better:

```python
# Weight by stock market cap or liquidity
stock_weights = train.groupby('stock_id')['volume'].mean()
stock_weights = stock_weights / stock_weights.sum()

# Weighted mean of predictions within each time bucket
bucket_weighted_mean = (
    preds.groupby('time_id')
    .apply(lambda g: np.average(g['pred'], weights=stock_weights[g['stock_id']]))
)

# Subtract weighted mean (not simple mean)
preds['adjusted_pred'] = preds['pred'] - preds['time_id'].map(bucket_weighted_mean)
```

**Why weighted**: Market-cap-weighted stocks have more influence on the index price. The adjustment should reflect this weight structure rather than treating all stocks equally.

## 300-Feature Selection via CatBoost Importance

Rather than using all engineered features:
1. Engineer the full feature set (~1000+ features)
2. Train CatBoost once on the full set
3. Rank by CatBoost feature importance
4. Take top 300
5. Retrain all models (CatBoost + GRU + Transformer) on the selected 300

CatBoost importance is used here rather than permutation importance because:
- Fast to obtain from the first training run
- Works well enough as a filter for the downstream models
- Reduces GRU/Transformer input dimensionality significantly

## GRU Architecture
```
Input: sequence of T time steps × 300 features per step
→ GRU(hidden_size=256, num_layers=2, dropout=0.2, bidirectional=False)
→ Take last hidden state
→ Linear(256 → 128) → ReLU → Dropout(0.2)
→ Linear(128 → 1)
```

## Transformer Architecture
```
Input: sequence of T time steps × 300 features per step
→ Linear projection to d_model=256
→ TransformerEncoder(nhead=8, num_layers=4, dim_feedforward=512, dropout=0.1)
→ Mean pool across time steps
→ Linear(256 → 1)
```

## Key Takeaways
1. Online retraining (every 12 days) adapted to market regime shifts in the test period
2. seconds_in_bucket grouping captures microstructure patterns specific to each second of the auction
3. Rank features within time buckets removes cross-stock scale differences
4. Weighted-mean post-processing (by market cap/volume) beats simple mean
5. HDF5 day-by-day loading enables memory-efficient online training on large time series
6. CatBoost 50% + GRU 30% + Transformer 20% — trees dominate, sequential models add complementary signal
