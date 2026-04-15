# Optiver Realized Volatility Prediction — 1st Place Solution
**Author**: nyanp | **Votes**: 488

---

## Competition
Predict short-term realized volatility of stocks from order book and trade data. 10-minute windows, 112 stocks, time-series data with multiple `time_id` buckets per stock. RMSPE metric (Root Mean Squared Percentage Error).

## The Time-ID Discovery (Reverse Engineering the LB)

**Key breakthrough**: The test set `time_id` values appeared random but were actually drawn from a specific ordered sequence. nyanp reverse-engineered the true ordering.

**Method**:
1. Construct a price matrix: rows = stocks, columns = time_id buckets
2. Run t-SNE on this matrix treating time_ids as points in stock-price space
3. The t-SNE embedding revealed a 1-dimensional manifold — time_ids had a natural ordering
4. Recover that ordering: this gives the true temporal sequence of the test time_ids

**Why this mattered**: Knowing the true time ordering enables:
- Better time-series features (look-back windows)
- Correct temporal cross-validation (prevent future leakage)
- Time-distance features to known anchor points

## Nearest Neighbor Aggregation Features

**360 of 600 total features** were nearest-neighbor (NN) aggregation features. These provided the largest single boost: **0.21 → 0.19 RMSPE**.

### What Are NN Aggregation Features?
For each (stock, time_id) observation, find the K most "similar" other observations and aggregate their target values (or engineered features) as new features.

**Similarity metric**: Euclidean distance in feature space (order book statistics, trade statistics, etc.)

```python
from sklearn.neighbors import NearestNeighbors

# Build feature matrix for all training observations
X_all = np.array(train_features)  # shape: (n_obs, n_base_features)

# Fit NN index
nn = NearestNeighbors(n_neighbors=50, metric='euclidean', algorithm='ball_tree')
nn.fit(X_all)

# For each observation, get K nearest neighbors
distances, indices = nn.kneighbors(X_all)

# Aggregate target values of neighbors
nn_mean_target = train_target[indices].mean(axis=1)
nn_std_target = train_target[indices].std(axis=1)
nn_min_target = train_target[indices].min(axis=1)

# These become new features: effectively "what volatility do similar situations produce?"
```

### Why NN Features Work Here
- Volatility is locally consistent: similar order book states tend to produce similar realized volatility
- NN aggregation creates a non-parametric "neighborhood" estimate
- Complements tree models which struggle with smooth local interpolation
- 360 NN features captured cross-stock, cross-time patterns that raw features missed

### Stock-Level NN (Cross-Stock Features)
Also compute NN aggregations within the same stock across different time periods, and across stocks at the same time period. The cross-stock features capture sector/market-wide volatility regimes.

## Adversarial Validation for Covariate Shift

Standard adversarial validation: train a classifier to distinguish train vs. test rows.

**nyanp's use**: Identified specific stocks and time periods with distribution shift. Used this to:
1. Remove high-shift training samples from CV (more representative hold-out)
2. Weight training samples inversely proportional to how "train-like" they are (higher weight to samples similar to test)

## Model Stack
| Model | Role | Notes |
|-------|------|-------|
| LightGBM | Primary tree model | All 600 features |
| MLP | Neural network | RankGauss inputs, bottleneck features |
| 1D-CNN | Sequence model | Raw order book/trade sequences as input |

**Blend**: Weighted average, weights tuned on OOF RMSPE.

## Feature Engineering
- **Order book statistics**: bid/ask spread, weighted mid-price, order imbalance, depth decay
- **Trade statistics**: price momentum, volume-weighted average price (VWAP), trade frequency
- **Realized volatility at different window sizes**: 1min, 2min, 5min (multi-scale)
- **Stock-level aggregations**: mean/std of base features across time_ids for each stock
- **Time-level aggregations**: mean/std of base features across stocks for each time_id
- **360 NN aggregation features** (the breakthrough)

## Key Takeaways
1. Reverse-engineering latent structure in test data (t-SNE ordering) is legitimate and powerful
2. Nearest-neighbor aggregation features are high-value for time-series prediction tasks
3. Adversarial validation can be used for sample weighting, not just feature dropping
4. Multi-scale features (1min/2min/5min volatility) capture different market dynamics
5. 1D-CNN on raw sequences complements tree + MLP models well
