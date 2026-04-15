---
title: "Financial & Trading Competition Patterns"
tags: [financial, time-series, purged-cv, jane-street, optiver, trading, tabular]
date: 2026-04-15
source_count: 5
status: active
---

## Summary

Financial Kaggle competitions require purged walk-forward CV with embargo gaps — standard KFold is always wrong. Neural networks (MLP, GRU, Transformer) dominate over GBDTs in Jane Street-style competitions because temporal dependencies are critical. Online retraining every 7-14 days is standard as financial data has concept drift. WAP and order flow imbalance are the baseline features for order book competitions.

## What It Is

Patterns extracted from Jane Street, Optiver, and G-Research competitions for financial time-series modeling. Fundamental differences from standard tabular competitions: temporal leakage is severe, regime shifts require retraining, and microstructure features dominate raw prices.

## Key Facts / Details

### CV Design: Purged Walk-Forward with Embargo

Standard KFold = always wrong for financial data. Purged CV prevents temporal leakage from overlapping samples:

```python
def purged_cv_splits(dates, n_splits=5, embargo_pct=0.01):
    n = len(dates)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits):
        val_start = fold_size * i
        val_end = fold_size * (i + 1)
        embargo_size = int(fold_size * embargo_pct)
        train_end = val_start - embargo_size
        splits.append((list(range(0, train_end)), list(range(val_start, val_end))))
    return splits
```

Jane Street 2024 winning approach: **200-day purged walk-forward CV**.

### Core Financial Features

**WAP (Weighted Average Price):**
```python
def compute_wap(bid_price, ask_price, bid_size, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
```

**Order flow imbalance:**
```python
ofi = (bid_sz - ask_sz) / (bid_sz + ask_sz)
```

**Rolling z-score normalization (removes regime shifts):**
```python
window = 252  # one trading year
df['feature_norm'] = df.groupby('date')['feature'].transform(
    lambda x: (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8)
)
```

**Realized Volatility:**
```python
def compute_rv(log_returns):
    return np.sqrt(np.sum(log_returns ** 2))

df['log_return'] = np.log(df.wap / df.wap.shift(1))
```

### Model Architecture Choices

| Task | Best Model | Why |
|---|---|---|
| Static tabular financial features | LightGBM | Fastest iteration |
| Order book sequences | GRU/LSTM | Temporal dependencies |
| Multi-horizon forecasting | Transformer | Long-range attention |
| Ensemble final submission | All three | Standard in top solutions |

**Jane Street 2024 ensemble:** 10-model NN (MLP + ResNet + Transformer) averaged.
**Optiver Close ensemble:** CatBoost 40% + GRU 35% + Transformer 25%.

### Online Retraining

Financial data has concept drift — stale models decay:

```python
# Retrain every 12 days (Optiver Close winning strategy)
# Exponential weighting: recent data matters more
sample_weight = np.exp(np.linspace(-1, 0, len(X_train)))
model.fit(X_train, y_train, sample_weight=sample_weight)
```

### t-SNE Time Ordering Trick (Optiver Volatility 1st Place)

```python
from sklearn.manifold import TSNE

# Compute t-SNE embedding of time IDs
tsne = TSNE(n_components=2, random_state=42)
time_embeddings = tsne.fit_transform(time_features)

# Find 360 nearest neighbors in t-SNE space
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=361)
nn.fit(time_embeddings)
distances, indices = nn.kneighbors(time_embeddings)

# Neighbor's RV as features → captures market regime similarity
for k in range(1, 361):
    df[f'rv_nn_{k}'] = df['realized_volatility'].iloc[indices[:, k]].values
```

This was the single most important feature set in the Optiver Volatility 1st place solution.

### G-Research: Fibonacci HMA Windows

```python
fibonacci = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

for period in fibonacci:
    df[f'hma_{period}'] = compute_hma(df['close'], period)
    df[f'return_{period}'] = df['close'].pct_change(period)
    df[f'vol_{period}'] = df['close'].pct_change().rolling(period).std()
```

### Target Denoising (Eigenvalue Clipping)

```python
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)
denoised_covariance = lw.covariance_
```

Removes noise from target correlation matrix using Marchenko-Pastur law.

### Multi-Target Learning (Jane Street 2024)

Train on all 9 responders (R_0 through R_8) simultaneously — even if only a few are scored:

```python
class MultiResponderModel(nn.Module):
    def __init__(self, n_features, n_responders=9):
        super().__init__()
        self.encoder = nn.Sequential(...)
        self.heads = nn.ModuleList([nn.Linear(512, 1) for _ in range(n_responders)])
    
    def forward(self, x):
        features = self.encoder(x)
        return [head(features) for head in self.heads]
```

## When To Use It

These patterns apply whenever:
- Competition involves financial market data
- Target is a price change, volatility, or return
- Data has a time dimension with market regimes

## Gotchas

- Never use standard KFold for financial data — temporal leakage is catastrophic
- Don't train on all-time data and validate on recent data without embargo — overlapping samples cause pseudo-CV
- Financial data distribution shifts quarterly — models trained on 2018 data may fail in 2024 regimes
- Order book features should be computed per time bucket (10 seconds), not per row

## Sources

- [[../raw/kaggle/financial-competition-strategies.md]] — full reference with all code
- [Jane Street Market Prediction 2024](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
- [Optiver Trading at Close 2023](https://www.kaggle.com/competitions/optiver-trading-at-the-close)

## Related

- [[concepts/time-series-cv]] — purged CV, walk-forward, embargo gap details
- [[concepts/time-series-features]] — lag features, rolling stats, Fourier encoding
- [[concepts/online-learning]] — 12-day retraining, HDF5 incremental loading
- [[concepts/multi-target-learning]] — multi-responder training
