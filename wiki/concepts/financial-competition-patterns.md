---
id: concept:financial-competition-patterns
type: concept
title: Financial & Trading Competition Patterns
slug: financial-competition-patterns
aliases: []
tags:
- financial
- time-series
- purged-cv
- jane-street
- optiver
- trading
- tabular
status: active
date: 2026-04-15
source_count: 5
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

### Cross-Sectional Stock Return Prediction (Non-Temporal Financial Data)

Not all financial competitions are time-series. Cross-sectional stock return prediction (e.g., predict 1-year return from fundamentals) has distinct patterns:

**CV: GroupKFold, NOT Expanding Window**

When training data spans multiple years of annual cross-sections, expanding window CV (train ≤2019, val 2020; train ≤2020, val 2021) creates regime-shift problems. Market returns vary wildly across years (2019: +4%, 2020: +74%, 2021: -11%), so validation loss spikes from regime mismatch — not model quality. This triggers premature early stopping (0-15 iterations).

```python
from sklearn.model_selection import GroupKFold

# Cross-sectional: group by start_year, not time-ordered
gkf = GroupKFold(n_splits=4)
for tr_idx, val_idx in gkf.split(X, y, groups=df['start_year']):
    # Each fold mixes years → stable validation loss
    ...
```

**GroupKFold mixes years across folds**, so each fold sees representative market regimes in both train and validation. This gives stable validation loss and meaningful early stopping signals.

**Target Clipping for Fat-Tailed Returns**

Stock returns have extreme outliers (10,000%+ returns). RMSE is dominated by these tails. Clip targets before training:

```python
# Multiple clip levels as separate target transforms — diversity source
clip_1_99 = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))   # [-80, 300]
clip_5_95 = np.clip(y, np.percentile(y, 5), np.percentile(y, 95))   # [-56, 118]
clip_fixed = np.clip(y, -95, 500)                                    # domain knowledge
```

Train separate models on each clip level → different bias-variance tradeoffs → ensemble diversity.

**Prediction Centering Matters for RMSE**

RMSE penalizes bias heavily. If market average return is ~15% but model predicts mean ~2%, the constant 15% prediction beats the ML model. Always check:

```python
print(f"Prediction mean: {preds.mean():.1f}, Target mean: {y.mean():.1f}")
# If prediction mean << target mean, model is under-centered
# Likely cause: log transforms or heavy regularization compressing toward 0
```

**Cross-Sectional Rank Features (Regime-Invariant)**

Rank percentiles within each year remove regime dependence:

```python
for col in numerical_features:
    df[f'{col}_rank'] = df.groupby('start_year')[col].rank(pct=True)
```

These features are invariant to market-level shifts — a company's *relative* P/E ratio matters more than its absolute value across different market regimes.

**Sector-Relative Features**

```python
for col in fundamentals:
    sector_mean = df.groupby(['start_year', 'sector'])[col].transform('mean')
    sector_std = df.groupby(['start_year', 'sector'])[col].transform('std')
    df[f'{col}_sector_z'] = (df[col] - sector_mean) / (sector_std + 1e-8)
```

**Signed Log Transform for Dollar-Scale Features**

Revenue, market cap, total assets vary across orders of magnitude. Signed log compresses while preserving sign:

```python
def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))
```

**Warning:** Don't use log transforms on the *target* (returns) — it compresses predictions too much. Models trained on `sign(r) * log1p(|r|)` targets tend to predict mean ≈ 0, killing RMSE performance. Use log on features, clip on targets.

## Gotchas

- Never use standard KFold for financial data — temporal leakage is catastrophic
- Don't train on all-time data and validate on recent data without embargo — overlapping samples cause pseudo-CV
- Financial data distribution shifts quarterly — models trained on 2018 data may fail in 2024 regimes
- Order book features should be computed per time bucket (10 seconds), not per row
- **Cross-sectional ≠ time-series**: When each row is a stock-year (not a time step), GroupKFold beats expanding window CV. Expanding window + early stopping = model death when year-to-year regime shifts are large
- **Log-transforming fat-tailed targets kills RMSE**: Log compresses predictions toward 0. For targets with std=138 and mean=18, raw clipped targets outperform log-transformed targets
- **CatBoost Huber delta must match target scale**: delta=1.0 is catastrophic when target std=138 — all residuals exceed delta, gradient becomes constant, model stops at 0 iterations. Use delta ∝ target_std (e.g., delta=50-100)

## Sources

- `raw/kaggle/financial-competition-strategies.md` — full reference with all code
- [Jane Street Market Prediction 2024](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)
- [Optiver Realized Volatility Prediction](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction)
- [Optiver Trading at Close 2023](https://www.kaggle.com/competitions/optiver-trading-at-the-close)
- [[../../raw/kaggle/solutions/missing-batch-finance-tabular.md]] — Two Sigma Connect (546 votes), Allstate Claims (482 votes), Winton Stock Market, Zillow Prize, KKBox Churn, Criteo CTR, Mitsui Commodity
- [[../../raw/kaggle/solutions/missing-batch-timeseries-signals.md]] — The Winton Stock Market Challenge, Two Sigma Financial News

## Related

- [[time-series-cv]] — purged CV, walk-forward, embargo gap details
- [[time-series-features]] — lag features, rolling stats, Fourier encoding
- [[online-learning]] — 12-day retraining, HDF5 incremental loading
- [[multi-target-learning]] — multi-responder training

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[concepts/time-series-features|Time Series Feature Engineering — Lags, Rolling Stats, Fourier]]
- _applied_in_ → [[competitions/stock-return-prediction|Predict 1-Year US Stock Returns from Fundamentals]]
- _cites_ → `source:missing-batch-finance-tabular` (Kaggle Solutions — Missing Batch — Finance & Tabular Classics)
- _cites_ → `source:missing-batch-timeseries-signals` (Kaggle Solutions — Missing Batch — Time-Series, Signals & EEG/Sensor)
- _works_with_ → [[concepts/multi-target-learning|Multi-Target & Auxiliary Learning]]
- _related_to_ → [[concepts/online-learning|Online Learning — Retraining During Test, Memory Management, Incremental Adaptation]]
- _related_to_ → [[concepts/time-series-cv|Time Series Cross-Validation — Walk-Forward, Purged CV, Gap Strategy]]

**Incoming:**
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
