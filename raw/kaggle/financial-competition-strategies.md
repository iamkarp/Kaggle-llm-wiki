# Financial & Trading Kaggle Competition Strategies

Compiled from Jane Street, Optiver, G-Research competition writeups. April 2026.

---

## Jane Street Market Prediction 2024 — 1st Place Patterns

**Setup:**
- Predict 9 responders (R_0 through R_8) simultaneously
- ~6M rows of financial time-series data
- Metric: Weighted R² across all responders

**CV Design — 200-Day Purged Walk-Forward:**
```python
# Purged CV with embargo for financial data
from sklearn.model_selection import TimeSeriesSplit

def purged_cv_splits(dates, n_splits=5, embargo_pct=0.01):
    """
    Returns indices for purged walk-forward CV.
    embargo_pct: fraction of training period excluded before validation
    """
    n = len(dates)
    fold_size = n // n_splits
    splits = []
    
    for i in range(n_splits):
        val_start = fold_size * i
        val_end = fold_size * (i + 1)
        
        # Embargo: exclude period just before validation
        embargo_size = int(fold_size * embargo_pct)
        train_end = val_start - embargo_size
        
        train_idx = list(range(0, train_end))
        val_idx = list(range(val_start, val_end))
        splits.append((train_idx, val_idx))
    
    return splits
```

**Key techniques:**
1. **200-day purged walk-forward CV** — not standard KFold. Each fold uses 200 days gap to prevent temporal leakage
2. **10-model NN ensemble** — diverse architectures (MLP, ResNet, Transformer) averaged
3. **Z-score rolling normalization** — center features per day to remove regime shifts:
   ```python
   # Rolling z-score normalization
   window = 252  # one trading year
   df['feature_norm'] = df.groupby('date')['feature'].transform(
       lambda x: (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-8)
   )
   ```
4. **Multi-responder auxiliary training** — train on all 9 responders jointly, even if only a few are scored

**Observed:** Winning solutions used neural networks (not GBDTs) — financial time series has temporal dependencies that GBDTs miss.

---

## Optiver Trading at Close 2023

**Task:** Predict 60-second price movements at market close.

**Features that won:**
- **WAP (Weighted Average Price):** `(bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)`
- **Order flow imbalance:** `(bid_sz - ask_sz) / (bid_sz + ask_sz)`
- **Price volatility:** Realized variance in last N seconds
- **Volume clock features:** Encode time as proportion of daily volume traded

**Architecture:** CatBoost + GRU + Transformer ensemble
```python
# GRU for sequential order book data
class OrderBookGRU(nn.Module):
    def __init__(self, input_size=20, hidden_size=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):  # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])  # last hidden state
```

**Retraining strategy:** Top solutions retrained every 12 days on the most recent data — financial data has regime shifts, so stale models decay.

**Ensemble weights:** CatBoost 40% + GRU 35% + Transformer 25% (optimized via OOF hill climbing).

---

## Optiver Realized Volatility Prediction 2021

**Winning features:**
1. **Realized Volatility (RV):** `sqrt(sum(log_returns^2))`
2. **WAP per minute:** Aggregated from order book snapshots
3. **EWM momentum:** `df.ewm(alpha=0.1).mean()` captures recent trend

```python
import numpy as np

def compute_wap(bid_price, ask_price, bid_size, ask_size):
    return (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)

def compute_rv(log_returns):
    return np.sqrt(np.sum(log_returns ** 2))

def compute_order_book_features(df):
    df['wap'] = compute_wap(df.bid_price1, df.ask_price1, df.bid_size1, df.ask_size1)
    df['log_return'] = np.log(df.wap / df.wap.shift(1))
    
    # Per time bucket (10 seconds)
    features = df.groupby('time_id').agg(
        rv=('log_return', compute_rv),
        wap_mean=('wap', 'mean'),
        wap_std=('wap', 'std'),
        bid_ask_spread_mean=('bid_ask_spread', 'mean'),
    )
    return features
```

**t-SNE time ordering trick (360 NN features):**
- Compute t-SNE embedding of time IDs
- Find 360 nearest neighbors in t-SNE space
- Use neighbor's realized volatility as features → captures market regime similarity

**Why it worked:** Similar market regimes (by t-SNE) have correlated volatility. This is a form of learned time-based feature encoding.

---

## G-Research Crypto Forecasting 2022

**Task:** Predict crypto returns across 14 assets.
**Dominant model:** LightGBM (fastest iteration cycle for financial data).

**Fibonacci HMA windows:**
```python
# Hull Moving Average with Fibonacci periods
fibonacci = [3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

for period in fibonacci:
    df[f'hma_{period}'] = compute_hma(df['close'], period)
    df[f'return_{period}'] = df['close'].pct_change(period)
    df[f'vol_{period}'] = df['close'].pct_change().rolling(period).std()
```

**Regime models:** Train separate models for bull/bear/sideways market regimes. Classify regime via VIX-equivalent signal, then blend models based on regime probability.

**Multi-asset correlation features:**
```python
# Feature: relative performance vs crypto market
df['btc_return'] = df[df['Asset_ID'] == 1]['Target']  # BTC as market proxy
df = df.merge(df[df['Asset_ID']==1][['timestamp','Target']].rename(
    columns={'Target':'btc_return'}), on='timestamp', how='left')
df['alpha_vs_btc'] = df['Target'] - df['btc_return']  # excess return
```

---

## Universal Patterns for Financial Competitions

### 1. CV Design Rules
```
Standard KFold = ALWAYS wrong for financial data
Walk-forward = minimum standard
Purged + embargo = gold standard (prevents lookahead bias from overlapping samples)
```

### 2. Feature Engineering Priority
1. Price-derived features (WAP, returns, volatility)
2. Order book imbalance features  
3. Cross-asset correlation features
4. Regime indicators (VIX, momentum signals)
5. Calendar features (day-of-week, time-to-close)

### 3. Model Selection
- **GBDTs:** Best for tabular financial data with many static features
- **RNNs/GRUs:** Best when temporal sequence matters (order book, intraday)
- **Transformers:** Best for long-horizon dependencies (multi-day prediction)
- **Ensemble all three:** Standard practice in top solutions

### 4. Target Denoising (Eigenvalue Clipping)
```python
# Remove noise from correlation matrix using Marchenko-Pastur law
from sklearn.covariance import LedoitWolf

lw = LedoitWolf()
lw.fit(returns)
covariance = lw.covariance_  # Denoised correlation structure
```

### 5. Online Retraining
- Financial data has concept drift
- Retrain every 7-14 days on most recent data window
- Use exponential weighting to give more weight to recent samples:
  ```python
  sample_weight = np.exp(np.linspace(-1, 0, len(X_train)))
  ```

---

Sources:
- Jane Street Market Prediction 2024: https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting
- Optiver Trading at Close 2023: https://www.kaggle.com/competitions/optiver-trading-at-the-close
- Optiver Realized Volatility Prediction: https://www.kaggle.com/competitions/optiver-realized-volatility-prediction
- G-Research Crypto Forecasting: https://www.kaggle.com/competitions/g-research-crypto-forecasting
- Purged CV methodology: https://github.com/stefan-jansen/machine-learning-for-algorithmic-trading
