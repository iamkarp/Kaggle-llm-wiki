---
id: concept:online-learning
type: concept
title: Online Learning — Retraining During Test, Memory Management, Incremental Adaptation
slug: online-learning
aliases: []
tags:
- online-learning
- retraining
- time-series
- distribution-shift
- hdf5
- memory
- trading
- market
- gru
- auxiliary-targets
status: active
date: 2026-04-15
source_count: 2
---

## What It Is
Online learning (in the Kaggle sense) means retraining models on newly revealed test data during the competition's evaluation phase. When a competition provides test data incrementally — one day or one batch at a time — you can incorporate recent data into the model before predicting the next batch. This combats distribution shift as market regimes or behavioral patterns evolve over the test period.

Distinguished from:
- **Standard offline training**: train once on all available training data, freeze model
- **Continual learning**: update model weights incrementally (gradient steps on new data)
- **Online Kaggle learning**: full model retraining on a growing dataset at scheduled intervals

## When Online Retraining Helps

**Good conditions**:
- Competition provides test data sequentially (day-by-day, batch-by-batch)
- Significant time gap between training cutoff and test period (weeks to months)
- Target behavior shifts over time (market regimes, seasonal patterns, concept drift)
- Retraining is fast enough to fit within compute budget

**When it doesn't help**:
- Test data is I.I.D. with training (no drift)
- Competition provides all test data simultaneously
- Retraining budget exceeds available compute
- Dataset is too small — adding a few new rows doesn't meaningfully change the model

## Retraining Interval Selection

Retraining on every new day is ideal but expensive. Key tradeoff:

| Interval | Freshness | Compute Cost | Best For |
|----------|-----------|-------------|---------|
| Every day | Maximum | Very high | Small fast models |
| Every 7 days | High | Moderate | Medium models |
| Every 12 days | Good | Low-moderate | Default starting point |
| Every 30 days | Low | Low | Large slow models |

**Rule of thumb**: Start with a 12-day interval (Optiver Close 1st place). Tune based on the detected drift rate — if model performance degrades quickly after training, shorten the interval.

## Implementation Pattern

```python
# Optiver Trading at the Close pattern
from pathlib import Path
import pandas as pd
import h5py

class OnlineLearner:
    def __init__(self, retrain_interval=12):
        self.retrain_interval = retrain_interval
        self.model = None
        self.days_since_retrain = 0
        self.accumulated_data = []
    
    def update(self, new_day_data: pd.DataFrame, new_day_targets: pd.Series):
        """Add new data; retrain if interval reached."""
        self.accumulated_data.append((new_day_data, new_day_targets))
        self.days_since_retrain += 1
        
        if self.days_since_retrain >= self.retrain_interval:
            self._retrain()
            self.days_since_retrain = 0
    
    def _retrain(self):
        # Combine original training data + all accumulated test data
        original = load_original_training()
        new_data = pd.concat([d for d, _ in self.accumulated_data])
        new_targets = pd.concat([t for _, t in self.accumulated_data])
        
        X_full = pd.concat([original['X'], new_data])
        y_full = pd.concat([original['y'], new_targets])
        
        self.model = train_model(X_full, y_full)
    
    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)
```

## Memory-Efficient Day-by-Day Loading (HDF5)

When accumulated data grows large (months of tick data), loading everything into RAM becomes impossible. Load day-by-day from HDF5 files:

```python
import h5py
import numpy as np

def load_training_data_incremental(data_dir: Path, days: list) -> tuple:
    """Load days one at a time, accumulate features incrementally."""
    X_parts = []
    y_parts = []
    
    for day in days:
        h5_path = data_dir / f'day_{day:04d}.h5'
        with h5py.File(h5_path, 'r') as f:
            X_day = f['features'][:]   # loads just this day
            y_day = f['targets'][:]
        
        X_parts.append(X_day)
        y_parts.append(y_day)
        # h5py file is closed; memory is freed
    
    return np.vstack(X_parts), np.concatenate(y_parts)

# HDF5 creation (from pandas)
def save_day_to_h5(df: pd.DataFrame, path: Path):
    df.to_hdf(path, key='df', mode='w', complevel=5, complib='blosc')
```

**Why HDF5 over CSV/Parquet**:
- Random access by row/column without loading full file
- Compressed by default (blosc compression ~3–5× ratio)
- Compatible with numpy array slicing syntax
- h5py handles partial reads efficiently

## Rank Features Within Time Buckets

From Optiver Close: rank-normalize features within each time bucket before feeding to models. This removes cross-unit scale differences and captures relative positioning:

```python
def add_rank_features(df: pd.DataFrame, feature_cols: list, bucket_col='time_bucket') -> pd.DataFrame:
    """Rank each feature within its time bucket (percentile rank)."""
    for col in feature_cols:
        df[f'{col}_rank'] = (
            df.groupby(bucket_col)[col]
            .rank(pct=True)           # rank as percentile [0, 1]
        )
    return df
```

**Why rank within bucket**: Financial features have very different scales across stocks (a stock at $5 vs. $500). Ranking within a time bucket normalizes by the cross-sectional distribution — the model sees "this stock is in the top 20% of bid_size for this second" rather than a raw dollar amount.

## Weighted-Mean Post-Processing

Standard: subtract the simple mean of predictions from a time bucket. Better: subtract the volume/market-cap weighted mean.

```python
def weighted_mean_adjust(preds_df: pd.DataFrame,
                          bucket_col='time_id',
                          weight_col='stock_weight') -> pd.Series:
    """Subtract weighted mean from predictions within each time bucket."""
    
    # Compute weighted mean per bucket
    def wmean(group):
        return np.average(group['pred'], weights=group[weight_col])
    
    bucket_wmeans = preds_df.groupby(bucket_col).apply(wmean)
    
    # Subtract
    adjusted = preds_df['pred'] - preds_df[bucket_col].map(bucket_wmeans)
    return adjusted
```

**Why weighted mean**: In closing auction prediction, large-cap stocks have disproportionate influence on the index. The post-processing should reflect this economic reality.

## Applicability to Jason's Work

### Forex Trading
The NFP straddle strategy already incorporates a form of online adaptation — live_straddle_v2.py executes in real-time using current market data. However, the strategy parameters (4-pip stop, 5-pip trail) are fixed from backtesting. Online retraining could:
- Adapt position sizing to current volatility regime
- Adjust stop widths based on recent NFP event volatility

**Caution**: Any parameter change requires explicit Jason approval.

### Kaggle Time-Series Competitions
For any future competition with sequential test release (trading, sales, demand forecasting):
1. Check if test data is released incrementally
2. If yes, implement 12-day retraining schedule by default
3. Use HDF5 for memory-efficient accumulation
4. Add rank features within time buckets

## GRU Inference-Time Weight Updates (Jane Street 2025)

A different, more surgical form of online learning: rather than full model retraining, perform **one gradient step per day** on auxiliary targets during inference. From Jane Street 2025 8th place (Grigoreva):

- **+0.008 CV improvement** from enabling online weight updates on GRU
- GRU benefits far more than MLP — recurrent hidden state adapts naturally to sequential updates
- Update on auxiliary targets (responder_7, responder_8, rolling 8-day and 60-day averages) — these are available same-day; the primary scored target is not
- One step with small LR (1e-5 to 1e-4); more steps risk catastrophic forgetting

```python
class OnlineGRU:
    def __init__(self, model, lr=1e-5):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def step_and_predict(self, X_day, y_aux_day):
        """One update on auxiliary targets, then predict primary target."""
        # Update step (auxiliary targets available same-day)
        self.model.train()
        self.optimizer.zero_grad()
        aux_pred = self.model(X_day, head='auxiliary')
        loss = self.loss_fn(aux_pred, y_aux_day)
        loss.backward()
        self.optimizer.step()
        
        # Predict primary target with updated weights
        self.model.eval()
        with torch.no_grad():
            return self.model(X_day, head='primary')
```

**Calculated rolling auxiliaries**: Grigoreva computed 8-day and 60-day rolling averages of the responder as additional auxiliary targets — smoother, lower-variance update signal than raw daily responder values.

**Two-head architecture**: Auxiliary head accepts same-day targets for online updates; primary head predicts the scored target. Shared GRU backbone means auxiliary updates improve primary head's hidden state too.

### GRU vs. MLP for Online Learning

| Model | Online Learning Gain | Why |
|-------|--------------------|----|
| GRU | **+0.008 CV** (massive) | Recurrent hidden state encodes temporal context; updates sharpen it |
| MLP | Marginal | Stateless — no accumulated context to refine; just shifts weights slightly |

This suggests: when online learning is permitted, prefer recurrent architectures specifically for their update-responsiveness.

## Sources
- [[../../raw/kaggle/solutions/optiver-close-1st-hyd.md]] — 12-day retraining, HDF5 loading, rank features, weighted-mean post-processing
- [[../../raw/kaggle/solutions/jane-street-2025-8th-grigoreva.md]] — GRU +0.008 from 1-step inference updates; auxiliary targets; 200-day CV window

## Related
- [[../concepts/validation-strategy]] — temporal CV design for online learning settings
- [[../concepts/feature-selection]] — time consistency selection addresses same distribution shift problem
- [[../strategies/nfp-straddle-forex]] — live trading analogue (fixed params, real-time execution)
- [[../concepts/ensembling-strategies]] — CatBoost+GRU+Transformer blend in the Optiver solution
- [[../concepts/denoising-autoencoders]] — GRU two-head architecture parallels supervised AE two-head design

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _cites_ → `source:jane-street-2025-8th-grigoreva` (Jane Street Real-Time Market Data Forecasting 2025 — 8th Place Solution)
- _cites_ → `source:optiver-close-1st-hyd` (Optiver Trading at the Close — 1st Place Solution)
- _works_with_ → [[concepts/denoising-autoencoders|Denoising Autoencoders — Tabular Representation Learning]]
- _works_with_ → [[concepts/ensembling-strategies|Ensembling Strategies — Fourth-Root Blend, Stacking, Diversity]]
- _works_with_ → [[concepts/feature-selection|Feature Selection — Time Consistency, Forward Selection, Permutation Importance]]
- _works_with_ → [[concepts/validation-strategy|Validation Strategy — CV Design, Gap Tracking, Anti-Patterns]]

**Incoming:**
- [[strategies/nfp-straddle-forex|NFP Straddle — Forex Volatility Strategy on Non-Farm Payroll]] _requires_ → here
- [[concepts/ensembling-strategies|Ensembling Strategies — Fourth-Root Blend, Stacking, Diversity]] _works_with_ → here
- [[concepts/financial-competition-patterns|Financial & Trading Competition Patterns]] _related_to_ → here
- [[concepts/time-series-cv|Time Series Cross-Validation — Walk-Forward, Purged CV, Gap Strategy]] _related_to_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
