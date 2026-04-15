# Jane Street Real-Time Market Data Forecasting 2025 — 8th Place Solution
**Author**: Grigoreva | **Year**: 2025 | **Votes**: 290

---

## Competition
Predict responders (market signals) from anonymized financial features. Time-series structure; data arrives sequentially by date. Online learning is explicitly permitted — models can update weights during inference as new data is revealed.

## Core Result: Online Learning +0.008 CV Improvement

The single biggest gain in the solution was enabling online learning on the GRU. Starting from a frozen GRU trained offline, adding one forward pass per day of weight updates during inference improved CV by **+0.008** — a massive jump for a financial time-series competition.

**GRU benefited far more than MLP from online learning.** The recurrent structure makes GRU naturally suited to sequential adaptation: it maintains a hidden state that already encodes temporal context, so weight updates on recent data sharpen that context quickly. MLP showed only marginal improvement from the same online learning procedure.

## One Forward Pass Per Day (Inference-Time Weight Update)

The online learning mechanism is minimal by design — one update step per day of test data:

```python
class OnlineGRU:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def update_and_predict(self, X_day, y_day_aux=None):
        """
        One forward+backward pass on today's data before predicting.
        Uses auxiliary targets (available same-day) for the update step.
        """
        if y_day_aux is not None:
            self.model.train()
            self.optimizer.zero_grad()
            preds_aux = self.model(X_day, head='auxiliary')
            loss = self.loss_fn(preds_aux, y_day_aux)
            loss.backward()
            self.optimizer.step()
        
        # Predict primary target with updated weights
        self.model.eval()
        with torch.no_grad():
            return self.model(X_day, head='primary')
```

**Key design choices**:
- Update on **auxiliary targets** (responder_7, responder_8, rolling averages) — not the primary target, which isn't available at inference time
- One step only — more steps risk overfitting to the most recent day
- Small learning rate for update step (1e-5 to 1e-4) — avoids catastrophic forgetting of offline training

## Auxiliary Targets: responder_7, responder_8, Rolling Averages

The competition provided multiple responder columns. The primary target was scored; auxiliary responders were available contemporaneously (same-day) and could be used for the online update.

Grigoreva also **calculated** additional auxiliary targets:
- Rolling 8-day average of the primary responder
- Rolling 60-day average of the primary responder

These rolling averages are computable from the revealed test data as inference progresses. They serve as smoother, lower-variance training signals for the online update step — less noisy than the raw daily responder.

```python
# Compute rolling auxiliary targets on the fly during test inference
def compute_rolling_auxiliaries(responder_history, windows=[8, 60]):
    aux = {}
    for w in windows:
        if len(responder_history) >= w:
            aux[f'resp_roll_{w}'] = np.mean(responder_history[-w:])
        else:
            aux[f'resp_roll_{w}'] = np.mean(responder_history)
    return aux
```

## Time-Series CV: 200-Day Validation Window

Standard k-fold is inappropriate for financial time series. Grigoreva used a single 200-day validation window at the end of the training period:

```
Training data:     Day 1 → Day T-200
Validation:        Day T-200 → Day T
Test (competition): Day T+1 → Day T+N
```

The 200-day window was chosen to match the expected test period length, ensuring CV simulates the actual evaluation distribution. No gap/purge needed here because the features themselves don't use future information.

## Market Average and Rolling Stat Features

Two categories of cross-sectional features:

### Market Average Features
At each timestamp, compute the cross-sectional mean of key features across all stocks/instruments:
```python
# For each time step, compute market-wide average
market_avg = df.groupby('date_id')[feature_cols].mean()
market_avg.columns = [f'{c}_market_avg' for c in feature_cols]
df = df.merge(market_avg, on='date_id', how='left')

# Deviation from market average (relative positioning)
for col in feature_cols:
    df[f'{col}_vs_market'] = df[col] - df[f'{col}_market_avg']
```

### Rolling Statistics
Multi-window rolling features capture trend and volatility at different time scales:
```python
windows = [8, 60]
for col in key_features:
    for w in windows:
        df[f'{col}_roll{w}_mean'] = df.groupby('symbol')[col].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f'{col}_roll{w}_std'] = df.groupby('symbol')[col].transform(
            lambda x: x.rolling(w, min_periods=1).std()
        )
```

The 8-day and 60-day windows mirror the auxiliary target horizons — the model sees both the feature rolling stats and the target rolling stats at the same timescales.

## GRU Architecture

```
Input: (batch, seq_len, n_features)
→ GRU(hidden_size=256, num_layers=3, dropout=0.2, bidirectional=False)
→ Last hidden state: (batch, 256)
→ Two heads:
    ├── Auxiliary head: Linear(256→128)→ReLU→Linear(128→n_aux)   [for online update]
    └── Primary head:  Linear(256→128)→ReLU→Linear(128→1)         [scored target]
```

The two-head design is essential: auxiliary head accepts same-day available targets for online updates; primary head predicts the scored target. They share the GRU backbone, so online updates on auxiliary targets also improve the primary head's hidden state.

## Key Takeaways
1. Online learning during inference can yield massive CV gains (+0.008) on financial time series
2. GRU benefits far more from online learning than MLP — recurrent hidden state adapts naturally
3. One update step per day with small LR is sufficient; more steps risk overfitting
4. Update on auxiliary targets (available same-day), not primary scored target
5. Calculated rolling auxiliaries (8-day, 60-day averages) provide smoother update signals
6. 200-day single validation window is more realistic than k-fold for sequential financial data
7. Market average features + deviations capture cross-sectional regime context
