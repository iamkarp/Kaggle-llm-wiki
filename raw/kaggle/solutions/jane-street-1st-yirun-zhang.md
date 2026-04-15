# Jane Street Market Prediction — 1st Place Solution
**Author**: Yirun Zhang | **Votes**: 348

---

## Competition
Predict whether to make a trade (action = 0 or 1) based on anonymized market features. Evaluation: utility score (weighted by `resp` values and trade size). Time-series financial data, ~2.4M training rows across ~500 dates.

## Core Architecture: Supervised Autoencoder + MLP (End-to-End)

The winning approach was a **supervised autoencoder** — an autoencoder where the bottleneck representation is simultaneously trained to reconstruct inputs AND predict the target. This is distinct from a standard DAE:

- **Standard DAE**: Pre-train reconstruction unsupervised → use bottleneck as features → train supervised model separately
- **Supervised AE**: Train reconstruction loss + supervised loss jointly, end-to-end, in a single model

```
Input (130 features)
→ Encoder: Dense(256, Swish) → Dense(128, Swish) → Dense(64, Swish)
→ Bottleneck (64-dim)
    ├── Decoder branch: Dense(128) → Dense(256) → Dense(130)  [reconstruction]
    └── Classifier branch: Dense(64) → Dense(32) → Dense(1, Sigmoid)  [prediction]

Total loss = λ_recon * MSE(input, reconstruction) + λ_pred * BCE(target, prediction)
```

The classifier branch uses the bottleneck — this forces the encoder to learn representations that are simultaneously good for reconstruction AND prediction.

## Why This Prevents Label Leakage

**Critical for financial time-series**: The 31-gap purged split means there's a 31-day gap between training and validation. If you train a DAE on all data first and then train the supervised model, you implicitly allow information from the validation period to leak into the bottleneck representation (through the reconstruction objective).

**The fix**: Train the entire supervised AE only on training fold data, including the encoder. Never let validation data touch the encoder during training.

```python
# Per-fold training — encoder is reinitalized and trained from scratch each fold
for fold_idx, (tr_idx, val_idx) in enumerate(purged_cv.split(X, y)):
    # Build fresh model for this fold
    model = build_supervised_autoencoder()
    # Train ONLY on training fold — no validation data touches the encoder
    model.fit(
        X[tr_idx], 
        {'reconstruction': X[tr_idx], 'classification': y[tr_idx]},
        epochs=100, batch_size=4096
    )
    oof_preds[val_idx] = model.predict(X[val_idx])['classification']
    test_preds += model.predict(X_test)['classification'] / n_folds
```

## 31-Gap Purged Group Time-Series Split

Standard `TimeSeriesSplit` is insufficient for financial data because:
- Features may use information from nearby dates
- Records within a date are correlated (same market state)

**Purged split**: Remove a gap of N days between train and validation folds so that any lookback features from the validation period can't "see" training targets.

```python
# Purge gap: remove 31 days between train and validation
# Group by date: all records from a date go to same fold
# Skip the 31 days before each validation window

class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits=5, gap=31):
        self.n_splits = n_splits
        self.gap = gap
    
    def split(self, X, y, groups):  # groups = date
        unique_dates = sorted(np.unique(groups))
        # Assign each unique date to a fold
        # Purge gap-many dates from the end of each train window
        ...
```

The 31-day gap matches the lookback window of some anonymous features, ensuring no leakage.

## Swish Activation > ReLU

Used `Swish` (also called SiLU) activation throughout: `f(x) = x * sigmoid(x)`

**Why Swish outperforms ReLU here**:
- Smooth, non-monotonic — allows small negative values to pass through
- Self-gated: the gate `sigmoid(x)` modulates the signal, similar to an attention mechanism
- Empirically outperformed ReLU, ELU, and GELU in Jane Street experiments
- Particularly effective in deep networks (smooth gradient flow)

```python
import tensorflow as tf
# In Keras: use 'swish' activation
Dense(256, activation='swish')
# Or manually: x * tf.sigmoid(x)
```

## Sample Weights = Mean Absolute Response

Training samples are weighted by `mean(|resp_1|, |resp_2|, |resp_3|, |resp_4|, |resp|)` — the mean absolute value of the 5 response columns (which relate to the trade utility at different time horizons).

**Rationale**: High-resp samples represent larger trades or higher-confidence price movements. Weighting by mean absolute response focuses the model on high-stakes decisions where getting it right matters more.

```python
sample_weights = df[['resp_1', 'resp_2', 'resp_3', 'resp_4', 'resp']].abs().mean(axis=1)
# Use in model.fit(sample_weight=sample_weights)
```

## Multiple Seed Averaging
Train the supervised AE with 5–10 different random seeds per fold and average predictions. This reduces variance from random weight initialization — particularly important for NNs where initialization can significantly affect the local minimum found.

## Key Takeaways
1. Supervised AE (joint reconstruction + classification) outperforms sequential DAE + supervised for financial data
2. End-to-end per-fold training prevents encoder label leakage — critical for time-series
3. Purged group time-series split (31-day gap) is the correct CV for this problem
4. Swish activation consistently outperformed ReLU/GELU in this setting
5. Sample weighting by mean absolute response prioritizes high-stakes decisions
6. Seed averaging reduces NN variance meaningfully — use 5+ seeds
