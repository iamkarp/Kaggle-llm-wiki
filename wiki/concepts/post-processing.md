---
title: "Post-Processing — RankGauss, Calibration, Clipping, Rank Blending"
tags: [post-processing, rankgauss, calibration, temperature-scaling, clipping, rank-transform, probability]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is

Post-processing techniques applied to model predictions *after training* to improve metric scores. These are quick wins that don't require retraining. Most relevant for log-loss and MAE metrics. Does not affect AUC (which is rank-based).

**Post-processing is the consistent gold-vs-silver differentiator in modern Kaggle.** Across competitions from mid-2024 through April 2026, post-processing contributed **+0.01 to +0.03 private-LB** in nearly every top-10 solution. Hunting for data quirks, label-distribution mismatches, and subject-level invariants is now a primary competitive skill — budget 2-3 days for it in any competition.

## RankGauss

Transform predictions to a Gaussian distribution via rank transformation. Useful for:
- Regression targets with heavy-tailed distributions
- Preprocessing model *inputs* (transform skewed numerical features before feeding to neural nets)

```python
from scipy.stats import rankdata
from scipy.special import erfinv
import numpy as np

def rankgauss(x):
    """Transform array to Gaussian via rank."""
    ranks = rankdata(x)
    ranks_norm = (ranks - 0.5) / len(ranks)
    return erfinv(2 * ranks_norm - 1)

# Apply to targets
y_transformed = rankgauss(y_train)

# Sklearn alternative for features
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal', random_state=42)
X_transformed = qt.fit_transform(X)
```

**When it helps:** Neural nets that struggle with skewed outputs; competition metrics that reward Gaussian-like prediction distributions.

**Porto Seguro 1st place:** RankGauss applied to features before feeding to DAE and neural nets — one of the key preprocessing choices that enabled the swap-noise DAE to work well.

## Temperature Scaling (Probability Calibration)

Neural nets and GBMs often output miscalibrated probabilities. Temperature scaling is a single-parameter fix:

```python
import numpy as np
from scipy.special import expit  # sigmoid
from scipy.optimize import minimize_scalar

def temperature_scale(logits, temperature):
    return expit(logits / temperature)

def find_optimal_temperature(logits, y_true):
    from sklearn.metrics import log_loss
    def loss_fn(temp):
        probs = temperature_scale(logits, temp)
        return log_loss(y_true, probs)
    result = minimize_scalar(loss_fn, bounds=(0.1, 10.0), method='bounded')
    return result.x

# Usage:
# 1. Get raw logits from model on validation set
# 2. Find optimal temperature
# 3. Apply to test predictions
logits_val = np.log(probs_val / (1 - probs_val))  # convert probs to logits
temp = find_optimal_temperature(logits_val, y_val)
calibrated_probs_test = temperature_scale(logits_test, temp)
```

**Multi-class variant:** Dirichlet calibration (extends temperature scaling to softmax).

**When to use:** Any competition with log-loss or cross-entropy metric. Does NOT affect AUC. Use on the validation set to find optimal temperature; apply to test predictions.

**Does not affect:** Rankings (AUC, MAP). Only affects probability quality (log-loss, Brier score).

## Probability Clipping

Prevents log-loss blowup from extreme (near 0 or near 1) predictions:

```python
predictions = np.clip(predictions, 1e-6, 1 - 1e-6)
```

Gains 0.001–0.005 improvement on log-loss metrics when models produce some very confident (and potentially wrong) predictions. Almost always safe to apply.

## Rank Blending

Blend raw probability with rank-transformed predictions to reduce sensitivity to outlier predictions:

```python
rank_pred = predictions.rank() / len(predictions)
final = 0.7 * predictions + 0.3 * rank_pred
```

**When helpful:** When some models in the ensemble produce outlier probability predictions that dominate the average. The rank component caps their influence.

## Weighted Mean Subtraction (Competition-Specific)

For competitions where predictions must be zero-centered within groups (e.g., financial relative returns, auction relative price movements):

```python
# Simple mean subtraction
group_mean = preds.groupby('group_id').transform('mean')
preds_adjusted = preds - group_mean

# Weighted mean (e.g., by market cap, trading volume)
def weighted_mean_subtract(preds_df, weight_col, group_col):
    wmean = preds_df.groupby(group_col).apply(
        lambda g: np.average(g['pred'], weights=g[weight_col])
    )
    return preds_df['pred'] - preds_df[group_col].map(wmean)
```

## Label-Based Post-Processing

For competitions with temporal stability metrics or known data patterns:

From Home Credit 2024 1st place: Apply score adjustment based on temporal proxy:
```python
# Rows in the first half of WEEK_NUM range get scores adjusted down
condition = df['WEEK_NUM'] < (max_week - min_week)*0.5 + min_week
df.loc[condition, 'score'] = (df.loc[condition, 'score'] - 0.03).clip(0)
```

**Lesson:** On competitions with complex temporal stability metrics, always analyze whether the metric can be exploited by time-conditional score adjustments. The post-processing delta (~0.0X) can dwarf pure ML differences (~0.00X).

## Target Winsorization for RMSE on Fat-Tailed Distributions

For regression targets with extreme outliers (e.g., stock returns with range [-99%, +10000%]), train on winsorized targets but evaluate on raw targets:

```python
# Multiple clip levels → different bias-variance tradeoffs → ensemble diversity
transforms = {
    'clip_1_99': np.clip(y, np.percentile(y, 1), np.percentile(y, 99)),
    'clip_5_95': np.clip(y, np.percentile(y, 5), np.percentile(y, 95)),
    'clip_fixed': np.clip(y, -95, 500),
}

# Train separate models on each, ensemble predictions
# Tight clips (5-95) reduce outlier influence → lower-variance but biased
# Wide clips (1-99) retain more signal → higher-variance but less biased
```

**Also clip predictions** to a reasonable range. Models can produce wild extrapolations on test data:

```python
preds = np.clip(preds, -95, 500)  # match domain knowledge bounds
```

## Prediction Mean Alignment (Centering)

For RMSE metrics, prediction mean matters. If the market average return is ~15% but your model predicts mean ~2%, a constant prediction of 15% beats the ML model.

```python
# Check centering
print(f"Pred mean: {preds.mean():.1f}, Target mean: {y_train.mean():.1f}")

# If off, diagnose:
# - Log transform on target compresses predictions toward 0
# - Heavy regularization with low learning rate → model hugs 0
# - Strong target clipping biases the learned mean

# Fix: use raw (clipped) targets, not log-transformed
```

**Root cause of poor centering:** Signed log transform `sign(x) * log1p(|x|)` on targets with large positive mean. The log compresses the positive tail more than the negative tail, shifting learned predictions toward 0. RMSE on raw targets heavily penalizes this bias.

## Data-Quirk Detection (Gold-vs-Silver Examples)

Beyond mechanical post-processing, the biggest gains come from detecting dataset-specific anomalies:

**CMI Detect Behavior with Sensor Data (2025)**
Two subjects (SUBJ_019262, SUBJ_045235) had worn their wrist device rotated 180° around the Z-axis. Teams that detected and corrected this at inference time gained meaningful private-LB points. Detection method: plot per-subject accelerometer distributions and look for sign inversions.

**NeurIPS Open Polymer Prediction 2025**
The training data contained a Tg (glass transition temperature) unit mismatch — some values in °C, others in K. The 1st-place team detected this by probing with ±0.1σ perturbations and applied an empirical correction worth significant private-LB gain. The underlying competition-data leak amplified the effect.

**ISIC 2024 Skin Cancer Detection**
Top solutions found that GBDTs on patient metadata plus a small CNN ensemble beat pure image models. The 15mm crop size left insufficient pixel signal — the real signal was in tabular metadata (age, body site, skin type). Post-processing here meant recognizing the modality switch.

**PhysioNet ECG Digitization (Research 2026)**
Top-27 all published writeups. Pre-processing (signal cleaning, trace extraction) and post-processing (median filtering) were the primary differentiator between gold and silver — not model architecture.

**General pattern**: For any competition, ask: (1) Are there subject-level or group-level invariants? (2) Are there unit or distribution mismatches between train/test? (3) Can label caps or structural constraints be enforced as post-processing?

## Gotchas

- **Temperature scaling invalidates AUC:** It doesn't — temperature scaling is monotone, so rankings don't change. Only probability magnitudes shift.
- **Clipping with non-log-loss metrics:** Don't clip when using AUC or ranking metrics — it's unnecessary and could theoretically hurt if predictions are being ranked.
- **Overfitting post-processing to validation:** If you tune temperature on the same validation set you use for model selection, you're overfitting. Use a separate calibration holdout or cross-validate the temperature tuning.
- **Log transform on regression targets with RMSE metric:** If the evaluation metric is RMSE on *raw* (untransformed) targets, training on log-transformed targets produces predictions centered near 0 — potentially worse than a constant prediction at the target mean. Only use log targets when the metric is also in log space (e.g., RMSLE).

## In Jason's Work
Not explicitly applied in March Mania 2026 (log-loss metric would benefit). Candidate for future competitions especially where models are stacked (stacked probabilities are often miscalibrated).

## Sources
- [[../../raw/kaggle/modern-tabular-dl-techniques.md]] — RankGauss, temperature scaling, clipping
- [[../../raw/kaggle/2024-2025-winning-solutions-tabular.md]] — label-based post-processing from Home Credit 2024
- [[../../raw/kaggle/solutions/porto-seguro-1st-jahrer.md]] — RankGauss as preprocessing step
- [[../../raw/kaggle/kaggle-meta-2024-2026.md]] — gold-vs-silver insight; CMI, Open Polymer, ISIC examples

## Related
- [[kaggle-landscape-2024-2026]] — meta-analysis confirming post-processing as the gold-vs-silver differentiator
- [[calibration]] — Platt scaling vs temperature scaling comparison
- [[ensembling-strategies]] — calibrate before ensembling for cleaner blending
- [[denoising-autoencoders]] — RankGauss used as preprocessing for swap-noise DAE
