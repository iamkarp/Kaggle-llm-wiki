---
title: "Post-Processing — RankGauss, Calibration, Clipping, Rank Blending"
tags: [post-processing, rankgauss, calibration, temperature-scaling, clipping, rank-transform, probability]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is

Post-processing techniques applied to model predictions *after training* to improve metric scores. These are quick wins that don't require retraining. Most relevant for log-loss and MAE metrics. Does not affect AUC (which is rank-based).

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

## Gotchas

- **Temperature scaling invalidates AUC:** It doesn't — temperature scaling is monotone, so rankings don't change. Only probability magnitudes shift.
- **Clipping with non-log-loss metrics:** Don't clip when using AUC or ranking metrics — it's unnecessary and could theoretically hurt if predictions are being ranked.
- **Overfitting post-processing to validation:** If you tune temperature on the same validation set you use for model selection, you're overfitting. Use a separate calibration holdout or cross-validate the temperature tuning.

## In Jason's Work
Not explicitly applied in March Mania 2026 (log-loss metric would benefit). Candidate for future competitions especially where models are stacked (stacked probabilities are often miscalibrated).

## Sources
- [[../../raw/kaggle/modern-tabular-dl-techniques.md]] — RankGauss, temperature scaling, clipping
- [[../../raw/kaggle/2024-2025-winning-solutions-tabular.md]] — label-based post-processing from Home Credit 2024
- [[../../raw/kaggle/solutions/porto-seguro-1st-jahrer.md]] — RankGauss as preprocessing step

## Related
- [[../concepts/calibration]] — Platt scaling vs temperature scaling comparison
- [[../concepts/ensembling-strategies]] — calibrate before ensembling for cleaner blending
- [[../concepts/denoising-autoencoders]] — RankGauss used as preprocessing for swap-noise DAE
