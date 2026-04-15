---
title: "Negative Downsampling — Training on Imbalanced Data at Scale"
tags: [downsampling, imbalanced, fraud, calibration, bagging, class-imbalance, large-scale]
date: 2026-04-15
source_count: 1
status: active
---

## What It Is
Negative downsampling discards a large fraction of the majority-class (negative) training examples, making training feasible on extremely imbalanced large datasets. After training on the downsampled data, predicted probabilities must be **recalibrated** to reflect the true class distribution.

Grounded in Facebook's ad click prediction paper (He et al., 2014) and Google's app install prediction work. Both independently discovered that training on a 50% subsampled negative set is feasible when corrected with prior calibration.

## When to Use

| Condition | Use Downsampling |
|-----------|-----------------|
| Dataset too large to train on (>50M rows) | Yes |
| Positive class is <1% of data | Yes — severe imbalance slows convergence |
| Class imbalance causes model to predict near-zero for all samples | Yes |
| Dataset is manageable (< 10M rows) | No — use class weights instead |
| Metric is threshold-dependent (F1, recall@k) | Be careful — calibration is critical |

**Prefer class weights for smaller datasets**: `scale_pos_weight` in XGBoost, `class_weight='balanced'` in sklearn. Downsampling is the right tool only when the dataset is too large to train on, or when the imbalance is so extreme (>99.8% negative) that class weights alone are insufficient.

## Sampling Strategy

Keep all positives; randomly sample negatives at rate `s`:

```python
import pandas as pd
import numpy as np

def negative_downsample(df, target_col, neg_sampling_rate=0.002, random_state=42):
    """
    Keep all positives; sample neg_sampling_rate fraction of negatives.
    """
    pos = df[df[target_col] == 1]
    neg = df[df[target_col] == 0]
    
    neg_sampled = neg.sample(frac=neg_sampling_rate, random_state=random_state)
    
    result = pd.concat([pos, neg_sampled]).sample(frac=1, random_state=random_state)
    
    print(f"Original: {len(pos):,} pos, {len(neg):,} neg ({len(pos)/len(df)*100:.3f}% positive)")
    print(f"Sampled:  {len(pos):,} pos, {len(neg_sampled):,} neg ({len(pos)/(len(pos)+len(neg_sampled))*100:.1f}% positive)")
    
    return result.reset_index(drop=True)

# TalkingData example: 0.17% positive, 99.83% negative
train_sampled = negative_downsample(train, 'is_attributed', neg_sampling_rate=0.002)
# Result: ~55% positive (highly artificial but training is now feasible)
```

## Prior Correction (Calibration After Downsampling)

After training on downsampled data, the model's predicted probability `p_model` reflects the sampled class distribution, not the true distribution. Must correct:

### Formula (From Facebook's Paper)
```
p_true = p_model / (p_model + (1 - p_model) / q)
```

Where `q` is the negative sampling rate (fraction of negatives kept).

Derivation: this is Bayes' theorem applied to correct for the artificial over-representation of positives.

```python
def prior_correction(p_model, neg_sampling_rate):
    """
    Correct predicted probabilities after negative downsampling.
    
    p_model: predicted probability from model trained on downsampled data
    neg_sampling_rate: fraction of negatives kept (e.g., 0.002 for TalkingData)
    
    Returns: corrected probability reflecting true class distribution
    """
    # Equivalent to: adjust odds ratio by sampling factor
    q = neg_sampling_rate
    return p_model / (p_model + (1 - p_model) / q)

# Example
p_model = np.array([0.8, 0.5, 0.1])
p_corrected = prior_correction(p_model, neg_sampling_rate=0.002)
# p_corrected will be much lower (true positive rate is ~0.17%)
```

**When calibration matters most**:
- Metric is AUC: calibration doesn't affect ranking → correction doesn't change AUC
- Metric is log-loss, Brier score: calibration matters → must correct
- Metric requires threshold (precision@k, F1): must correct to get meaningful thresholds
- Downstream use of predicted probabilities: always correct

For AUC-only competitions, prior correction often doesn't change the score — but it's still good practice.

## 5-Bag Averaging Over Different Negative Samples

The specific negative sample chosen introduces variance. Average over multiple bags:

```python
def train_bagged_downsampled(train_pos, train_neg, model_factory,
                              neg_sampling_rate=0.002, n_bags=5,
                              X_test=None):
    """
    Train n_bags models on different negative samples; average predictions.
    """
    bag_test_preds = []
    bag_oof_preds = []
    
    for bag_seed in range(n_bags):
        # Different negative sample per bag
        neg_sampled = train_neg.sample(frac=neg_sampling_rate, random_state=bag_seed)
        train_bag = pd.concat([train_pos, neg_sampled]).sample(frac=1, random_state=bag_seed)
        
        X_bag = train_bag[feature_cols]
        y_bag = train_bag[target_col]
        
        model = model_factory()
        model.fit(X_bag, y_bag)
        
        if X_test is not None:
            bag_test_preds.append(model.predict_proba(X_test)[:, 1])
    
    # Average across bags
    final_preds = np.mean(bag_test_preds, axis=0)
    # Apply prior correction
    final_preds = prior_correction(final_preds, neg_sampling_rate)
    
    return final_preds

pos_df = train[train['target'] == 1]
neg_df = train[train['target'] == 0]
preds = train_bagged_downsampled(pos_df, neg_df, LGBMClassifier, n_bags=5, X_test=X_test)
```

**Why bagging over samples reduces variance**:
- Each bag sees a different set of ~368K negatives out of 184M total
- The decision boundary learned from 5 different negative samples is more robust
- Equivalent to ensemble diversity from data subsampling (like the bagging in Random Forests)

## Interaction with Other Techniques

| Technique | Interaction |
|-----------|------------|
| SMOTE/oversampling | Alternative to downsampling; works on small-medium datasets |
| Class weights | Better for <50M row datasets; no calibration needed |
| Focal loss | Alternative loss function for extreme imbalance in NNs |
| Threshold tuning | Choose operating point after correct calibration |
| Pseudo-labeling | Can be combined: pseudo-label high-confidence negatives from test to add back |

## Negative Sampling Rate Selection

```
True positive rate = p_pos
Target positive rate in training = p_target (e.g., 0.5 for balanced)

neg_sampling_rate = (p_pos / (1 - p_pos)) / (p_target / (1 - p_target))
```

For TalkingData: `p_pos = 0.0017`, `p_target = 0.5`:
```python
p_pos = 0.0017
p_target = 0.5
rate = (p_pos / (1 - p_pos)) / (p_target / (1 - p_target))
# rate ≈ 0.0034 → use 0.002 (slightly more aggressive; empirically works well)
```

In practice, try `neg_sampling_rate` in {0.001, 0.002, 0.005, 0.01} and pick by CV AUC.

## Sources
- [[../../raw/kaggle/solutions/talkingdata-fraud-1st-komaki.md]] — 99.8% negative discard, 5-bag averaging, prior correction

## Related
- [[../concepts/categorical-embeddings]] — TalkingData uses both downsampling and topic embeddings
- [[../concepts/pseudo-labeling]] — alternative semi-supervised approach for imbalanced data
- [[../concepts/calibration]] — prior correction is a form of post-hoc calibration
- [[../concepts/validation-strategy]] — downsampled CV requires careful stratification to maintain true class ratio in held-out set
