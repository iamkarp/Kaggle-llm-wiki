---
title: Test-Time Augmentation (TTA)
category: techniques
tags: [inference, augmentation, ensemble]
created: 2026-04-15
updated: 2026-04-15
---

# Test-Time Augmentation (TTA)

At inference time, make predictions on multiple augmented versions of each test sample and average the probability outputs. A free accuracy boost that doesn't require retraining.

## L-R Flip TTA for HAR

For wearable sensor data:
1. Predict on the original test sample → probabilities P1
2. Apply [[techniques/lr-swap-augmentation]] to the test sample (negate x-axis, swap sensor ID)
3. Predict on the augmented sample → probabilities P2
4. Final prediction = argmax((P1 + P2) / 2)

```python
# Original prediction
probs_orig = model(x, sensor_ids)

# Flipped prediction
x_flip = x.clone()
x_flip[:, :, 0] *= -1  # negate x-axis
sensor_flip = swap_sensor_ids(sensor_ids)
probs_flip = model(x_flip, sensor_flip)

# Average
probs_final = (probs_orig + probs_flip) / 2
```

## General TTA Strategies

| Domain | Augmentations | Typical Gain |
|--------|---------------|--------------|
| Images | H-flip, crops, rotations | +0.5-2% |
| Time series | L-R flip, jitter, scaling | +0.5-1% |
| Tabular | N/A (use model ensemble) | — |
| NLP | N/A (use model ensemble) | — |

## When to Use

- Always try it when the augmentation used during training can be applied at test time
- Nearly free (2x inference cost for L-R flip TTA)
- Diminishing returns with many augmentations — 2-4 variants is usually optimal
- Stack with [[techniques/threshold-optimization]] for additional gains

## See Also
- [[techniques/lr-swap-augmentation]]
- [[techniques/threshold-optimization]]
