---
title: Left-Right Swap Augmentation
category: techniques
tags: [augmentation, har, wearable, symmetry]
created: 2026-04-15
updated: 2026-04-15
---

# Left-Right Swap Augmentation

Data augmentation for wearable sensor data that exploits bilateral body symmetry. For each training sample from a right-side sensor, create a mirror sample as if it came from the left side (and vice versa).

## How It Works

1. **Negate the x-axis** (lateral axis) of the accelerometer signal
2. **Swap the sensor ID**: right_arm ↔ left_arm, right_leg ↔ left_leg
3. **Keep the label unchanged** (the activity is the same regardless of which side)

```python
# For a sample from right_arm:
augmented_data = original_data.copy()
augmented_data[:, 0] *= -1           # negate x-axis
augmented_sensor_id = swap_map[original_sensor_id]  # right_arm -> left_arm
```

## Why It Works

- Most human activities are approximately bilaterally symmetric
- A right-arm bicep curl looks like a mirrored left-arm bicep curl
- Doubles the effective training data for each sensor location
- The 1st WEAR challenge winner (2024) used this as a key ingredient for 91.87% macro F1

## Caveats

- Assumes bilateral symmetry — may not hold for all activities (e.g., writing)
- Only negate the lateral axis, not all axes
- Keep video features unchanged (egocentric camera is not limb-specific)
- Only augment training data, never validation

## Pairs Well With
- [[techniques/test-time-augmentation]] — average predictions with L-R flipped input at inference

## See Also
- [[techniques/test-time-augmentation]]
- [[competitions/wear-hasca-2026]]
