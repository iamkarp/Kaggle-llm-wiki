---
title: Sensor Location Embedding
category: techniques
tags: [har, embedding, wearable, categorical]
created: 2026-04-15
updated: 2026-04-15
---

# Sensor Location Embedding

A learnable embedding that encodes which sensor/body location a data window comes from. Critical for HAR when training with multiple sensor positions but testing with a single unknown position per sample.

## Implementation

```python
self.sensor_emb = nn.Embedding(num_locations, emb_dim)  # e.g., (4, 16)

# In forward:
sensor_token = self.sensor_emb(sensor_id)  # (batch, 16)
# Expand and add to each patch embedding
x = x + sensor_token.unsqueeze(1)  # broadcast over patch dimension
```

## Why It Matters

In the WEAR challenge, training data has 4 sensor locations per window (left/right arm/leg), but test data has only 1 random location per window. Without sensor embedding:
- The model treats all locations identically
- It can't learn location-specific motion patterns (e.g., arm swing vs leg stride)
- It can't compensate for orientation differences between mounting points

With sensor embedding, the model learns a 16-dim representation of each body location that modulates its interpretation of the accelerometer signal.

## When to Use

- Multi-sensor HAR where sensor identity varies between train/test
- Any problem where a categorical context variable (device type, user group, sensor ID) should modulate how the model interprets the time series
- Analogous to segment embeddings in BERT or modality tokens in multimodal models

## See Also
- [[techniques/patchtst]]
- [[competitions/wear-hasca-2026]]
