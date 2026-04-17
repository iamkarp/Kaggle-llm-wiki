---
title: "Multimodal Classification"
tags: [vision, vlm, multimodal, classification, vqa]
date: 2026-04-16
source_count: 1
status: draft
---

## What It Is

Multimodal classification combines information from multiple data modalities (images, text, metadata) to produce structured predictions. In Kaggle context, this often involves using vision-language models (VLMs) to extract features from images and combining with tabular or text features for final classification.

## When To Use It

- Competition data includes images alongside structured labels
- Output requires multi-column categorical predictions from visual input
- Pre-trained VLMs can understand domain-specific visual content (medical, automotive, satellite)

## Approaches

1. **VLM pipeline**: Use a VLM for scene understanding → LLM for structured extraction (AUTOPILOT pattern)
2. **Feature fusion**: Extract embeddings from image encoder + tabular features → train classifier on concatenated features
3. **End-to-end**: Fine-tune a multimodal model directly on the task (requires sufficient labeled data)

## Hyperparameters

- VLM temperature: 0 for deterministic extraction, 0.1-0.3 for diverse descriptions
- Fusion method: early (concatenation), late (ensemble), or attention-based
- Image preprocessing: resolution, augmentation strategy

## Gotchas

- VLMs can hallucinate visual details — always validate extracted features against known constraints
- Multi-column output requires careful schema validation (one bad column can tank macro metrics)
- API costs scale linearly with image count — budget for full dataset inference before committing

## In Jason's Work

Used in the AUTOPILOT VQA 2026 competition with a 3-stage Qwen-VL + Claude Sonnet pipeline for 25-column dashcam classification.

## Related

- [[../strategies/autopilot-vqa-pipeline]] — implementation using this approach
- [[../entities/qwen-vl]] — VLM used for visual understanding
- [[../concepts/image-classification-tricks]] — single-modality image techniques
