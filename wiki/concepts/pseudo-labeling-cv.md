---
id: concept:pseudo-labeling-cv
type: concept
title: Pseudo-Labeling for CV
slug: pseudo-labeling-cv
aliases: []
tags:
- cv
- pseudo-labeling
- semi-supervised
- segmentation
- confidence-threshold
status: active
date: 2026-04-14
source_count: 2
---

# Pseudo-Labeling for CV

Pseudo-labeling (also called self-training) uses a model's own predictions on unlabeled data as training targets. In CV competitions, it is most powerful for **segmentation** (where labeled data is expensive) and **classification** with large unlabeled test sets.

See also [[pseudo-labeling]] for the tabular variant.

## Multi-Stage Pipeline for Segmentation

TGS Salt (Babakhin, 2018) established the canonical 3-stage segmentation pseudo-labeling pipeline:

```
Stage 1: Train on N labeled images → Model M1
  ↓
Generate pseudo-labels on test set (ensemble M1 predictions)
Filter: keep confident predictions (ensemble agreement > threshold)
  ↓
Stage 2: Train on N labeled + K confident pseudo-labeled images → Model M2 (stronger)
  ↓
Generate new pseudo-labels using M2 (more accurate, can lower threshold)
  ↓
Stage 3: Train on N labeled + M pseudo-labeled (M > K) → Final model
```

Each stage uses a more accurate model to generate better pseudo-labels, enabling more test images to cross the confidence threshold. Total training set grows from N → N+K → N+M.

## Confidence Filtering: Ensemble Agreement

For segmentation, pixel-level ensemble agreement is more reliable than single-model confidence:

```python
def confident_pseudo_labels(model_ensemble, test_images,
                             agreement_threshold=0.8):
    """
    Keep pseudo-labels only when ensemble members agree on every pixel.
    """
    keep_mask = []
    pseudo_labels = []

    with torch.no_grad():
        for img in test_images:
            # Get binary prediction from each ensemble member
            member_preds = [
                (m.predict(img) > 0.5).float()
                for m in model_ensemble.members
            ]
            # Pixel-wise agreement rate across members
            agreement = torch.stack(member_preds).float().mean(0)  # (H, W)

            # Keep only if all members agree on every pixel
            confident = (agreement.min() > agreement_threshold or
                         agreement.max() < (1 - agreement_threshold))
            keep_mask.append(confident)
            pseudo_labels.append((model_ensemble.predict(img) > 0.5).float())

    return pseudo_labels, keep_mask
```

**Ensemble agreement vs. single-model confidence**:
- Single model: `pred.max() > 0.9` — only checks the most confident prediction
- Ensemble agreement: all members must agree — much more conservative and reliable
- Typical threshold: 0.8 agreement (start high; lower in later stages as model improves)

## Loss Progression with Pseudo-Labels

Pseudo-labeling in segmentation pairs naturally with the BCE → Lovász loss progression:

```python
def get_loss_and_pseudo_config(stage):
    if stage == 1:
        # Stable training on labeled data only
        return BCEWithLogitsLoss(), agreement_threshold=0.85
    elif stage == 2:
        # Introduce Lovász (post-BCE warmup)
        return LovaszBCELoss(weight_bce=0.4, weight_lovasz=0.6), agreement_threshold=0.75
    elif stage == 3:
        # Full Lovász for final metric optimization
        return LovaszBCELoss(weight_bce=0.2, weight_lovasz=0.8), agreement_threshold=0.65
```

BCE → Lovász within each stage (see [[loss-functions-cv]]); threshold loosens across stages.

## Classification Pseudo-Labeling

For binary/multi-class classification, the simpler form:

```python
# Standard classification pseudo-labeling
soft_preds = ensemble.predict_proba(test_images)  # (N_test, C)

# High-confidence: include as training data
confident_mask = soft_preds.max(axis=1) > 0.95
pseudo_X = test_images[confident_mask]
pseudo_y = soft_preds[confident_mask].argmax(axis=1)

# Retrain with combined dataset
train_with(np.concat([train_X, pseudo_X]),
           np.concat([train_y, pseudo_y]))
```

**Shuffle augmentation** (Santander): for tabular/1D data, create multiple shuffled copies of each pseudo-labeled example to increase effective sample count.

## CV Complications

Pseudo-labeled test data cannot be in the validation fold. Options:
1. Re-run full CV with pseudo-labels added only to training folds
2. Use pseudo-labels only for final retraining after CV (no CV benefit, but safe)
3. Single holdout validation (not K-fold) if pseudo-label volume is large

**Leakage risk**: if pseudo-labels correlate with the test target distribution (they will), validation metrics become optimistic. Always validate on a holdout that contains no pseudo-labeled images.

## When Pseudo-Labeling Is Worth It

| Condition | Benefit |
|-----------|---------|
| Large unlabeled test set (>50% of labeled size) | High |
| Model already achieves good CV score (not underfitting) | High |
| Ensemble available for confidence filtering | High |
| Single model, low accuracy | Low (noise propagation) |
| Small test set | Low (few pseudo-labels to add) |

## Related
- [[pseudo-labeling]] — tabular variant with prior correction
- [[loss-functions-cv]] — BCE→Lovász progression
- [[segmentation-architectures]] — U-Net architectures generating the pseudo-labels

## Sources
- `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` — 3-stage segmentation pseudo-labeling, ensemble agreement filter
- `raw/kaggle/solutions/santander-transaction-1st-fl2o.md` — classification pseudo-labeling with shuffle augmentation

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _improves_on_ → [[concepts/pseudo-labeling|Pseudo-Labeling — Semi-Supervised Learning with High-Confidence Test Predictions]]
- _works_with_ → [[concepts/loss-functions-cv|Loss Functions for CV]]
- _works_with_ → [[concepts/pseudo-labeling|Pseudo-Labeling — Semi-Supervised Learning with High-Confidence Test Predictions]]
- _related_to_ → [[concepts/segmentation-architectures|Segmentation Architectures]]

**Incoming:**
- [[concepts/image-augmentation|Image Augmentation]] _works_with_ → here
- [[concepts/loss-functions-cv|Loss Functions for CV]] _works_with_ → here
- [[concepts/medical-imaging-patterns|Medical Imaging & Bioinformatics Competition Patterns]] _related_to_ → here
- [[concepts/segmentation-architectures|Segmentation Architectures]] _related_to_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
