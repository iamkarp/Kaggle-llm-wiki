---
id: concept:image-classification-tricks
type: concept
title: Image Classification Tricks
slug: image-classification-tricks
aliases: []
tags:
- cv
- classification
- efficientnet
- pooling
- ensembling
- metadata-fusion
- multi-resolution
status: active
date: 2026-04-14
source_count: 3
---

# Image Classification Tricks

Competition-winning image classification solutions share a set of recurring architectural and training tricks beyond the backbone choice. This page documents the most impactful ones.

## AdaptiveConcatPool2d (avg + max)

Replace global average pooling with a concatenation of average AND max pooling:

```python
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size)
        self.max = nn.AdaptiveMaxPool2d(output_size)

    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], dim=1)

# Replace DenseNet / EfficientNet final pooling
model.features.add_module('pool5', AdaptiveConcatPool2d())
# Classifier input now doubles: 1024 → 2048
model.classifier = nn.Linear(2048, n_classes)
```

**Why**: Average pooling captures mean activation (distributed patterns); max pooling captures peak activation (rare, localized patterns). Concatenating both doubles the signal without adding parameters. Originated in fast.ai; now standard for competitions.

**Applies to**: Any classification head. Drop-in replacement for `nn.AdaptiveAvgPool2d(1)`.

**Source**: Human Protein Atlas (bestfitting 2019).

## Diagnosis-as-Target (Richer Supervision Labels)

When more granular labels exist, use them as the training target even if the competition only scores on a coarser label:

```python
# Competition: binary melanoma classification
# Available: 9-class diagnosis labels (melanoma, nevus, keratosis, ...)

# Standard approach (suboptimal):
loss = F.binary_cross_entropy_with_logits(pred, is_melanoma)

# Winning approach:
loss = F.cross_entropy(pred_9class, diagnosis_label)  # 9-way CE

# At inference, extract melanoma class probability:
melanoma_prob = F.softmax(pred_9class, dim=1)[:, MELANOMA_CLASS_IDX]
```

**Result**: +~0.01 AUC on SIIM-ISIC Melanoma (Bo 2020).

**Why it works**: Richer labels provide more gradient signal per example. "Not melanoma" is not monolithic — the model learns what distinguishes each diagnosis, not just melanoma vs. other.

**Generalization principle**: Always check if higher-cardinality labels are available in the dataset metadata. Even if the competition scores on binary/low-cardinality, training with fine-grained labels almost always helps.

## Multi-Resolution Ensemble

Train identical or similar architectures at different input resolutions and ensemble predictions:

```
EfficientNet B3 @ 384×384    (fast, high diversity)
EfficientNet B5 @ 456×456
EfficientNet B7 @ 600×600    (slow, highest accuracy)
SE-ResNeXt101  @ 384×384     (different architecture)
```

**Resolution-performance tradeoff**:
- Higher resolution = more detail = better accuracy for fine-grained tasks
- Diminishing returns above the object's natural scale
- Ensemble benefit: different resolutions catch different features; errors are less correlated than same-resolution ensemble

**Practical guide**: Train your main model at standard resolution. Add one run at +25% resolution (significant gain, modest cost). Add lower-resolution models for ensemble diversity (fast, different errors).

## Rank-Based Ensembling

Convert each model's raw probabilities to percentile rank before averaging:

```python
from scipy.stats import rankdata
import numpy as np

def rank_ensemble(predictions_list):
    """
    predictions_list: list of (N,) probability arrays, one per model.
    Returns: (N,) ensemble prediction as average of percentile ranks.
    """
    ranked = [rankdata(preds) / len(preds) for preds in predictions_list]
    return np.mean(ranked, axis=0)
```

**Why rank over mean probability**:
- Removes scale/calibration differences between models
- Model A may output 0.3 for its top predictions; Model B may output 0.9 — both indicate "high probability" but simple averaging gives Model B more weight
- Each rank distribution is uniform [0,1]; averaging uniforms is always well-scaled
- Robust to outlier overconfident predictions

**When to use**: Ensembling architectures with different output scales (EfficientNet vs. ResNet vs. SE-ResNeXt). Less important when all models are the same architecture.

**Source**: SIIM-ISIC Melanoma (Bo 2020).

## Metadata Fusion

Fuse patient/image metadata into the classifier head:

```python
class MetadataFusionHead(nn.Module):
    def __init__(self, img_features, meta_dim, hidden=512, n_classes=9):
        super().__init__()
        self.meta_embed = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.fusion = nn.Sequential(
            nn.Linear(img_features + 64, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, img_features, metadata):
        meta_encoded = self.meta_embed(metadata)
        return self.fusion(torch.cat([img_features, meta_encoded], dim=1))
```

**Useful metadata**: age, sex, anatomical site, imaging source, scan parameters.

**Impact**: Modest (+0.003-0.005 AUC in melanoma), but reliable. Anatomical site is usually the most informative metadata feature for medical imaging.

**Architecture note**: Fuse metadata at the head level (after image features), not at the input level (concatenating with the image). Head-level fusion is simpler and works equally well.

## Snapshot Ensembling with Cosine Annealing

Train once with cosine LR and save model checkpoints at each LR minimum:

```python
from torch.optim.lr_scheduler import CosineAnnealingLR
import copy

scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
snapshot_models = []

for epoch in range(total_epochs):
    train_epoch(...)
    scheduler.step()
    if (epoch + 1) % T_max == 0:  # LR minimum = end of cosine cycle
        snapshot_models.append(copy.deepcopy(model.state_dict()))

# Each snapshot is at a different loss landscape basin → free ensemble diversity
```

**Source**: TGS Salt (Babakhin 2018). Single training run → 3-5 ensemble members at no extra training cost.

## External Data Pretraining

Pattern common across multiple competitions:
1. Pretrain on external data (ImageNet → competition domain)
2. Fine-tune on competition data

**Key**: Use external data at full resolution; fine-tune at competition resolution. For medical imaging, look for prior year's challenges on the same body region.

## Classification Checklist

1. **Backbone**: EfficientNet or ConvNeXt for efficiency; ViT for large datasets
2. **Pooling**: AdaptiveConcatPool2d (avg + max)
3. **Loss**: BCEWithLogitsLoss baseline; FocalLoss if class imbalance; Lovász if optimizing F1/IoU
4. **Target**: Use richest available label for training, map to competition label at inference
5. **Ensemble**: Multi-resolution + multi-backbone; rank-based ensembling before averaging
6. **Metadata**: Fuse at classifier head if available
7. **TTA**: At least H-flip; D4 group (8 augmentations) for rotationally symmetric tasks

## Related
- [[metric-learning-cv]] — ArcFace + NN retrieval as alternative to softmax classification
- [[image-augmentation]] — augmentation strategies for classification
- [[loss-functions-cv]] — FocalLoss, Lovász for imbalanced classification

## Sources
- `raw/kaggle/solutions/human-protein-atlas-1st-bestfitting.md` — AdaptiveConcatPool2d, FocalLoss + Lovász
- `raw/kaggle/solutions/siim-isic-melanoma-1st-bo.md` — rank ensembling, diagnosis-as-target, metadata fusion, multi-resolution
- `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` — cosine annealing with snapshot ensembling

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[concepts/image-augmentation|Image Augmentation]]
- _works_with_ → [[concepts/loss-functions-cv|Loss Functions for CV]]
- _works_with_ → [[concepts/metric-learning-cv|Metric Learning for CV]]

**Incoming:**
- [[concepts/image-augmentation|Image Augmentation]] _works_with_ → here
- [[concepts/loss-functions-cv|Loss Functions for CV]] _works_with_ → here
- [[concepts/metric-learning-cv|Metric Learning for CV]] _works_with_ → here
- [[concepts/multimodal-classification|Multimodal Classification]] _works_with_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
