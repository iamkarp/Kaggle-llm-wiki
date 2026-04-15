---
title: Loss Functions for CV
tags: [cv, loss-functions, focal-loss, lovasz, dice, tversky, skeleton-recall, bce]
date: 2026-04-14
source_count: 3
status: active
---

# Loss Functions for CV

CV competitions use specialized loss functions tuned to their metrics and class imbalance characteristics. This page covers the main loss families and when to use each.

## BCE → Lovász Progression

Start with BCE for stable early training, then switch to Lovász to optimize the IoU metric directly:

```python
def get_loss(epoch, total_epochs):
    if epoch < total_epochs * 0.4:
        return F.binary_cross_entropy_with_logits  # stable gradients early
    else:
        return LovaszBCELoss(weight_bce=0.3, weight_lovasz=0.7)

class LovaszBCELoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_lovasz=0.5):
        super().__init__()
        self.w_bce = weight_bce
        self.w_lovasz = weight_lovasz

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        lovasz = lovasz_hinge(pred, target)  # from lovasz-softmax library
        return self.w_bce * bce + self.w_lovasz * lovasz
```

**Why BCE first**: Lovász has poor gradient behavior from random initialization — many local minima where the model predicts everything as empty. After BCE warms up the model, Lovász fine-tunes toward the actual IoU metric.

**Source**: TGS Salt (Babakhin 2018). Also used in Human Protein Atlas.

## Focal Loss

Focal loss down-weights easy examples, focusing training on hard examples. Critical for class imbalance:

```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma  # focusing parameter; 2.0 is standard
        self.alpha = alpha  # class weight for positive class

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

**When to use**: Any classification/detection task where easy negatives dominate training (object detection, medical screening with <5% positive rate). The `(1-pt)^gamma` term makes the loss near-zero for well-classified examples.

**Combined with Lovász** for multi-label imbalanced classification:
```python
loss = 0.5 * focal_loss(pred, target) + 0.5 * lovasz_loss(pred, target)
```

**Source**: Human Protein Atlas (bestfitting 2019) — 28 classes, some with <100 examples.

## Tversky Loss and Focal-Tversky++

Tversky loss generalizes Dice by separately weighting false positives and false negatives:

```python
class FocalTverskyPlusPlus(nn.Module):
    """
    alpha: FN weight (set > 0.5 to penalize missing positives more)
    beta:  FP weight
    gamma: focal exponent
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()

        tversky = (tp + 1) / (tp + self.alpha * fn + self.beta * fp + 1)
        focal_tversky = (1 - tversky) ** (1 / self.gamma)
        return focal_tversky
```

**Parameter selection rule**:
- `alpha=0.7, beta=0.3`: penalizes false negatives more — medically appropriate (missing an aneurysm is worse than a false alarm)
- `alpha=0.5, beta=0.5`: reduces to standard Dice loss
- `gamma < 1`: increases relative penalty on easy cases; `gamma > 1`: standard focal behavior

**Source**: RSNA Aneurysm (tomoon33 2025) — 3D vessel segmentation.

## Dice Loss

Standard Dice coefficient loss for binary segmentation:

```python
def dice_loss(pred, target, smooth=1.0):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
```

**Limitation**: Dice penalizes incorrect volume. For thin/tubular structures (vessels, nerves), a single broken connection causes a large Dice penalty disproportionate to the visual error.

## SkeletonRecall Loss (for Tubular Structures)

Adds a term penalizing breaks in the structural centerline:

```python
class SkeletonRecallLoss(nn.Module):
    def __init__(self, alpha=0.5):  # alpha: weight of skeleton term
        super().__init__()
        self.alpha = alpha

    def compute_skeleton(self, mask):
        from skimage.morphology import skeletonize
        skeleton = np.zeros_like(mask)
        for b in range(mask.shape[0]):
            skeleton[b] = skeletonize(mask[b].cpu().numpy())
        return torch.tensor(skeleton, device=mask.device)

    def forward(self, pred, target):
        dice = 1 - dice_coefficient(pred, target)

        target_skeleton = self.compute_skeleton(target > 0.5)
        pred_binary = (pred > 0.5).float()
        skeleton_recall = (pred_binary * target_skeleton).sum() / (target_skeleton.sum() + 1e-6)
        skeleton_loss = 1 - skeleton_recall

        return (1 - self.alpha) * dice + self.alpha * skeleton_loss
```

**When to use**: Any segmentation where **connectivity matters** more than pixel count:
- Blood vessels, nerves, airways (medical imaging)
- Roads, rivers (satellite imagery)
- Cracks, wires (industrial inspection)

A vessel that is 99% correctly segmented but has one break is clinically useless. SkeletonRecall specifically penalizes that break.

**Source**: RSNA Aneurysm (tomoon33 2025).

## Lovász Loss

Lovász-Softmax (Berman et al., 2018) is a differentiable surrogate for Jaccard/IoU metric. Install via `pip install lovasz-softmax`:

```python
from lovasz_losses import lovasz_hinge, lovasz_softmax

# Binary segmentation
loss = lovasz_hinge(logits, labels)

# Multi-class segmentation
loss = lovasz_softmax(F.softmax(logits, dim=1), labels)
```

**Key property**: Lovász directly optimizes the IoU metric, not a proxy. BCE minimizes cross-entropy which correlates with IoU but isn't the same.

## Loss Selection Guide

| Task | Metric | Recommended Loss |
|------|--------|-----------------|
| Binary segmentation (balanced) | IoU/Dice | Dice → Lovász progression |
| Binary segmentation (imbalanced) | IoU | Tversky(α=0.7) + Focal |
| Thin tubular structures | IoU | Dice + SkeletonRecall |
| Multi-class segmentation | mIoU | Lovász-Softmax |
| Binary classification (imbalanced) | AUC | FocalLoss (γ=2) |
| Multi-label classification (imbalanced) | F1 | FocalLoss + Lovász |
| 3D medical segmentation | Dice | FocalTversky++ |

## Related
- [[segmentation-architectures]] — which architectures pair with these losses
- [[pseudo-labeling-cv]] — BCE→Lovász progression across pseudo-label stages
- [[image-classification-tricks]] — classification losses

## Sources
- `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` — BCE→Lovász progression, Dice
- `raw/kaggle/solutions/rsna-aneurysm-1st-tomoon33.md` — SkeletonRecall, FocalTversky++
- `raw/kaggle/solutions/human-protein-atlas-1st-bestfitting.md` — FocalLoss + Lovász for multi-label
