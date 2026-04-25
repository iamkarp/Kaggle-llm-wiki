---
id: concept:segmentation-architectures
type: concept
title: Segmentation Architectures
slug: segmentation-architectures
aliases: []
tags:
- cv
- segmentation
- unet
- scse
- fpa
- nnunet
- coarse-to-fine
- decoder
- attention
status: active
date: 2026-04-14
source_count: 2
---

# Segmentation Architectures

Medical and competition segmentation relies heavily on U-Net variants. This page covers the standard U-Net decoder enhancements, nnU-Net's auto-configuration approach, and coarse-to-fine pipelines for 3D medical imaging.

## U-Net Baseline

Standard U-Net: encoder (pretrained backbone) + skip connections + decoder. The encoder extracts features at multiple scales; skip connections preserve spatial detail; the decoder upsamples back to full resolution.

```
Encoder:    [conv→pool] × N levels → bottleneck
Skip:        copy encoder features at each level
Decoder:    [upsample + concat(skip) + conv] × N levels → output
```

For competitions, use pretrained ImageNet encoders (ResNet, EfficientNet, SE-ResNeXt) rather than training encoder weights from scratch.

## scSE Attention in Decoder Blocks

Concurrent Spatial and Channel Squeeze-and-Excitation (scSE) applied at each decoder block:

```python
class scSEBlock(nn.Module):
    """Recalibrate features spatially AND channel-wise in parallel."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel SE: which features matter globally?
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        # Spatial SE: which spatial locations matter?
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        cse_weight = self.cse(x).view(x.size(0), x.size(1), 1, 1)
        sse_weight = self.sse(x)
        return x * cse_weight + x * sse_weight  # concurrent (not sequential)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        self.attention = scSEBlock(out_channels)

    def forward(self, x, skip):
        x = self.conv(x)
        x = torch.cat([x, skip], dim=1)
        return self.attention(x)
```

**scSE vs. standard SE**: Standard SE is channel-only. Spatial SE adds per-pixel gating. Concurrent application (sum, not product) gives the network more flexibility to ignore either type when not helpful.

**Source**: TGS Salt (Babakhin 2018). Now standard in most segmentation competitions.

## Feature Pyramid Attention (FPA) Center Block

Between encoder and decoder, FPA aggregates multi-scale features using multiple kernel sizes, gated by global context:

```python
class FPA(nn.Module):
    """Multi-scale pyramid attention between encoder and decoder."""
    def __init__(self, channels):
        super().__init__()
        # Global branch: full-image context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
        )
        # Multi-scale local branches
        self.branch1 = nn.Conv2d(channels, channels, 7, padding=3)  # large receptive field
        self.branch2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.branch3 = nn.Conv2d(channels, channels, 3, padding=1)  # small receptive field
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        global_ctx = F.interpolate(self.global_pool(x),
                                   size=x.shape[-2:], mode='bilinear')
        multi_scale = self.fuse(self.branch1(x) + self.branch2(x) + self.branch3(x))
        return multi_scale * global_ctx  # global context gates local features
```

**Role of FPA**: The bottleneck in U-Net only sees local features at the lowest resolution. FPA adds global context (from the 1×1 pooling branch) before the decoder begins upsampling. The element-wise multiplication gates local features by global context — if the global context says "no structure here", local features are suppressed.

**Source**: TGS Salt (Babakhin 2018).

## nnU-Net: Self-Configuring Medical Segmentation

nnU-Net (Isensee et al., 2020) automatically configures the entire U-Net training pipeline from a "dataset fingerprint":

```python
from nnunetv2.run.run_training import run_training

# All configuration is automatic — nnU-Net reads dataset.json
run_training(
    dataset_name_or_id="Dataset101_Aneurysm",
    configuration="3d_fullres",   # auto-selected based on voxel spacing
    fold=0,
)
```

**What nnU-Net auto-configures from dataset statistics**:
| Dataset Property | nnU-Net Decision |
|-----------------|-----------------|
| Median image shape | Patch size, batch size |
| Voxel spacing | Network depth (more pooling for large images) |
| Intensity statistics | Normalization strategy |
| Training set size | Augmentation intensity |
| GPU memory | Batch size cap |

**For 3D CTA (aneurysm detection)**:
- Patch size: 128×128×128
- Batch size: 2
- Network depth: 6 levels
- Normalization: clip [-1000, 1000] HU, z-score normalize foreground

**When to use nnU-Net**: Any medical imaging segmentation task. Start with nnU-Net as the baseline — it almost always beats naive U-Net implementations.

## Coarse-to-Fine 3D Pipeline

For large 3D medical volumes (512³ voxels ≈ 512MB per scan), full-resolution processing is infeasible:

```
Stage 1 — Coarse:
  Input: Downsampled volume (~3mm/voxel, 3× less data)
  Model: nnU-Net coarse vessel segmentation
  Output: Bounding box ROIs containing aneurysms (~5% of total volume)

Stage 2 — Fine:
  Input: ROI patches at full resolution (~0.5mm/voxel)
  Model: nnU-Net fine segmentation (vessels + aneurysms)
  Output: Precise vessel mask + aneurysm mask

Stage 3 — Classification:
  Input: Aneurysm ROI + vessel mask from Stage 2
  Model: Location-Aware Transformer
  Output: Rupture risk
```

**Computation savings**: Fine segmentation runs only on ~5% of the volume → 10-20× less computation. Mimics radiologist workflow (overview → zoom into suspicious region).

**Source**: RSNA Aneurysm (tomoon33 2025).

## Auxiliary Sphere Segmentation

Add an auxiliary task predicting a sphere centered on each aneurysm:

```python
# Multi-task loss: main segmentation + auxiliary sphere
loss = (dice_loss(pred_aneurysm, target_aneurysm) +
        0.3 * dice_loss(pred_sphere, target_sphere))  # sphere weight 0.3
```

**Why spheres**: Sphere annotations are auto-generated from center + size metadata. The sphere task is geometrically simpler, stabilizing early-training localization before the model learns exact boundaries.

**Generalizes to**: Any segmentation where object center/size is available — generate a sphere/ellipsoid auxiliary target from that metadata.

## Architecture Selection Guide

| Task | Recommended Architecture |
|------|--------------------------|
| 2D natural image segmentation | U-Net + pretrained ResNet/EfficientNet encoder + scSE |
| 3D medical segmentation | nnU-Net (coarse-to-fine for large volumes) |
| Large structure segmentation | FPA center block |
| Thin tubular structures | SkeletonRecall loss (see [[loss-functions-cv]]) |
| Multi-task (seg + classification) | Shared encoder, separate decoder and classifier heads |

## Related
- [[loss-functions-cv]] — Dice, SkeletonRecall, FocalTversky++ for segmentation
- [[pseudo-labeling-cv]] — 3-stage pseudo-labeling for segmentation
- [[image-augmentation]] — ElasticTransform for training segmentation models

## Sources
- `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` — scSE block, FPA, 2D U-Net pipeline
- `raw/kaggle/solutions/rsna-aneurysm-1st-tomoon33.md` — nnU-Net, coarse-to-fine 3D, auxiliary sphere task

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _works_with_ → [[concepts/image-augmentation|Image Augmentation]]
- _works_with_ → [[concepts/loss-functions-cv|Loss Functions for CV]]
- _related_to_ → [[concepts/pseudo-labeling-cv|Pseudo-Labeling for CV]]

**Incoming:**
- [[concepts/medical-imaging-patterns|Medical Imaging & Bioinformatics Competition Patterns]] _uses_ → here
- [[concepts/loss-functions-cv|Loss Functions for CV]] _works_with_ → here
- [[concepts/pseudo-labeling-cv|Pseudo-Labeling for CV]] _related_to_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
