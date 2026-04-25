---
id: concept:image-augmentation
type: concept
title: Image Augmentation
slug: image-augmentation
aliases: []
tags:
- cv
- augmentation
- albumentations
- tta
- cyclegan
- synthetic-data
- autoaugment
status: active
date: 2026-04-14
source_count: 4
---

# Image Augmentation

Image augmentation is the most reliable regularization technique in CV competitions. This page covers standard Albumentations recipes, Test-Time Augmentation (TTA), AutoAugment policy transfer, and CycleGAN synthetic data for unseen classes.

## Standard Albumentations Pipelines

### Classification (dermoscopy / natural images)
```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms(image_size=384):
    return A.Compose([
        A.RandomResizedCrop(image_size, image_size, scale=(0.7, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=45, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                             val_shift_limit=10, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15,
                                   contrast_limit=0.15, p=0.5),
        A.Cutout(num_holes=8, max_h_size=image_size//8,
                 max_w_size=image_size//8, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
```

### Segmentation (geological / salt structures)
```python
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=30, p=0.5),
    A.ElasticTransform(alpha=120, sigma=120*0.05,
                       alpha_affine=120*0.03, p=0.3),  # key for geological
    A.GridDistortion(p=0.3),           # simulate seismic deformation
    A.OpticalDistortion(distort_limit=0.5, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])
```

**ElasticTransform + GridDistortion** are especially important for segmentation — they simulate natural deformation of structures (geological layers, tissue boundaries, vessel paths). Use them for any segmentation task.

### Key Augmentation Principles
- For medical imaging: ElasticTransform simulates anatomical variation
- For satellite/aerial: no HueSaturation (multispectral); focus on rotations + flips
- Cutout/CoarseDropout forces models to not rely on any single image region
- Always apply geometric augmentations before photometric ones

## Test-Time Augmentation (TTA)

Apply multiple augmentations at inference and average predictions:

```python
def tta_predict(model, image, n_augmentations=8):
    """
    Apply TTA with horizontal/vertical flips and 90° rotations.
    """
    augmentations = [
        lambda x: x,                          # original
        lambda x: torch.flip(x, [3]),         # horizontal flip
        lambda x: torch.flip(x, [2]),         # vertical flip
        lambda x: torch.rot90(x, 1, [2, 3]),  # 90°
        lambda x: torch.rot90(x, 2, [2, 3]),  # 180°
        lambda x: torch.rot90(x, 3, [2, 3]),  # 270°
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [3]),  # 90° + hflip
        lambda x: torch.flip(torch.rot90(x, 1, [2, 3]), [2]),  # 90° + vflip
    ][:n_augmentations]

    preds = []
    with torch.no_grad():
        for aug in augmentations:
            augmented = aug(image.unsqueeze(0))
            pred = model(augmented)
            # Reverse geometric transform on prediction (for segmentation)
            preds.append(pred.squeeze(0))

    return torch.stack(preds).mean(0)
```

**TTA guidelines**:
- D4 group (flips + rotations): 8 augmentations; works for any rotationally symmetric problem
- For segmentation, reverse geometric transforms on the mask prediction before averaging
- Start with just H-flip (2× cost, ~0.002-0.005 gain), then add more if inference budget allows
- Scale TTA (multiple crop sizes) is expensive but strong for classification

## AutoAugment Policy Transfer

AutoAugment learns optimal augmentation policies for specific datasets. When no AutoAugment policy exists for your domain, use the closest proxy:

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

# For handwritten character recognition → SVHN policy (digit recognition from images)
AutoAugment(policy=AutoAugmentPolicy.SVHN)

# For natural image classification → ImageNet policy
AutoAugment(policy=AutoAugmentPolicy.IMAGENET)

# For CIFAR-like small images → CIFAR10 policy
AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
```

**Policy selection**: SVHN transfers well to handwritten text/characters (Bengali graphemes). ImageNet policy generalizes broadly but may be suboptimal for domain-specific tasks.

## CycleGAN Synthetic Data for Zero-Shot Learning

When classes exist in theory but have no training examples, use CycleGAN to generate synthetic training data:

```
Domain H: real handwritten images (training set)
Domain F: clean font-rendered images (all classes, including unseen)

CycleGAN learns bidirectional translation:
  G_HF: Handwritten → Font style
  G_FH: Font style → Handwritten style

Apply G_FH to font images of unseen classes:
  → "Handwritten-style" synthetic training images for unseen classes
```

```python
from PIL import ImageFont, ImageDraw, Image

def render_from_font(character_unicode, font_path, size=128):
    """Generate clean synthetic image from TTF font file."""
    font = ImageFont.truetype(font_path, size=64)
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    bbox = draw.textbbox((0, 0), character_unicode, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2
    y = (size - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), character_unicode, font=font, fill=0)
    return np.array(img)

# Render from multiple fonts for diversity
for font_path in available_fonts:
    synthetic_images[class_id].append(render_from_font(unicode_char, font_path))
```

**When font files are available** (character recognition, symbol classification), they provide free labeled synthetic data for all classes. The CycleGAN step makes the synthetic images look like the real domain.

## nnU-Net Augmentation (3D Medical Imaging)

For 3D medical volumes, nnU-Net auto-configures augmentation based on dataset fingerprint:
- Rotation (all three axes), scaling, Gaussian noise
- Mirroring (axis-specific, respects anatomical symmetry)
- Elastic deformation (simulates anatomical variation)
- CT-specific: clip HU range [-1000, 1000], z-score normalize foreground voxels

## Related
- [[image-classification-tricks]] — architecture tricks that complement augmentation
- [[pseudo-labeling-cv]] — synthetic pseudo-labels as augmentation variant
- [[metric-learning-cv]] — CycleGAN provides training data for zero-shot metric learning
- [[loss-functions-cv]] — FocalLoss to handle imbalance alongside augmentation

## Sources
- `raw/kaggle/solutions/siim-isic-melanoma-1st-bo.md` — Albumentations classification pipeline, 8-way TTA
- `raw/kaggle/solutions/tgs-salt-1st-babakhin.md` — ElasticTransform+GridDistortion for segmentation
- `raw/kaggle/solutions/bengali-grapheme-1st-deoxy.md` — CycleGAN + font rendering for zero-shot; SVHN AutoAugment
- `raw/kaggle/solutions/rsna-aneurysm-1st-tomoon33.md` — nnU-Net 3D augmentation pipeline

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _works_with_ → [[concepts/image-classification-tricks|Image Classification Tricks]]
- _works_with_ → [[concepts/loss-functions-cv|Loss Functions for CV]]
- _works_with_ → [[concepts/metric-learning-cv|Metric Learning for CV]]
- _works_with_ → [[concepts/pseudo-labeling-cv|Pseudo-Labeling for CV]]

**Incoming:**
- [[concepts/image-classification-tricks|Image Classification Tricks]] _uses_ → here
- [[concepts/metric-learning-cv|Metric Learning for CV]] _works_with_ → here
- [[concepts/segmentation-architectures|Segmentation Architectures]] _works_with_ → here
- [[concepts/tabular-augmentation|Tabular Data Augmentation Techniques]] _works_with_ → here
- [[concepts/medical-imaging-patterns|Medical Imaging & Bioinformatics Competition Patterns]] _related_to_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
