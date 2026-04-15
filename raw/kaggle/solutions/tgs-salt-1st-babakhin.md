# TGS Salt Identification Challenge — 1st Place Solution
**Author**: Babakhin | **Votes**: 418

---

## Competition
Binary segmentation of salt deposits from seismic reflection images. 101×101 grayscale images with binary masks. IoU (Intersection over Union) metric with thresholding at multiple IoU levels. ~4000 training images.

## 3-Stage Pseudo-Labeling Pipeline

The solution iteratively improves by generating pseudo-labels on test data and retraining. Three complete rounds:

```
Stage 1:
  Train on 4000 labeled images
  → Models: ResNeXt50-UNet + ResNet34-UNet

Stage 2:
  Generate pseudo-labels on test set using Stage 1 ensemble
  Threshold: keep confident predictions (predicted IoU > 0.8)
  Train on 4000 labeled + ~N confident pseudo-labeled test images
  → New models: same architectures, better performance

Stage 3:
  Generate new pseudo-labels using Stage 2 ensemble (more accurate)
  Threshold: can afford to lower threshold (Stage 2 is more accurate)
  Train on 4000 labeled + ~M pseudo-labeled test images (M > N)
  → Final models: trained on ~7000-8000 images total
```

### Pseudo-Label Confidence Filtering

```python
def confident_pseudo_labels(model_ensemble, test_images,
                             iou_threshold=0.8, empty_threshold=0.5):
    """
    Keep pseudo-labels only when ensemble is highly confident.
    """
    keep_mask = []
    pseudo_labels = []
    
    with torch.no_grad():
        for img in test_images:
            pred = model_ensemble.predict(img)  # (H, W) probability map
            
            # Binarize
            binary_pred = (pred > 0.5).float()
            
            # Compute agreement across ensemble members
            member_preds = [(m.predict(img) > 0.5).float()
                           for m in model_ensemble.members]
            agreement = torch.stack(member_preds).float().mean(0)  # pixel-wise agree rate
            
            # Only keep if high agreement across ensemble
            confident = agreement.min() > iou_threshold  # all agree on every pixel
            keep_mask.append(confident)
            pseudo_labels.append(binary_pred)
    
    return pseudo_labels, keep_mask
```

## Loss Progression: BCE → Lovász

Rather than a single loss throughout training, progress from BCE to Lovász:

```python
def get_loss(epoch, total_epochs):
    """Switch from BCE to Lovász partway through training."""
    if epoch < total_epochs * 0.4:
        # Early training: BCE is more stable and faster to converge
        return BCEWithLogitsLoss()
    else:
        # Later training: Lovász directly optimizes IoU metric
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

**Why BCE first, Lovász later**:
- BCE provides stable gradients early in training when predictions are noisy
- Lovász can have poor gradient behavior from random initialization (too many empty-prediction local minima)
- After BCE warms up the model, Lovász fine-tunes toward the actual IoU metric

## scSE Attention in Decoder (Squeeze-and-Excitation)

Concurrent Spatial and Channel Squeeze-and-Excitation (scSE) applied at each decoder block:

```python
class scSEBlock(nn.Module):
    """Concurrent spatial + channel squeeze-excitation."""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel SE
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )
        # Spatial SE
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        # Channel recalibration
        cse_weight = self.cse(x).view(x.size(0), x.size(1), 1, 1)
        # Spatial recalibration
        sse_weight = self.sse(x)
        # Concurrent application
        return x * cse_weight + x * sse_weight

# Applied at each decoder block in U-Net
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

## Feature Pyramid Attention (FPA) Center Block

Between encoder and decoder, a Feature Pyramid Attention module aggregates multi-scale features:

```python
class FPA(nn.Module):
    """Feature Pyramid Attention center block."""
    def __init__(self, channels):
        super().__init__()
        # Global pooling branch: captures full-image context
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
        )
        # Multi-scale pyramid branches
        self.branch1 = nn.Conv2d(channels, channels, 7, padding=3)
        self.branch2 = nn.Conv2d(channels, channels, 5, padding=2)
        self.branch3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.fuse = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x):
        global_ctx = F.interpolate(self.global_pool(x),
                                    size=x.shape[-2:], mode='bilinear')
        p1 = self.branch1(x)
        p2 = self.branch2(x)
        p3 = self.branch3(x)
        fused = self.fuse(p1 + p2 + p3)
        return fused * global_ctx  # element-wise gating by global context
```

## Cosine Annealing with Snapshots

Cosine annealing LR schedule with multiple restarts (snapshot ensembling):

```python
from torch.optim.lr_scheduler import CosineAnnealingLR

# Restart every T_max epochs; save model at LR minimum (best generalization point)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

snapshot_models = []
for epoch in range(total_epochs):
    train_epoch(...)
    scheduler.step()
    
    # Save snapshot at cosine minimum (end of each cycle)
    if (epoch + 1) % T_max == 0:
        snapshot_models.append(copy.deepcopy(model.state_dict()))

# Ensemble all snapshots
final_ensemble = [load_snapshot(s) for s in snapshot_models]
```

Each snapshot captures the model at a different point in weight space (different basins of the loss landscape). Averaging snapshots = cheap ensemble from a single training run.

## Albumentations Augmentation

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                       rotate_limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.ElasticTransform(alpha=120, sigma=120*0.05,
                       alpha_affine=120*0.03, p=0.3),
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(distort_limit=0.5, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.3),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])
```

ElasticTransform and GridDistortion are especially important for segmentation — they simulate natural deformation of geological structures.

## Key Takeaways
1. 3-stage pseudo-labeling with increasing pseudo-label count: Stage 1 (4K) → Stage 2 (4K+N) → Stage 3 (4K+M) with M>N
2. BCE first then Lovász: stable early training → metric-aligned fine-tuning
3. scSE attention in each decoder block improves spatial and channel feature recalibration
4. FPA center block aggregates multi-scale context before decoding
5. Cosine annealing with snapshots provides free ensemble diversity from single training run
6. Ensemble agreement as pseudo-label confidence filter (not just single model confidence)
7. Albumentations ElasticTransform + GridDistortion simulate geological deformation realistically
