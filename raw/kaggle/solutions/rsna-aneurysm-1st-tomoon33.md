# RSNA Intracranial Aneurysm Detection and Classification — 1st Place Solution
**Author**: tomoon33 | **Year**: 2025 | **Votes**: 151

---

## Competition
Detect and classify intracranial aneurysms from 3D CTA (CT Angiography) scans. Tasks: vessel segmentation, aneurysm localization, rupture risk classification. Multi-task 3D medical imaging.

## Coarse-to-Fine Pipeline

```
Stage 1: Coarse — Low-resolution whole-volume vessel localization
  Input: Downsampled CTA volume (~3mm/voxel)
  Model: nnU-Net coarse vessel segmentation
  Output: Vessel probability map → bounding box regions of interest

Stage 2: Fine — High-resolution vessel + aneurysm segmentation  
  Input: ROI patches at full resolution (~0.5mm/voxel)
  Model: nnU-Net fine segmentation (vessels + aneurysms separately)
  Output: Fine vessel mask + aneurysm mask

Stage 3: Classification — Rupture risk from ROI features
  Input: Aneurysm ROI + vessel context
  Model: Location-Aware Transformer + Vessel Region-Masked Pooling
  Output: Rupture risk (binary)
```

**Why coarse-to-fine**:
- Full-resolution 3D CTA is enormous (512³ voxels × float32 ≈ 512MB per scan)
- Coarse pass identifies the small region containing aneurysms (~5% of volume)
- Fine segmentation runs only on the ROI — 10-20× less computation
- Mimics radiologist workflow (scan overview → zoom into suspicious region)

## nnU-Net: Self-Configuring Medical Image Segmentation

nnU-Net (Isensee et al., 2020) automatically configures the U-Net architecture and training pipeline for any given medical imaging dataset:

```python
# nnU-Net auto-configures based on dataset fingerprint:
# - Median image shape → determines patch size, batch size
# - Voxel spacing → determines network depth (more pooling for small voxels)
# - Intensity statistics → determines normalization (CT-specific: clip to body HU range)
# - Training set size → determines augmentation intensity

from nnunetv2.run.run_training import run_training

# Configuration is automatic — nnU-Net reads dataset.json
run_training(
    dataset_name_or_id="Dataset101_Aneurysm",
    configuration="3d_fullres",   # auto-selected based on voxel size
    fold=0,
)
```

**nnU-Net's auto-configured choices for this dataset**:
- Patch size: 128×128×128 (from median aneurysm + context size)
- Batch size: 2 (GPU memory constraint)
- Network depth: 6 levels (from voxel spacing)
- Normalization: CT-specific (clip [-1000, 1000] HU, z-score per foreground)
- Augmentation: rotation, scaling, Gaussian noise, mirroring, elastic deformation

## Location-Aware Transformer for Classification

Standard Vision Transformer treats all spatial positions equally. For aneurysm rupture risk, **location within the brain vasculature** is clinically critical (anterior communicating artery aneurysms have different rupture rates than posterior inferior cerebellar artery aneurysms).

```python
class LocationAwareTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.patch_embed = nn.Conv3d(1, d_model, kernel_size=8, stride=8)
        # Location encoding: learnable per-position embedding
        self.location_embed = nn.Parameter(
            torch.randn(1, 1000, d_model)  # 1000 possible patch positions
        )
        # Standard transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024,
                                       dropout=0.1, batch_first=True),
            num_layers=num_layers
        )
        self.classifier = nn.Linear(d_model, 2)  # ruptured / unruptured
    
    def forward(self, x, vessel_mask):
        # x: (B, 1, D, H, W) aneurysm ROI
        patches = self.patch_embed(x)  # (B, d_model, d', h', w')
        B, C, D, H, W = patches.shape
        patches = patches.flatten(2).transpose(1, 2)  # (B, N, d_model)
        
        # Add location-aware positional encoding
        patches = patches + self.location_embed[:, :patches.shape[1], :]
        
        # Vessel-masked pooling (see below) replaces naive CLS token
        output = self.transformer(patches)
        return self.classifier(output.mean(1))
```

## Vessel Region-Masked Pooling

Instead of pooling over all spatial positions, pool only over positions that contain vessel tissue (identified by Stage 2 segmentation):

```python
def vessel_masked_pooling(features, vessel_mask):
    """
    Pool transformer features only from voxels within vessel region.
    
    features: (B, N, d_model) — transformer output
    vessel_mask: (B, N) — binary mask, 1 = vessel voxel
    """
    # Expand mask for broadcasting
    mask = vessel_mask.unsqueeze(-1).float()  # (B, N, 1)
    
    # Masked mean pooling
    masked_sum = (features * mask).sum(dim=1)   # (B, d_model)
    mask_count = mask.sum(dim=1).clamp(min=1)   # (B, 1)
    return masked_sum / mask_count
```

**Why masked pooling**: Features from non-vessel regions (brain parenchyma, skull) are noise for rupture prediction. Restricting pooling to the vessel region forces the classifier to focus on vascular structure.

## SkeletonRecall Loss for Thin Vessel Connectivity

Standard Dice loss struggles with thin, tubular structures (vessels can be 1-2mm diameter). A vessel that is nearly perfectly segmented except for one broken connection scores poorly on Dice because the centerline is severed.

SkeletonRecall loss adds a term specifically rewarding correct centerline prediction:

```python
class SkeletonRecallLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # weight of skeleton recall term
    
    def compute_skeleton(self, mask):
        """Compute morphological skeleton (centerline) of binary mask."""
        # Using skimage.morphology.skeletonize
        from skimage.morphology import skeletonize
        skeleton = np.zeros_like(mask)
        for b in range(mask.shape[0]):
            skeleton[b] = skeletonize(mask[b].cpu().numpy())
        return torch.tensor(skeleton, device=mask.device)
    
    def forward(self, pred, target):
        # Dice loss term
        dice = 1 - dice_coefficient(pred, target)
        
        # Skeleton recall term: how many skeleton voxels are correctly predicted?
        target_skeleton = self.compute_skeleton(target > 0.5)
        pred_binary = (pred > 0.5).float()
        skeleton_recall = (pred_binary * target_skeleton).sum() / (target_skeleton.sum() + 1e-6)
        skeleton_loss = 1 - skeleton_recall
        
        return (1 - self.alpha) * dice + self.alpha * skeleton_loss
```

**When to use**: Any segmentation task with thin, tubular, or tree-like structures where connectivity matters more than pixel count (vessels, nerves, roads, rivers).

## Auxiliary Sphere Segmentation Task

Rather than only predicting the aneurysm mask (irregular shape, hard to learn), add an auxiliary task: predict a sphere centered on each aneurysm with radius = aneurysm diameter.

```python
# Multi-task loss
loss = (dice_loss(pred_aneurysm, target_aneurysm) +
        0.3 * dice_loss(pred_sphere, target_sphere))  # auxiliary sphere task
```

**Why spheres**: The sphere annotation is easy to generate automatically from aneurysm center + size metadata. The sphere prediction task gives the model a simpler geometric target that stabilizes early training and helps the model localize aneurysms before learning exact boundaries.

## Focal-Tversky++ Loss

Extension of Tversky loss with per-class weighting and focal component:

```python
class FocalTverskyPlusPlus(nn.Module):
    """
    Tversky loss with focal weighting and class-specific alpha/beta.
    alpha: weight of false negatives (higher = more penalty for missing positives)
    beta: weight of false positives
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

Tversky with `alpha=0.7, beta=0.3`: penalizes false negatives (missing an aneurysm) more than false positives — medically appropriate.

## Key Takeaways
1. Coarse-to-fine pipeline matches radiologist workflow and reduces computation 10-20×
2. nnU-Net auto-configures everything from dataset fingerprint — use it as the segmentation baseline for any medical imaging task
3. Location-Aware Transformer incorporates spatial context critical for clinical classification
4. Vessel Region-Masked Pooling restricts attention to clinically relevant voxels
5. SkeletonRecall loss penalizes connectivity breaks in thin tubular structures
6. Auxiliary sphere segmentation stabilizes localization learning before fine boundary prediction
7. Focal-Tversky++ with alpha=0.7 prioritizes false negative reduction for medical safety
