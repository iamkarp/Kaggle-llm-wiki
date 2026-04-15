# SIIM-ISIC Melanoma Classification — 1st Place Solution
**Author**: Bo | **Year**: 2020 | **Votes**: 384

---

## Competition
Binary classification of melanoma (skin cancer) from dermoscopy images + patient metadata. AUC metric. ~33K training images with severe class imbalance (~1.76% positive). Key challenge: correlation between image size/source and label due to dataset curation artifacts.

## Model Ensemble Architecture

Final ensemble spans 5 backbone families across multiple input resolutions:

```
EfficientNet B3 (384×384)
EfficientNet B4 (384×384)
EfficientNet B5 (456×456)
EfficientNet B6 (528×528)
EfficientNet B7 (600×600)
SE-ResNeXt101 (384×384)
ResNeSt101 (384×384)

Ensemble: rank-based blend (all predictions converted to percentile rank before averaging)
```

Higher resolution = better performance but slower inference. B7 at 600×600 is the strongest single model. Ensembling across architectures + resolutions captures complementary failure modes.

## Diagnosis Cross-Entropy: Using 9-Class Labels as Target

The training data included `diagnosis` labels (9 categories: melanoma, nevus, seborrheic keratosis, etc.) even though the competition metric is binary (melanoma vs. not).

**Standard approach**: Train with binary cross-entropy on the melanoma label.

**Winning approach**: Train with 9-class cross-entropy on the `diagnosis` column, then convert the melanoma class probability to a binary prediction.

```python
# Instead of:
loss = F.binary_cross_entropy_with_logits(pred, is_melanoma)

# Use:
loss = F.cross_entropy(pred_9class, diagnosis_label)  # 9-way classification
# At inference:
melanoma_prob = F.softmax(pred_9class, dim=1)[:, MELANOMA_CLASS_IDX]
```

**Why this works (+~0.01 AUC)**:
- 9-class labels encode richer supervision: the model learns what makes each diagnosis distinct
- "Not melanoma" is not a monolithic class — seborrheic keratosis and nevus are very different
- Multi-class gradients provide more signal per image than binary (only 2% positive) gradients
- Forces the model to learn discriminative features for all skin conditions, not just melanoma vs. all

## External Data: 2018 + 2019 ISIC Archives

```python
# Training data composition:
# 2020 competition data: ~33K images (original)
# 2018 ISIC challenge: ~10K images
# 2019 ISIC challenge: ~25K images
# Total: ~68K images

# External data filtering: keep images that map cleanly to 2020 diagnosis labels
# Remove duplicate patient images across years (dedup by patient_id)
```

**Why external data stabilizes CV**:
- 2020 dataset has artifacts: image resolution correlates with malignancy (high-res cameras at specialized clinics see more melanoma)
- Adding 2018+2019 data reduces this correlation artifact
- More balanced distribution across image sources
- Better CV-LB alignment (without external data, CV is unreliable)

## Rank-Based Ensembling

Rather than averaging raw probabilities across models, convert each model's output to percentile rank before averaging:

```python
import numpy as np
from scipy.stats import rankdata

def rank_ensemble(predictions_list):
    """
    predictions_list: list of (N,) arrays, one per model
    Returns: (N,) ensemble prediction
    """
    ranked = []
    for preds in predictions_list:
        # Convert to percentile rank (0 to 1)
        r = rankdata(preds) / len(preds)
        ranked.append(r)
    
    # Average the ranks
    return np.mean(ranked, axis=0)

# Usage:
model_preds = [model1_probs, model2_probs, ..., model7_probs]
final_preds = rank_ensemble(model_preds)
```

**Why rank ensembling**:
- Removes scale differences between models (EfficientNet B7 may be more "confident" than B3)
- Robust to outlier predictions from individual models
- Particularly important when mixing architectures with different calibration characteristics
- Each model's rank distribution is uniform [0,1] — averaging uniform distributions is more stable than averaging miscalibrated probabilities

## Metadata Fusion

Patient metadata (age, sex, anatomical site) is fused into the model:

```python
class MetadataFusionHead(nn.Module):
    def __init__(self, img_features=2560, meta_dim=6, hidden=512):
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
            nn.Linear(hidden, 9),  # 9-class diagnosis
        )
    
    def forward(self, img_features, metadata):
        meta_encoded = self.meta_embed(metadata)
        combined = torch.cat([img_features, meta_encoded], dim=1)
        return self.fusion(combined)
```

Metadata features: age (normalized), sex (binary), anatomical site (one-hot, 6 categories).

**Impact**: Modest improvement (+0.003-0.005 AUC). Anatomical site is the most informative: certain body sites have different baseline melanoma rates.

## Albumentations Augmentation Pipeline

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
        A.CoarseDropout(max_holes=8, max_height=image_size//8,
                        max_width=image_size//8,
                        min_holes=1, fill_value=0, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_tta_transforms(image_size=384):
    """Test-Time Augmentation: 8 flips/rotations."""
    return [
        A.Compose([A.Resize(image_size, image_size),
                   A.Normalize(...), ToTensorV2()]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Resize(image_size, image_size),
                   A.Normalize(...), ToTensorV2()]),
        A.Compose([A.VerticalFlip(p=1.0), A.Resize(image_size, image_size),
                   A.Normalize(...), ToTensorV2()]),
        # ... 5 more flip/rotation combinations
    ]
```

TTA with 8 augmentations applied at inference: average over all 8 predictions per image.

## Key Takeaways
1. Diagnosis cross-entropy (9-class) outperforms binary BCE by ~0.01 AUC — richer supervision, better gradients
2. Rank-based ensembling before averaging normalizes scale across architectures and calibrations
3. External data (2018+2019 ISIC) stabilizes CV by reducing dataset-specific artifacts
4. Multi-resolution ensemble (B3→B7, 384→600px) captures complementary features
5. Metadata fusion (age, sex, site) provides modest but consistent lift (+0.003-0.005)
6. TTA with 8 flip/rotation augmentations at inference, averaged over all
7. The "use richer supervision target" lesson generalizes: always check if higher-cardinality labels are available
