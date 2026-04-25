---
id: concept:medical-imaging-patterns
type: concept
title: Medical Imaging & Bioinformatics Competition Patterns
slug: medical-imaging-patterns
aliases: []
tags:
- medical-imaging
- cv
- unet
- rsna
- isic
- 2.5d
- foundation-model
- pathology
- bioinformatics
status: active
date: 2026-04-15
source_count: 7
---

## Summary

Medical imaging competitions have domain-specific winning patterns: CLAHE preprocessing (consistent +1-3%), 2.5D slice stacking for 3D CT (enables ImageNet pretraining), two-stage detect-then-classify for RSNA-style competitions, and domain foundation models (Phikon for pathology, EVA02 for dermoscopy). Pseudo-labeling is the #1 technique in BirdCLEF-style audio competitions.

## What It Is

Patterns and architectures extracted from ISIC (skin lesion), RSNA (radiology), UBC-OCEAN (pathology), BirdCLEF (audio), and BELKA (drug discovery) competitions.

## Key Facts / Details

### ISIC — Skin Lesion Classification

**Architecture:** EVA02-Large (CLIP-based ViT) + 45 GBDT models on extracted features.

**Stable Diffusion for rare class augmentation:**
```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)

def augment_rare_subtype(image_path, prompt, strength=0.3, n_samples=5):
    init_image = load_image(image_path).resize((512, 512))
    return pipe(prompt=prompt, image=init_image, strength=strength,
                guidance_scale=7.5, num_images_per_prompt=n_samples).images
```

**CLAHE preprocessing (always use for skin lesions):**
```python
def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

**Hair removal (skin-specific):**
```python
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
```

### RSNA — Two-Stage Approach (Standard Pattern)

**Stage 1 — Keypoint/Region Detection:**
```python
class KeypointDetector(nn.Module):
    def __init__(self, backbone='efficientnet_b4'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, 5 * 2)  # 5 keypoints × (x,y)
    
    def forward(self, x):
        return self.head(self.backbone(x)).view(-1, 5, 2)
```

**Stage 2 — Extract patches → classify severity per region.**

Multi-view fusion (Sagittal T1, Sagittal T2, Axial T2): attention-weighted fusion across views.

### 2.5D Slice Stacking (3D CT Competitions)

**Problem:** 3D CT volumes too large for GPU. 2D CNNs miss 3D context.
**Solution:** Stack adjacent slices as RGB channels.

```python
def create_rgb_2_5d(volume, slice_idx):
    prev = volume[max(0, slice_idx - 1)]
    curr = volume[slice_idx]
    next_ = volume[min(volume.shape[0]-1, slice_idx + 1)]
    return np.stack([prev, curr, next_], axis=-1)  # (H, W, 3) → standard RGB CNN
```

**Why it works:** Standard ImageNet-pretrained CNNs can be used directly; captures local 3D context.
Used in RSNA 2023 Abdominal Trauma Detection.

### Focal-Tversky++ Loss (Small Structure Segmentation)

For small medical structures with extreme foreground/background imbalance:

```python
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.alpha = alpha  # weight FN (miss rate)
        self.beta = beta    # weight FP
        self.gamma = gamma  # focal component
    
    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        tp = (preds * targets).sum(dim=(2,3,4))
        fn = ((1-preds) * targets).sum(dim=(2,3,4))
        fp = (preds * (1-targets)).sum(dim=(2,3,4))
        tversky = (tp + 1e-6) / (tp + self.alpha*fn + self.beta*fp + 1e-6)
        return ((1 - tversky) ** (1/self.gamma)).mean()
```

Used in RSNA Aneurysm Detection 1st place.

### Pathology: Foundation Model + MIL

**Phikon** (owkin/phikon): pathology-specific ViT pretrained on 40M patches.

```python
from transformers import AutoModel

class PhikonMIL(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('owkin/phikon')
        self.attention = nn.Linear(768, 1)
        self.classifier = nn.Linear(768, n_classes)
    
    def forward(self, bag):  # bag: (n_patches, 3, 224, 224)
        features = self.encoder(bag).last_hidden_state[:, 0]  # CLS tokens
        scores = self.attention(features).squeeze(-1)
        K = 10
        top_idx = scores.topk(K).indices
        bot_idx = scores.topk(K, largest=False).indices
        aggregated = torch.cat([features[top_idx], features[bot_idx]]).mean(0)
        return self.classifier(aggregated)
```

**MIL paradigm:** Each patient = a "bag" of patches. Label is bag-level.

### BirdCLEF — Pseudo-Labeling as #1 Technique

```python
# Standard pseudo-labeling with confidence filtering
pseudo_probs = model.predict_proba(X_unlabeled)
high_conf_mask = pseudo_probs.max(axis=1) > 0.85
X_combined = np.vstack([X_labeled, X_unlabeled[high_conf_mask]])
y_combined = np.concatenate([y_labeled, pseudo_probs[high_conf_mask].argmax(axis=1)])
```

**Max-label MixUp for multi-label audio:**
```python
def max_label_mixup(audio1, label1, audio2, label2, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    mixed_audio = lam * audio1 + (1 - lam) * audio2
    mixed_label = np.maximum(label1, label2)  # KEY: max, not weighted average
    return mixed_audio, mixed_label
```

**External data:** xeno-canto.org — standard approved source for bird audio.

### BELKA Drug Discovery — SMILES Pretraining

```python
from transformers import RobertaForMaskedLM, RobertaConfig

# Stage 1: pretrain on SMILES as language model
config = RobertaConfig(vocab_size=600, hidden_size=512, num_hidden_layers=6)
smiles_lm = RobertaForMaskedLM(config)  # Train on ChEMBL/ZINC SMILES

# Stage 2: fine-tune on binding affinity
class BindingPredictor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.roberta
        self.head = nn.Linear(512, 3)  # 3 binding targets
```

**Molecular fingerprints as supplementary features:**
```python
from rdkit.Chem import AllChem
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
```

### Universal Medical Patterns Table

| Technique | Competitions | Impact |
|---|---|---|
| CLAHE preprocessing | Skin lesion, chest X-ray | Consistent +1-3% |
| 2.5D slice stacking | All 3D CT | Enables ImageNet pretraining |
| Two-stage (detect→classify) | RSNA competitions | Handles class imbalance |
| Foundation model (Phikon, EVA02) | Pathology, dermoscopy | +5-15% vs scratch |
| Focal-Tversky loss | Small structure segmentation | Critical for rare findings |
| Pseudo-labeling | BirdCLEF audio | #1 technique |
| Max-label MixUp | Multi-label audio | Better than standard MixUp |
| SMILES pretraining | Drug discovery | +10-20% vs fingerprints |
| Stable Diffusion augmentation | ISIC rare subtypes | Class imbalance correction |

## Gotchas

- CLAHE clip_limit=2.0, tile_grid=(8,8) is the standard starting point — competition-specific tuning rarely needed
- 2.5D requires preprocessing: window/level normalization for CT before slice stacking
- MIL with Phikon requires GPU with >16GB VRAM for reasonable batch sizes
- BirdCLEF pseudo-label confidence threshold 0.85 is a common baseline — too low = noisy labels

## Sources

- `raw/kaggle/medical-bioinformatics-solutions.md` *(not yet ingested)* — full reference with all code
- [ISIC 2024 competition](https://www.kaggle.com/competitions/isic-2024-challenge/)
- [nnU-Net paper](https://www.nature.com/articles/s41592-020-01008-z)
- [Phikon pathology model](https://huggingface.co/owkin/phikon)
- [BirdCLEF 2024](https://www.kaggle.com/competitions/birdclef-2024)
- [BELKA drug discovery](https://www.kaggle.com/competitions/leash-BELKA)

## Related

- [[loss-functions-cv]] — Lovász, Focal-Tversky, loss function guide
- [[segmentation-architectures]] — UNet, nnU-Net, coarse-to-fine
- [[image-augmentation]] — Albumentations, TTA, CycleGAN
- [[pseudo-labeling-cv]] — 3-stage segmentation pseudo-labeling
- [[metric-learning-cv]] — ArcFace for rare classes

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[concepts/segmentation-architectures|Segmentation Architectures]]
- _works_with_ → [[concepts/metric-learning-cv|Metric Learning for CV]]
- _related_to_ → [[concepts/image-augmentation|Image Augmentation]]
- _related_to_ → [[concepts/loss-functions-cv|Loss Functions for CV]]
- _related_to_ → [[concepts/pseudo-labeling-cv|Pseudo-Labeling for CV]]

**Incoming:**
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
