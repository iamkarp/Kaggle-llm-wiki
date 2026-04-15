# Medical Imaging & Bioinformatics Kaggle Solutions

Compiled from ISIC, RSNA, UBC-OCEAN, BirdCLEF, BELKA competition writeups. April 2026.

---

## ISIC 2024 Skin Lesion Classification — Top Solutions

**Key architecture:** EVA02-Large (OpenAI CLIP-based ViT) as feature extractor.

**Winning ensemble structure:**
- 10 ViT models (EVA02, ConvNeXt-L, EfficientNet-V2) → feature fusion
- 45 GBDT models on extracted features + metadata
- Total: 55-model ensemble with hill climbing weights

**Stable Diffusion for minority augmentation:**
```python
# Generate synthetic lesion images for rare subtypes
from diffusers import StableDiffusionImg2ImgPipeline

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)

def augment_rare_subtype(image_path, prompt, strength=0.3, n_samples=5):
    """
    strength=0.3: moderate augmentation (preserves original structure)
    """
    init_image = load_image(image_path).resize((512, 512))
    images = pipe(
        prompt=prompt,
        image=init_image,
        strength=strength,
        guidance_scale=7.5,
        num_images_per_prompt=n_samples,
    ).images
    return images

# Example: augment melanoma class (minority)
prompt = "dermoscopic image of melanoma skin lesion, high quality medical photography"
synthetic_images = augment_rare_subtype('melanoma_001.jpg', prompt)
```

**CLAHE preprocessing (standard for skin lesions):**
```python
import cv2
import numpy as np

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
```

**Hair removal preprocessing (skin-specific):**
```python
def remove_hair(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)
```

---

## RSNA 2024 Lumbar Spine Degenerative Classification

**Two-stage approach:**
1. **Keypoint detection** (Stage 1): Detect vertebral levels + spinal canal landmarks
2. **Classification** (Stage 2): Classify severity (Normal/Mild/Moderate/Severe) per level

```python
# Stage 1: Keypoint detection with ViT
class LumbarKeypointDetector(nn.Module):
    def __init__(self, backbone='efficientnet_b4'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        n_features = self.backbone.num_features
        # Output: (x, y) coordinates for each of 5 vertebral levels × 3 views
        self.head = nn.Linear(n_features, 5 * 3 * 2)
    
    def forward(self, x):
        features = self.backbone(x)
        keypoints = self.head(features).view(-1, 5, 3, 2)
        return keypoints

# Stage 2: Extract patches around keypoints → classify
def extract_patches(volume, keypoints, patch_size=64):
    patches = []
    for level_kp in keypoints:
        for view_kp in level_kp:
            x, y = int(view_kp[0]), int(view_kp[1])
            patch = volume[y-patch_size//2:y+patch_size//2, 
                          x-patch_size//2:x+patch_size//2]
            patches.append(patch)
    return patches
```

**Multi-view fusion:** Sagittal T1, Sagittal T2, Axial T2 — attention-weighted fusion across views.

---

## RSNA Intracranial Aneurysm Detection — 3D nnU-Net Approach

**Architecture:**
1. **3D nnU-Net** for segmentation (fully automated hyperparameter optimization)
2. **Location-Aware Transformer** for classification using segmentation mask + 3D coordinates
3. **Focal-Tversky++ loss** for extreme foreground/background imbalance

```python
# Focal-Tversky loss for small medical structures
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
        focal_tversky = (1 - tversky) ** (1/self.gamma)
        return focal_tversky.mean()
```

**3D nnU-Net key advantage:** Automatically determines optimal patch size, architecture, and normalization for any medical imaging dataset.

---

## RSNA 2023 Abdominal Trauma Detection — 2.5D Slice Stacking

**Problem:** 3D CT volumes are too large for GPU. 2D CNNs miss 3D context.

**2.5D solution:**
```python
def create_2_5d_slice(volume, slice_idx, n_adjacent=2):
    """
    Stack adjacent slices as RGB channels.
    2 adjacent slices on each side → 5-channel input (treated as RGB+2 extra channels).
    """
    slices = []
    for offset in range(-n_adjacent, n_adjacent + 1):
        idx = max(0, min(slice_idx + offset, volume.shape[0] - 1))
        slices.append(volume[idx])
    return np.stack(slices, axis=-1)  # (H, W, 5)

# For pure 2.5D (3-channel RGB-equivalent):
def create_rgb_2_5d(volume, slice_idx):
    prev = volume[max(0, slice_idx - 1)]
    curr = volume[slice_idx]
    next_ = volume[min(volume.shape[0]-1, slice_idx + 1)]
    return np.stack([prev, curr, next_], axis=-1)  # (H, W, 3) → use standard RGB CNN
```

**Why 2.5D works:** Standard ImageNet-pretrained CNNs can be used directly on 3-channel input, while capturing local 3D context. Massive pretrained backbone advantage.

**Full 3D processing:** Use only for final refinement on flagged suspicious regions (too slow for full volume).

---

## UBC Ovarian Cancer Subtype Classification (UBC-OCEAN) — Foundation Model + MIL

**Approach:** Pathology foundation model (Phikon) + Multiple Instance Learning (MIL) head

**Phikon:** Pathology-specific ViT pretrained on 40M pathology patches. Direct drop-in encoder for WSI (Whole Slide Image) problems.

```python
# Phikon + Chowder MIL head
from transformers import AutoModel

class PhikonMIL(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        # Phikon: pathology foundation model
        self.encoder = AutoModel.from_pretrained('owkin/phikon')
        feature_dim = 768
        
        # Chowder aggregation: take top-K and bottom-K bag scores
        self.attention = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, n_classes)
    
    def forward(self, bag):  # bag: (n_patches, 3, 224, 224)
        # Extract patch features
        features = self.encoder(bag).last_hidden_state[:, 0]  # CLS tokens
        
        # Chowder: top-K + bottom-K selection
        scores = self.attention(features).squeeze(-1)
        K = 10
        top_idx = scores.topk(K).indices
        bot_idx = scores.topk(K, largest=False).indices
        selected = torch.cat([features[top_idx], features[bot_idx]])
        
        # Aggregate and classify
        aggregated = selected.mean(0)
        return self.classifier(aggregated)
```

**MIL paradigm:** Each patient = a "bag" of patches. Label is bag-level (patient has cancer subtype X). Model learns which patches are most discriminative.

---

## BirdCLEF 2024 — Audio Species Classification

**#1 technique: Pseudo-labeling**

```python
# BirdCLEF pseudo-labeling workflow
# Step 1: Train on labeled data
model.fit(X_labeled, y_labeled)

# Step 2: Generate pseudo-labels on unlabeled audio (xeno-canto external data)
pseudo_probs = model.predict_proba(X_unlabeled)
confidence_mask = pseudo_probs.max(axis=1) > 0.85  # high-confidence only
X_pseudo = X_unlabeled[confidence_mask]
y_pseudo = pseudo_probs[confidence_mask].argmax(axis=1)

# Step 3: Retrain on labeled + pseudo-labeled
X_combined = np.vstack([X_labeled, X_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])
model.fit(X_combined, y_combined)
```

**External data: xeno-canto:** Freely available bird audio recordings. Standard augmentation source for BirdCLEF.

**Max-label MixUp for multi-label audio:**
```python
def max_label_mixup(audio1, label1, audio2, label2, alpha=0.4):
    """For multi-label: use element-wise max of labels (not linear interpolation)"""
    lam = np.random.beta(alpha, alpha)
    mixed_audio = lam * audio1 + (1 - lam) * audio2
    mixed_label = np.maximum(label1, label2)  # KEY: max, not weighted average
    return mixed_audio, mixed_label
```

**Why max-label:** For rare species audio, linear interpolation of labels creates ambiguous weak labels. Max preserves both species' presence.

**Spectrogram features (standard):**
```python
import librosa

def extract_mel_spectrogram(audio, sr=32000, n_mels=128, hop_length=512):
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length,
        fmin=50, fmax=14000
    )
    return librosa.power_to_db(mel_spec, ref=np.max)
```

---

## BELKA Drug Discovery — SMILES Pretraining

**Task:** Predict binding affinity of small molecules.

**Two-stage approach:**
1. **Pretraining on SMILES** (Stage 1): Masked token prediction on SMILES strings → ChemBERTa-style LM
2. **Fine-tuning on binding labels** (Stage 2): Add regression/classification head

```python
# Stage 1: Pretrain on SMILES as language model
from transformers import RobertaForMaskedLM, RobertaConfig

config = RobertaConfig(
    vocab_size=600,  # SMILES token vocabulary
    hidden_size=512,
    num_hidden_layers=6,
    num_attention_heads=8,
)
smiles_lm = RobertaForMaskedLM(config)
# Train on 100M+ SMILES from ChEMBL/ZINC databases

# Stage 2: Fine-tune on binding affinity
class BindingPredictor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.encoder = pretrained_model.roberta
        self.head = nn.Linear(512, 3)  # 3 protein binding targets
    
    def forward(self, input_ids, attention_mask):
        features = self.encoder(input_ids, attention_mask).last_hidden_state[:, 0]
        return self.head(features)
```

**Molecular fingerprints as supplementary features:**
```python
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp)
```

---

## Universal Medical Imaging Patterns

| Technique | Competition | Impact |
|---|---|---|
| CLAHE preprocessing | Skin lesion, chest X-ray | Consistent +1-3% |
| 2.5D slice stacking | All 3D CT competitions | Enables ImageNet pretraining |
| Two-stage (detect→classify) | RSNA competitions | Handles class imbalance |
| Foundation model (Phikon, EVA02) | Pathology, general CV | +5-15% vs training from scratch |
| Focal-Tversky loss | Small structure segmentation | Critical for rare findings |
| Pseudo-labeling | Audio (BirdCLEF) | #1 technique |
| Max-label MixUp | Multi-label audio | Better than standard MixUp |
| SMILES pretraining | Drug discovery | +10-20% vs fingerprints alone |

---

Sources:
- ISIC 2024 top solutions: https://www.kaggle.com/competitions/isic-2024-challenge/discussion
- RSNA 2024 Lumbar: https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification
- nnU-Net paper: https://www.nature.com/articles/s41592-020-01008-z
- Phikon pathology model: https://huggingface.co/owkin/phikon
- BirdCLEF 2024: https://www.kaggle.com/competitions/birdclef-2024
- BELKA drug discovery: https://www.kaggle.com/competitions/leash-BELKA
- Stable Diffusion for augmentation: https://arxiv.org/abs/2305.16807
