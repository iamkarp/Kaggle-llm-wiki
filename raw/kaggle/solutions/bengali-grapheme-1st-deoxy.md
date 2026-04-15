# Bengali.AI Handwritten Grapheme Classification — 1st Place Solution
**Author**: deoxy | **Votes**: 487

---

## Competition
Classify handwritten Bengali grapheme components (root, vowel diacritic, consonant diacritic) from images. 168 grapheme roots + 11 vowel diacritics + 7 consonant diacritics = 186 unique component combinations, but the test set included 49 unseen grapheme roots not present in training. Multi-output classification, recall-weighted macro average.

The crucial challenge: **zero-shot generalization to unseen grapheme classes at test time**.

## Core Innovation: CycleGAN-Based Synthetic Data for Zero-Shot Learning

### The Problem
49 of 168 grapheme roots had no training examples. Standard classifiers simply cannot predict classes they've never seen.

### The Solution: Font Files → Synthetic Data → CycleGAN → Classifier

**Step 1**: Render Bengali graphemes from TTF (TrueType font) files. Each font renders all 168 grapheme roots in a clean, printed style. This gives synthetic "clean" images for ALL classes, including the 49 unseen ones.

```python
from PIL import ImageFont, ImageDraw, Image

def render_grapheme_from_font(grapheme_unicode, font_path, size=128):
    """Render a Bengali grapheme from a TTF font file."""
    font = ImageFont.truetype(font_path, size=64)
    img = Image.new('L', (size, size), 255)
    draw = ImageDraw.Draw(img)
    # Center the character
    bbox = draw.textbbox((0, 0), grapheme_unicode, font=font)
    x = (size - (bbox[2] - bbox[0])) // 2
    y = (size - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), grapheme_unicode, font=font, fill=0)
    return np.array(img)

# Render all 168 graphemes from multiple TTF fonts (diversity)
synthetic_images = {}
for grapheme_id, unicode_char in grapheme_mapping.items():
    for font_path in available_fonts:
        img = render_grapheme_from_font(unicode_char, font_path)
        synthetic_images.setdefault(grapheme_id, []).append(img)
```

**Step 2**: Train a CycleGAN to translate between the handwritten domain (real training data) and the font domain (synthetic rendered data).

```
CycleGAN learns:
  G_HF: Handwritten → Font style
  G_FH: Font style → Handwritten style
  
Training data:
  Domain H: real handwritten grapheme images (training set, 168 classes)
  Domain F: font-rendered grapheme images (all 168 classes)
```

**Step 3**: Apply G_FH to the font-rendered images of the 49 unseen classes → generates "handwritten-style" synthetic training images for unseen classes.

**Step 4**: Train the final EfficientNet-b7 classifier on:
- Real handwritten images (seen classes)
- G_FH(font images) (unseen classes, CycleGAN-translated)

## EfficientNet-b7 Backbone

```python
import timm

model = timm.create_model('efficientnet_b7', pretrained=True, num_classes=0)
# Replace classifier head for multi-output
model.classifier = nn.Sequential(
    nn.Linear(model.num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 168 + 11 + 7)  # roots + vowels + consonants
)
```

Shared backbone for all three output heads. EfficientNet-b7 was the strongest model at the time for this scale of image classification.

## AutoAugment for SVHN

Used Google's AutoAugment policy trained on SVHN (Street View House Numbers) dataset — chosen because SVHN also involves character/digit recognition from natural images, similar domain to handwritten graphemes.

```python
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    AutoAugment(policy=AutoAugmentPolicy.SVHN),  # SVHN policy
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

AutoAugment learns optimal augmentation policies (rotation, shear, brightness, etc.) for a specific dataset. SVHN policy transfers well to handwritten character recognition tasks.

## Out-of-Distribution Detection for Seen vs. Unseen Separation

A binary OOD (out-of-distribution) detector classifies whether each test image belongs to a seen grapheme root or an unseen one:

```python
# Train OOD detector: seen_class=1, unseen=0
# At test time: use OOD score to route predictions
ood_score = ood_detector.predict_proba(test_image)[:, 1]

if ood_score > OOD_THRESHOLD:
    # Seen class: use standard classifier prediction
    prediction = standard_classifier(test_image)
else:
    # Unseen class: use embedding-based retrieval from synthetic data
    prediction = retrieve_from_synthetic_embeddings(test_image)
```

The OOD detector prevents the standard classifier from confidently predicting wrong seen-class labels for unseen grapheme images.

## OHEM: Online Hard Example Mining

OHEM (Liu et al., 2016) selects the hardest examples within each training batch for gradient computation:

```python
class OHEMLoss(nn.Module):
    def __init__(self, ratio=0.7):
        super().__init__()
        self.ratio = ratio  # keep top (1-ratio) fraction of hardest examples
    
    def forward(self, pred, target):
        loss_per_sample = F.cross_entropy(pred, target, reduction='none')
        # Sort by loss; keep the top (1-ratio) hardest
        n_keep = max(1, int(len(loss_per_sample) * (1 - self.ratio)))
        loss_sorted, _ = torch.sort(loss_per_sample, descending=True)
        return loss_sorted[:n_keep].mean()
```

OHEM prevents the model from being dominated by easy examples, focusing gradients on the hardest-to-learn grapheme combinations.

## Key Takeaways
1. CycleGAN for synthetic training data generation enables zero-shot learning on unseen classes
2. Font files are a powerful source of clean, labeled synthetic data for handwritten character recognition
3. G_FH (font→handwritten) translation provides realistic unseen-class training examples
4. OOD detection cleanly separates seen vs. unseen routing at inference
5. AutoAugment with the SVHN policy transfers well to handwritten character recognition
6. OHEM keeps training focused on hard examples; important for 168+-way classification
7. EfficientNet-b7 shared backbone for multi-output (root + vowel + consonant) classification
