# Kaggle Past Solutions — CV & Segmentation

Source: ndres.me/kaggle-past-solutions catalog
Ingested: 2026-04-16

---

## Understanding Clouds from Satellite Images — 1st Place (pudae)

**Competition:** Segment cloud formations in satellite images into 4 types. Multi-class segmentation.
**Writeup:** https://www.kaggle.com/c/understanding_cloud_organization/discussion/118080

### Approach
- Dual architecture: UNet + FPN (Feature Pyramid Network), ensembled
- **Segmentation-as-classifier trick**: use segmentation masks to determine presence/absence of cloud type (classification head on top of segmentation)
- EMA (Exponential Moving Average) weights for more stable predictions
- BCE + Dice loss combination

### Key Techniques
1. **Dual UNet+FPN**: Different architectures capture different scale features; FPN better for large clouds, UNet for detailed boundaries
2. **EMA weights**: Keep running average of model weights during training; use EMA weights for inference (smoother predictions)
3. **Post-processing**: Per-class minimum area thresholds — if predicted mask area < threshold, set to empty
4. **Backbone**: EfficientNet-B5 encoder for both architectures
5. **8-way TTA**: horizontal flip, vertical flip, and their combinations

### How to Reuse
- For any multi-class segmentation: ensemble UNet + FPN with same backbone
- EMA weights: `ema_weight = decay * ema_weight + (1 - decay) * current_weight`, decay=0.999
- Per-class minimum area thresholds are essential when many images have no mask for a given class
- BCE + Dice loss works better than either alone for imbalanced segmentation

---

## Severstal Steel Defect Detection — 1st Place (R Guo team)

**Competition:** Detect and segment 4 types of surface defects in steel images. Multi-class segmentation.
**Writeup:** https://www.kaggle.com/c/severstal-steel-defect-detection/discussion/114254

### Approach
- **Two-stage pipeline**: Classification (has defect?) → Segmentation (where?)
- Custom defect blackout augmentation
- Pseudo labeling with agreement filtering
- Per-class pos_weight tuning

### Architecture
```
Stage 1: EfficientNet classifier → binary "has any defect?"
Stage 2: UNet with ResNet34 encoder → per-pixel segmentation (only for images flagged as defective)
```

### Key Techniques
1. **Classification gate**: Filter out ~70% of defect-free images before expensive segmentation. Dramatically reduces false positives
2. **Defect blackout augmentation**: Randomly black out known defect regions during training to force the model to learn non-defect patterns
3. **Pseudo labeling**: Train initial model → predict on test → filter high-confidence predictions → retrain with pseudo labels. Agreement filter: only use pseudo labels where multiple models agree
4. **Per-class pos_weight**: Different defect types have different prevalence. Set `pos_weight = num_negative / num_positive` per class in BCE loss
5. **Test-time augmentation**: Horizontal flip + original, averaged

### How to Reuse
- Two-stage (classify → segment) is standard for sparse defect detection
- Defect blackout augmentation: applicable to any anomaly detection task
- Per-class pos_weight is essential when class frequencies vary 10x+
- Agreement-based pseudo labeling is safer than single-model pseudo labeling

---

## Kuzushiji Recognition — 1st Place (tascj)

**Competition:** Detect and classify ancient Japanese (Kuzushiji) characters in historical documents. Object detection.
**Code:** https://github.com/tascj/kaggle-kuzushiji-recognition

### Approach
- **Cascade R-CNN** with **HRNet** backbone (High-Resolution Network)
- Multi-scale training and testing
- Built on MMDetection framework
- Crop-train / full-test strategy

### Architecture
```
Backbone: HRNet-W32 (maintains high resolution throughout, unlike ResNet which downsamples)
Neck: FPN
Head: Cascade R-CNN (3-stage refinement with IoU thresholds 0.5/0.6/0.7)
```

### Key Techniques
1. **HRNet backbone**: Maintains high-resolution representations — crucial for small character detection in large document images
2. **Cascade R-CNN**: Progressive refinement at increasing IoU thresholds. Each stage refines bounding boxes from the previous stage
3. **Multi-scale training**: Random resize between 0.5x-1.5x of original resolution
4. **Multi-scale testing**: Predict at 3 scales, merge with NMS
5. **Crop-train / Full-test**: Train on cropped 1024x1024 patches (fits in GPU memory), test on full images (better context)

### How to Reuse
- For dense small object detection: HRNet > ResNet backbone
- Cascade R-CNN gives 1-2 mAP improvement over Faster R-CNN for free
- MMDetection configs: `cascade_rcnn_hrnetv2p_w32_20e.py` as starting point
- Multi-scale test with NMS is standard for object detection competitions

---

## Recursion Cellular Image Classification — 4th Place (yu4u / cab team)

**Competition:** Classify 1,108 genetic perturbations from 6-channel fluorescence microscopy images. Fine-grained classification.
**Code:** https://github.com/ngxbac/Kaggle-Recursion-Cellular

### Approach
- **Control image pretraining**: Pre-train on control (untreated) images to learn cell morphology before fine-tuning on perturbation classification
- Channel subset ensembling
- ChannelDropout augmentation
- Linear sum assignment for plate constraints

### Key Techniques
1. **Control image pretraining**: The dataset has control images (no perturbation) for each experiment plate. Pre-training on these teaches the model baseline cell morphology, making perturbation detection easier
2. **Channel subset ensembling**: 6-channel images → train separate models on different channel subsets (e.g., channels [1,2,3], [4,5,6], [1,3,5]) → ensemble predictions. Each subset captures different biological signals
3. **ChannelDropout**: During training, randomly zero out entire channels (similar to Dropout but for input channels). Forces model to not rely on any single channel
4. **Plate-aware inference**: Each 384-well plate has exactly one sample per perturbation. Use linear sum assignment (Hungarian algorithm) to enforce this constraint at test time — each perturbation assigned exactly once per plate

### How to Reuse
- Multi-channel images: try channel subset ensembling instead of just using all channels
- Domain-specific pretraining (control images, unlabeled data) before fine-tuning
- ChannelDropout: `torchvision.transforms` doesn't have this — implement as custom augmentation
- Constraint-based post-processing (Hungarian algorithm) when you know structural properties of test data

---

## APTOS 2019 Blindness Detection — 1st Place (Guanshuo Xu)

**Competition:** Grade severity of diabetic retinopathy from retinal images (0-4 scale). Ordinal regression.
**Writeup:** https://www.kaggle.com/c/aptos2019-blindness-detection/discussion/108065

### Approach
- **Regression with SmoothL1Loss** (not classification)
- Generalized Mean Pooling (GeM)
- External data with label smoothing
- QWK threshold optimization

### Key Techniques
1. **Regression for ordinal targets**: Instead of 5-class classification, treat as regression (target 0.0-4.0). SmoothL1Loss is more robust to label noise than MSE
2. **Generalized Mean Pooling (GeM)**: Replaces standard GlobalAveragePool. `GeM(p)` learns the pooling exponent — when p=1 it's average pooling, p→∞ it's max pooling. Typically learns p≈3
   ```python
   class GeM(nn.Module):
       def __init__(self, p=3, eps=1e-6):
           super().__init__()
           self.p = nn.Parameter(torch.ones(1) * p)
           self.eps = eps
       def forward(self, x):
           return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), 
                              (x.size(-2), x.size(-1))).pow(1./self.p)
   ```
3. **External data with label smoothing**: Used 2015 Diabetic Retinopathy competition data as additional training. Applied label smoothing (0.1) on external data since labels may differ slightly
4. **QWK threshold optimization**: After regression, find optimal thresholds to convert continuous scores to discrete grades (0-4) that maximize Quadratic Weighted Kappa. Use `scipy.optimize.minimize` on validation set
5. **Backbone**: EfficientNet-B5, fine-tuned from ImageNet

### How to Reuse
- For any ordinal classification with QWK metric: regression + threshold optimization > classification
- GeM pooling: drop-in replacement for GAP, usually +0.5-1% improvement
- External data from related competitions: use with label smoothing (0.05-0.1)
- SmoothL1Loss for ordinal regression (more robust than MSE to noisy labels)

---

## Cross-Cutting Patterns

| Pattern | Clouds | Severstal | Kuzushiji | Recursion | APTOS |
|---------|--------|-----------|-----------|-----------|-------|
| Task type | Seg | Seg | Detect | Classify | Ordinal |
| Two-stage pipeline | No | Yes | No | No | No |
| TTA | 8-way | 2-way | Multi-scale | No | 4-way |
| Pseudo labeling | No | Yes | No | No | No |
| Custom pooling | No | No | No | No | GeM |
| External data | No | No | No | Controls | 2015 comp |
| Post-processing | Area thresh | Classify gate | NMS | Hungarian | QWK thresh |
| Framework | PyTorch | PyTorch | MMDetection | PyTorch | PyTorch |
