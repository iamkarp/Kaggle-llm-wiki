# Human Protein Atlas Image Classification — 1st Place Solution
**Author**: bestfitting | **Votes**: 628

---

## Competition
Multi-label image classification of protein subcellular localization from fluorescence microscopy images. 28 classes, severely imbalanced (some classes have <100 training examples). Macro-averaged F1 metric. ~31K training images, 4-channel (RGBY).

## Architecture: DenseNet121 with AdaptiveConcatPool2d

Base backbone: DenseNet121 (pretrained ImageNet). Key modification to the pooling layer:

### AdaptiveConcatPool2d
Replace the standard GlobalAveragePooling with a concatenation of average AND max pooling:

```python
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, output_size=1):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size)
        self.max = nn.AdaptiveMaxPool2d(output_size)
    
    def forward(self, x):
        return torch.cat([self.avg(x), self.max(x)], dim=1)

# Replace DenseNet's final pooling
model.features.add_module('norm5', nn.BatchNorm2d(1024))
model.features.add_module('relu5', nn.ReLU(inplace=True))
model.features.add_module('pool5', AdaptiveConcatPool2d())
# Classifier input now 2048 (1024 avg + 1024 max)
model.classifier = nn.Linear(2048, 28)
```

**Why concat pooling**: Average pooling captures the mean activation (good for distributed patterns), max pooling captures the peak activation (good for rare/localized patterns). For protein localization, both matter — some proteins are uniformly distributed, others appear in specific puncta. Concatenating both gives the classifier twice the signal.

This trick originated in fast.ai and has since become standard for image classification competitions.

## Loss Functions: FocalLoss + Lovász Combo

### Focal Loss (for class imbalance)
```python
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        return focal_loss.mean()
```

Focal loss down-weights easy examples (high pt) and focuses on hard examples (low pt). Critical here because the model quickly learns the common classes (Nucleus, Cytoplasm) — focal loss keeps it learning the rare classes.

### Lovász Loss (for F1 optimization)
Lovász-Softmax directly optimizes a differentiable surrogate of the Jaccard/IoU metric, which is closely related to F1. Used in combination with Focal:

```python
# Combined loss
loss = 0.5 * focal_loss(pred, target) + 0.5 * lovasz_loss(pred, target)
```

BCE + Lovász is also a common and effective combination.

## The Core Breakthrough: ArcFace Metric Learning for Label Transfer

**The single biggest improvement: +0.03 on the public LB.**

### ArcFace Background
ArcFace (Deng et al., 2019) is a loss function from face recognition that maximizes angular separability of feature embeddings. The key property: features from the same class cluster tightly in the embedding space.

```python
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, n_classes, s=64.0, m=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s      # scale factor
        self.m = m      # margin (angular)
    
    def forward(self, features, labels):
        # Normalize features and weights
        cos_theta = F.linear(F.normalize(features), F.normalize(self.weight))
        
        # Add angular margin to the target class
        theta = torch.acos(torch.clamp(cos_theta, -1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.m)
        
        # One-hot encode labels
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * target_logit + (1 - one_hot) * cos_theta) * self.s
        return F.cross_entropy(output, labels)
```

### Label Transfer via NN Retrieval

After training the ArcFace model:
1. Extract embeddings for all training images and all test images
2. For each test image, find K nearest neighbors in the training set (by cosine similarity in embedding space)
3. If the nearest neighbor has high retrieval accuracy (Top-1 > 0.9), replace the predicted label with the neighbor's ground-truth label

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

# Extract embeddings
train_embeddings = extract_features(model, train_loader)  # (N_train, d)
test_embeddings = extract_features(model, test_loader)    # (N_test, d)

# Fit NN index on train embeddings
nn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_index.fit(train_embeddings)

# For each test image, find nearest neighbors
distances, indices = nn_index.kneighbors(test_embeddings)

# Replace prediction if retrieval confidence is high
RETRIEVAL_THRESHOLD = 0.9  # cosine similarity threshold

final_predictions = base_predictions.copy()  # start from classifier predictions
for i, (dists, nbr_indices) in enumerate(zip(distances, indices)):
    cosine_sims = 1 - dists  # convert distance to similarity
    if cosine_sims[0] > RETRIEVAL_THRESHOLD:
        # High-confidence retrieval: use neighbor's label
        neighbor_labels = train_labels[nbr_indices[0]]
        final_predictions[i] = neighbor_labels
```

**Why this works so well**:
- For extremely rare classes (<100 training examples), the classifier head overfits but the embedding space still generalizes
- ArcFace training forces tight clustering by class in embedding space
- When a test image is very close to a training image, that training image's label is more reliable than the softmax classifier
- Top-1 retrieval accuracy >0.9 means the NN assignment is correct >90% of the time — a high-quality pseudo-label oracle

## Multi-Label Handling

The competition is multi-label (multiple classes per image). Adaptations:
- ArcFace (originally single-label) adapted to multi-label via summing class-specific margins
- Threshold tuning per class (not a global threshold) because class frequencies differ drastically
- Calibrated threshold search over validation set for each class separately

## External Data
Used extra HPAv18 data (older version of Human Protein Atlas) for pretraining. Fine-tuned on competition data. Standard practice for competitions with extra available data.

## Key Takeaways
1. AdaptiveConcatPool2d (avg + max) is a simple boost over standard global average pooling
2. FocalLoss + Lovász combination works well for imbalanced multi-label classification
3. ArcFace metric learning → NN retrieval for label transfer is the key technique (+0.03)
4. Top-1 retrieval accuracy >0.9 is a reliable enough threshold for label replacement
5. For rare classes: embedding-based retrieval generalizes better than softmax classifier
6. Always tune classification thresholds per class independently for multi-label problems
