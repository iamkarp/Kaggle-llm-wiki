# Mechanisms of Action (MoA) Prediction — 1st Place Solution
**Author**: Mark Peng | **Year**: 2020 | **Votes**: 295

---

## Competition
Multi-label classification: given gene expression and cell viability data from drug experiments, predict which of 206 molecular mechanisms of action (MoA) a drug activates. Log-loss metric. ~23K training samples, 875 features (gene expression + cell viability), 206 binary targets. Additional "non-scored" targets available (402 more labels, not evaluated).

## Architecture Overview: 3-Stage NN Stacking

```
Stage 1: Base models → OOF predictions for scored + non-scored targets
Stage 2: Meta-features from Stage 1 → predict scored targets
Stage 3: Blend of Stage 1 + Stage 2 outputs

Plus: DeepInsight branch (tabular → image → EfficientNet)
Plus: TabNet branch
Final: 7-model weighted blend
```

## Key Innovation 1: Non-Scored Targets as Meta-Features

The competition provided 402 "non-scored" targets — additional MoA labels that weren't evaluated but were available for training. Most competitors ignored them or used them only as auxiliary loss.

Mark Peng's insight: **predict the non-scored targets in Stage 1, then use those predictions as meta-features for Stage 2**.

```
Stage 1:
    Input: 875 raw features
    → 2-heads ResNet predicts:
        - 206 scored targets (OOF)
        - 402 non-scored targets (OOF)
    Output: 608 OOF meta-features (206 + 402)

Stage 2:
    Input: 608 meta-features from Stage 1
    → Smaller NN predicts 206 scored targets
    Output: final scored predictions

Blend: 0.5 × Stage1 + 0.5 × Stage2
```

**Why it works**: The 402 non-scored targets are highly correlated with the 206 scored targets (they're all MoA labels from the same drug experiments). Stage 1 predictions of non-scored targets capture drug-mechanism structure; Stage 2 learns to translate this structure into scored target predictions.

## 2-Heads ResNet Architecture

```python
class TwoHeadResNet(nn.Module):
    def __init__(self, n_features, n_scored, n_nonscored, dropout=0.3):
        super().__init__()
        # Shared backbone
        self.backbone = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Residual blocks
            ResidualBlock(1024, dropout),
            ResidualBlock(1024, dropout),
            ResidualBlock(512, dropout),
        )
        # Two prediction heads
        self.scored_head = nn.Linear(512, n_scored)
        self.nonscored_head = nn.Linear(512, n_nonscored)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.scored_head(features), self.nonscored_head(features)

class ResidualBlock(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(size),
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size),
            nn.Linear(size, size),
        )
    
    def forward(self, x):
        return x + self.net(x)  # residual connection
```

## DeepInsight: Tabular Features → Images → CNN

DeepInsight (Sharma et al., 2019) converts tabular features into 2D images so that CNNs can be applied. The technique:

1. Apply t-SNE or UMAP to the feature vectors to get 2D coordinates
2. Each feature gets an (x, y) position in a 2D grid
3. For each sample, the feature values are placed at their grid positions
4. The result is a 2D image (grayscale or multi-channel) per sample
5. Train EfficientNet on these images

```python
from sklearn.manifold import TSNE
import numpy as np

# Get 2D positions for features via t-SNE
feature_matrix = X_train.T  # (n_features, n_samples)
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
feature_positions = tsne.fit_transform(feature_matrix)  # (n_features, 2)

# Map to pixel grid
def to_image(sample_features, feature_positions, image_size=64):
    img = np.zeros((image_size, image_size))
    # Normalize positions to [0, image_size-1]
    x = ((feature_positions[:, 0] - feature_positions[:, 0].min()) /
         (feature_positions[:, 0].max() - feature_positions[:, 0].min()) * (image_size - 1)).astype(int)
    y = ((feature_positions[:, 1] - feature_positions[:, 1].min()) /
         (feature_positions[:, 1].max() - feature_positions[:, 1].min()) * (image_size - 1)).astype(int)
    img[y, x] = sample_features
    return img
```

**Why EfficientNet on tabular-to-image**:
- Local spatial structure in the t-SNE layout reflects feature correlations
- CNN convolutional filters learn local correlation patterns in feature space
- EfficientNet is parameter-efficient; pretrained on ImageNet but fine-tunable
- Provides a fundamentally different model class for ensemble diversity

## TabNet in the Stack

TabNet (Arik & Pfenning, 2019) uses sequential attention to select features for each decision step. Unlike standard NNs that use all features equally, TabNet learns which features to "attend to" for each step.

```python
from pytorch_tabnet.tab_model import TabNetClassifier

tabnet = TabNetClassifier(
    n_d=64, n_a=64,     # prediction and attention embedding dims
    n_steps=5,           # number of sequential attention steps
    gamma=1.5,           # sparsity regularization
    n_independent=2,     # number of independent GLU layers per step
    n_shared=2,          # number of shared GLU layers
    momentum=0.02,       # batch norm momentum
)
tabnet.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```

TabNet outputs were used as an additional stacking component, providing sparse attention-based predictions as a contrast to the dense ResNet and EfficientNet models.

## 7-Model Weighted Blend

Final ensemble:
| Model | Weight | Notes |
|-------|--------|-------|
| 2-heads ResNet (Stage 1) | 0.20 | Base scorer + non-scored predictor |
| Stage 2 NN on meta-features | 0.20 | Uses Stage 1 as input |
| EfficientNet-B3 (DeepInsight) | 0.15 | CNN on tabular-to-image |
| EfficientNet-B4 (DeepInsight) | 0.15 | Larger CNN variant |
| TabNet | 0.10 | Sequential attention |
| LightGBM | 0.10 | Tree model |
| MLP (vanilla) | 0.10 | Baseline NN |

Weights tuned by OOF score on validation folds.

## Key Takeaways
1. Non-scored targets as meta-features: predict everything available in Stage 1, use all predictions as Stage 2 input
2. 2-heads architecture: predicting auxiliary targets jointly with scored targets improves representation
3. DeepInsight (tabular → t-SNE image → CNN) provides genuine ensemble diversity from a completely different model family
4. 3-stage stacking with non-scored meta-features is the core structural insight
5. TabNet adds sparse attention diversity to the ensemble
6. Always use auxiliary targets if available — as co-training objectives and/or as meta-features
