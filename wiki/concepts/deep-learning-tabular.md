---
title: "Deep Learning on Tabular Data — When DNNs Beat GBMs"
tags: [deep-learning, tabular, vsn, variable-selection, dropout, normalization, attention, deepinsight, tabnet]
date: 2026-04-15
source_count: 3
status: active
---

## The Core Question: When Do DNNs Beat Trees?

Conventional wisdom: gradient boosted trees (LightGBM, XGBoost, CatBoost) beat neural networks on tabular data. This is mostly true. But "mostly" has important exceptions:

| Condition | Winner | Reason |
|-----------|--------|--------|
| Large dataset, diverse features | GBM | Fast training, robust hyperparams, no normalization needed |
| Medium dataset, mixed types | GBM | More reliable generalization |
| **Very small dataset (<1K rows)** | **DNN** (with extreme regularization) | GBMs overfit catastrophically; NNs with dropout generalize better |
| Sequential data (time series, sentences) | DNN | RNN/Transformer; GBMs can't model sequence order |
| Symmetric features (all interchangeable) | DNN | Attention/VSN learns feature importance; one-hot-like GBM splits are wasteful |
| Auxiliary/multi-task targets available | DNN | Multi-head training exploits all available signal |
| Need ensemble diversity | Both | Blend GBM + NN for complementary errors |

**Key insight from ICR competition (617 rows)**: On sufficiently tiny datasets, GBMs hit irreducible overfitting. A DNN with extreme regularization can generalize better because dropout forces the model to learn with random feature subsets — a stronger regularizer than tree pruning.

## Variable Selection Networks (VSN)

From the Temporal Fusion Transformer (TFT) paper; adapted for tabular in ICR 1st place.

### Core Idea
Rather than treating features as a flat vector, VSN gives each feature:
1. Its own learned linear projection (8 neurons)
2. A gated residual transformation (GRN)
3. An attention weight learned jointly with the prediction head

This allows the model to learn what transformation of each feature is most informative, not just what its scaled value is.

### Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GRN(nn.Module):
    """Gated Residual Network — the core VSN building block."""
    def __init__(self, d, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)
        self.gate = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d)
    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * h + (1 - g) * x)

class VSN(nn.Module):
    def __init__(self, n_features, d=8):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(1, d) for _ in range(n_features)])
        self.grn  = nn.ModuleList([GRN(d) for _ in range(n_features)])
        self.attn = nn.Linear(d, 1)
    
    def forward(self, x):
        hs = [grn(proj(x[:, i:i+1]))
              for i, (proj, grn) in enumerate(zip(self.proj, self.grn))]
        stacked = torch.stack(hs, dim=1)          # (B, n_features, d)
        w = torch.softmax(self.attn(stacked), dim=1)  # (B, n_features, 1)
        return (w * stacked).sum(dim=1)            # (B, d)

class VSNClassifier(nn.Module):
    def __init__(self, n_features, d=8, hidden=256, n_classes=1, dropout=0.3):
        super().__init__()
        self.vsn = VSN(n_features, d)
        self.head = nn.Sequential(
            nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2), nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(hidden//2, n_classes),
        )
    def forward(self, x):
        return self.head(self.vsn(x))
```

### VSN vs. Standard Normalization
| Approach | Per-Feature Transform | Learned? | Expressiveness |
|----------|----------------------|---------|----------------|
| StandardScaler | Subtract mean, divide std | No | Fixed linear |
| RankGauss | Rank → probit | No | Fixed nonlinear |
| Min-Max | Scale to [0,1] | No | Fixed linear |
| **VSN projection** | Linear(1 → 8) + GRN | **Yes** | **Learned nonlinear** |

VSN projections allow the model to find optimal feature transformations task-specifically. For anonymized features (ICR, Santander) where no domain knowledge guides preprocessing, this is particularly valuable.

## Extreme Dropout for Tiny Datasets

Standard dropout: 0.1–0.3. For datasets with < 1000 rows:

```python
# ICR architecture: 617 training rows
model = nn.Sequential(
    nn.Linear(n_features, 512),
    nn.ReLU(),
    nn.Dropout(0.75),    # drop 75% of neurons — extreme!
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.25),
    nn.Linear(128, 1),
    nn.Sigmoid(),
)
```

**Why it works on tiny data**:
- With p=0.75, each forward pass uses a random 25% of the first layer — the model must distribute knowledge across all neurons
- Effective ensemble over ~2^512 sub-networks (each dropout mask = a different sub-network)
- Dramatically reduces effective capacity without shrinking architecture
- Prevents co-adaptation: no single neuron can rely on another always being present

**When to use extreme dropout**:
- Dataset size < 2000 rows
- Model is clearly overfitting despite standard regularization
- Start at p=0.5 on the first hidden layer, increase to 0.75 if overfitting persists

## Repeated Training + Cherry-Picking

For tiny datasets where NN variance is very high:

```python
def train_best_of_n(X_train, y_train, X_val, y_val,
                    n_runs=30, keep_best_k=2, model_factory=None):
    """Train model n_runs times; keep k best by validation loss."""
    results = []
    for seed in range(n_runs):
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = model_factory()
        model.fit(X_train, y_train)
        val_loss = model.evaluate(X_val, y_val)
        results.append((val_loss, model))
    
    results.sort(key=lambda x: x[0])
    best_models = [m for _, m in results[:keep_best_k]]
    
    # Average predictions of the k best
    preds = np.mean([m.predict(X_val) for m in best_models], axis=0)
    return best_models, preds
```

**ICR**: 10–30 runs per fold, keep 2 best. The 2-model average reduces variance without cherry-picking too aggressively.

**Risk**: On larger datasets, this becomes a form of validation set overfitting (if you cherry-pick heavily). Keep the number of seeds modest (≤5) for medium datasets.

## DeepInsight: Tabular → Image → CNN

Convert tabular features to 2D images using t-SNE or UMAP feature layout, then apply EfficientNet/ResNet.

```python
from sklearn.manifold import TSNE
import numpy as np
from PIL import Image

def create_feature_layout(X_train: np.ndarray, image_size: int = 64):
    """Use t-SNE to assign 2D positions to features."""
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    positions = tsne.fit_transform(X_train.T)  # (n_features, 2) — features as points
    
    # Normalize to pixel grid
    x = ((positions[:, 0] - positions[:, 0].min()) /
         (positions[:, 0].ptp()) * (image_size - 1)).astype(int)
    y = ((positions[:, 1] - positions[:, 1].min()) /
         (positions[:, 1].ptp()) * (image_size - 1)).astype(int)
    return x, y

def sample_to_image(sample: np.ndarray, px: np.ndarray, py: np.ndarray,
                    image_size: int = 64) -> np.ndarray:
    """Convert one sample's feature values to a 2D image."""
    img = np.zeros((image_size, image_size), dtype=np.float32)
    img[py, px] = sample  # place feature values at t-SNE positions
    return img
```

**Advantages**:
- CNNs learn local correlation patterns in the t-SNE layout (correlated features cluster together)
- Provides genuine model family diversity (CNN vs. LGBM vs. MLP)
- Leverages ImageNet pre-training (EfficientNet weights)

**Disadvantages**:
- t-SNE layout requires the full training feature matrix — can't do it with a held-out test set at layout time
- Image creation per sample is slow (parallelize with multiprocessing)
- Adds significant engineering complexity

## TabNet: Sequential Attention for Tabular

TabNet uses sequential attention steps to select a subset of features for each decision, rather than using all features at every step.

```python
from pytorch_tabnet.tab_model import TabNetClassifier

model = TabNetClassifier(
    n_d=64, n_a=64,        # dimension of prediction + attention layers
    n_steps=5,              # number of sequential attention steps
    gamma=1.5,              # sparsity regularization (higher = sparser attention)
    n_independent=2,        # independent GLU layers per step
    n_shared=2,             # shared GLU layers across steps
    momentum=0.02,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={'lr': 2e-2},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={'step_size': 10, 'gamma': 0.9},
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], patience=15)
```

TabNet typically underperforms LGBM as a standalone model but provides useful ensemble diversity in stacking pipelines (MoA 1st place used it at 10% weight).

## Quick Decision Guide

```
Dataset size?
  < 1000 rows:
    → DNN with VSN + extreme dropout (0.75→0.5→0.25)
    → Repeat 10–30x, keep 2 best per fold
    → Blend with GBM
  1K – 100K rows:
    → GBM first; add DNN for ensemble diversity
    → Standard dropout (0.2–0.4); RankGauss normalization
  > 100K rows:
    → GBM dominates; DNN adds marginal gain
    → If sequential: RNN/Transformer is essential

Feature type?
  Symmetric / anonymous → VSN or attention NN
  Sequential / time-ordered → GRU or Transformer
  Can be imaged → DeepInsight + EfficientNet for diversity

Auxiliary targets available?
  → Multi-head training (2-heads ResNet, MoA pattern)
```

## Sources
- [[../../raw/kaggle/solutions/icr-age-conditions-1st-room722.md]] — VSN, extreme dropout, repeated training
- [[../../raw/kaggle/solutions/moa-1st-mark-peng.md]] — 2-heads ResNet, DeepInsight, TabNet
- [[../../raw/kaggle/solutions/porto-seguro-1st-jahrer.md]] — RankGauss normalization baseline

## Related
- [[../concepts/denoising-autoencoders]] — DAE as an alternative NN approach for tabular
- [[../concepts/knowledge-distillation]] — LGBM → NN to initialize from a stronger starting point
- [[../concepts/ensembling-strategies]] — DNN + GBM blending for complementary errors
- [[../concepts/stacking-deep]] — DeepInsight + TabNet + ResNet as diverse L1 base models
- [[../concepts/feature-engineering-tabular]] — when to engineer vs. let the NN learn
