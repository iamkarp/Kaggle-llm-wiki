---
title: "Tabular Data Augmentation Techniques"
tags: [tabular-augmentation, scarf, vime, saint, tta, mixup, tabular, neural-network]
date: 2026-04-15
source_count: 5
status: active
---

## Summary

Swap noise (SCARF, ICLR 2022) is the gold standard for tabular pretraining — 30% feature corruption via random sampling. SAINT's CutMix+Mixup significantly improves classification accuracy (+2-4%). TTA for tabular works by adding feature noise at inference time and averaging predictions. TabMDA extends MixUp to work with tree-based models (XGBoost, LightGBM).

## What It Is

Data augmentation techniques adapted for tabular data — primarily useful for neural network tabular models. Less effective (but not useless) for GBDTs.

## Key Facts / Details

### SCARF — Swap Noise Contrastive Pretraining (ICLR 2022)

Corruption strategy: randomly replace ~30% of features with values sampled from the marginal distribution of that feature.

```python
import torch
import torch.nn as nn

class SCARFEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, n_layers=4, corruption_rate=0.3):
        super().__init__()
        self.corruption_rate = corruption_rate
        layers = []
        current_dim = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(current_dim, emb_dim), nn.ReLU()])
            current_dim = emb_dim
        self.encoder = nn.Sequential(*layers)
    
    def corrupt(self, x, x_marginals):
        """Replace ~30% of features with random samples from marginal distribution."""
        corruption_mask = torch.bernoulli(
            torch.full(x.shape, self.corruption_rate)
        ).bool()
        x_random_rows = x_marginals[torch.randint(0, len(x_marginals), (len(x),))]
        x_corrupted = torch.where(corruption_mask, x_random_rows, x)
        return x_corrupted, corruption_mask
    
    def forward(self, x, x_marginals=None):
        if self.training and x_marginals is not None:
            x_corrupted, mask = self.corrupt(x, x_marginals)
            z = self.encoder(x)
            z_corrupted = self.encoder(x_corrupted)
            return z, z_corrupted, mask
        return self.encoder(x)
```

**Downstream performance:** SCARF pretraining consistently improves downstream classification performance. Typical gain: +0.5-2% on tabular benchmarks.

### VIME — Alternative Swap Noise

```python
def vime_corrupt(x, corruption_rate=0.3):
    """
    VIME-style corruption with mask-based reconstruction objective.
    Returns corrupted x and binary mask.
    """
    mask = torch.bernoulli(torch.full(x.shape, corruption_rate)).bool()
    x_random = x[torch.randperm(len(x))]  # Permute within column
    x_corrupted = torch.where(mask, x_random, x)
    return x_corrupted, mask.float()
```

**Difference from SCARF:** VIME corrupts by column permutation; SCARF corrupts by sampling from marginal distribution. SCARF tends to perform slightly better.

### SAINT — CutMix + Embedding Mixup

```python
class SAINTModel(nn.Module):
    """
    SAINT: Self-Attention and Intersample Attention Transformer
    Key innovation: intersample attention + CutMix+Mixup augmentation
    """
    def __init__(self, n_features, n_classes, cutmix_lam=0.5, mixup_lam=0.3):
        super().__init__()
        self.cutmix_lam = cutmix_lam
        self.mixup_lam = mixup_lam
        # ... transformer layers

def saint_augmentation(x_batch, y_batch, lam_cutmix=0.5, lam_mixup=0.3):
    """CutMix: randomly select features from another sample."""
    batch_size = len(x_batch)
    idx = torch.randperm(batch_size)
    
    # CutMix: swap features
    cut_mask = torch.bernoulli(torch.full((x_batch.shape[1],), lam_cutmix)).bool()
    x_cutmix = x_batch.clone()
    x_cutmix[:, cut_mask] = x_batch[idx][:, cut_mask]
    
    # Mixup: interpolate in embedding space (done inside model)
    # y_mixed = lam_mixup * y_batch + (1-lam_mixup) * y_batch[idx]
    return x_cutmix, idx, lam_mixup
```

**Results:** SAINT significantly improves performance on multiclass classification (demonstrated on UCI datasets).

### TTA for Tabular Data

Test-Time Augmentation via feature noise injection:

```python
def tta_predict_tabular(model, X_test, n_augmentations=20, noise_std=0.05):
    """
    Add Gaussian noise to numerical features at inference time.
    Average predictions across augmentations.
    """
    all_preds = []
    
    # Original prediction
    all_preds.append(model.predict_proba(X_test)[:, 1])
    
    for _ in range(n_augmentations - 1):
        X_noisy = X_test.copy()
        # Add noise only to numerical columns
        for col in X_test.select_dtypes(include=[np.float32, np.float64]).columns:
            noise = np.random.normal(0, noise_std * X_test[col].std(), len(X_test))
            X_noisy[col] = X_test[col] + noise
        all_preds.append(model.predict_proba(X_noisy)[:, 1])
    
    return np.mean(all_preds, axis=0)
```

**Typical gain:** +0.001-0.002 AUC at inference time. Very low effort.

**Feature permutation TTA (for categorical features):**
```python
def categorical_tta_predict(model, X_test, cat_cols, n_augmentations=10):
    """Randomly drop categorical features to create diverse predictions."""
    all_preds = [model.predict_proba(X_test)[:, 1]]
    
    for _ in range(n_augmentations - 1):
        X_aug = X_test.copy()
        cols_to_drop = np.random.choice(cat_cols, size=len(cat_cols)//3, replace=False)
        for col in cols_to_drop:
            X_aug[col] = X_test[col].mode()[0]  # Replace with most common value
        all_preds.append(model.predict_proba(X_aug)[:, 1])
    
    return np.mean(all_preds, axis=0)
```

### TabMDA — MixUp for Tree-Based Models

Paper: TabMDA (2024). MixUp extended to work with non-differentiable tree models.

```python
def tabmda_augment(X_train, y_train, model, n_synthetic=5000, alpha=0.2):
    """
    Generate synthetic samples via MixUp, train model on augmented data.
    Works with XGBoost, LightGBM, RandomForest.
    """
    n = len(X_train)
    synthetic_X = []
    synthetic_y = []
    
    for _ in range(n_synthetic):
        i, j = np.random.randint(0, n, 2)
        lam = np.random.beta(alpha, alpha)
        
        x_mix = lam * X_train[i] + (1 - lam) * X_train[j]
        y_mix = lam * y_train[i] + (1 - lam) * y_train[j]
        
        synthetic_X.append(x_mix)
        synthetic_y.append(y_mix)
    
    X_augmented = np.vstack([X_train, synthetic_X])
    y_augmented = np.concatenate([y_train, synthetic_y])
    
    return X_augmented, y_augmented
```

**Competition result:** TabMDA showed +0.5-1% improvement on several datasets when applied to XGBoost.

### Competition Summary Table

| Technique | Neural Net | GBDT | Impact | When to Use |
|---|---|---|---|---|
| SCARF swap noise pretraining | Yes | No | Medium | Neural net tabular models |
| VIME corruption | Yes | No | Medium | Alternative to SCARF |
| SAINT CutMix+Mixup | Yes | No | Medium-High | Multi-class tabular NN |
| TTA with feature noise | Yes | Yes | Low | Low-effort gain at inference |
| TabMDA MixUp | No | Yes | Low-Medium | XGBoost/LGB with limited data |
| Standard Mixup | Yes | No | Medium | Neural net regression |

## Gotchas

- SCARF and VIME only work with neural network models — GBDTs can't learn from augmented samples in the same way
- TTA noise_std=0.05 × feature_std is a good starting point; too high = degrades predictions
- TabMDA with soft (fractional) labels requires using probability outputs — doesn't work with hard-label GBDTs
- SAINT is computationally expensive — primarily worth it for competitions with strong NLP/heterogeneous features

## Sources

- [[../raw/kaggle/tabular-data-augmentation.md]] — full reference with code
- [SCARF paper (ICLR 2022)](https://arxiv.org/abs/2106.15147)
- [SAINT paper](https://arxiv.org/abs/2106.01342)
- [TabMDA paper (2024)](https://arxiv.org/abs/2404.08434)
- [VIME paper](https://arxiv.org/abs/2006.10751)

## Related

- [[concepts/deep-learning-tabular]] — neural network tabular architectures
- [[concepts/pseudo-labeling]] — another semi-supervised approach
- [[concepts/image-augmentation]] — augmentation for CV competitions
- [[concepts/tabpfn-tabm]] — TabPFN and TabM as alternative tabular approaches
