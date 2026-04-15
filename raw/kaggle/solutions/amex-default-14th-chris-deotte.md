# American Express Default Prediction — 14th Place (Gold) Solution
**Author**: Chris Deotte | **Year**: 2022 | **Votes**: 268

---

## Competition
Same as 1st place: predict credit card default. Binary classification, proprietary Amex metric. Sequential statement data (up to 13 months per customer).

## Core Innovation: LGBM → NN Knowledge Distillation

The key insight: **train a Transformer NN using LGBM soft labels (predicted probabilities), then fine-tune on hard targets**. This is knowledge distillation applied across model families.

### Why Distillation Works Here
- LGBM produces well-calibrated soft probabilities that contain richer information than binary 0/1 labels
- The soft labels "smooth" the training signal — the NN learns a more gradual decision boundary
- LGBM saw more signal in the tabular features; the NN learns from LGBM's knowledge, then adds sequential pattern knowledge the LGBM couldn't capture

## 4-Cycle Cosine Schedule: Alternating Distillation and Fine-Tuning

Rather than a single distillation pass followed by fine-tuning, use a cyclic schedule:

```
Cycle 1: Train NN on LGBM soft labels (distillation)     — high learning rate
Cycle 2: Fine-tune NN on hard targets (true labels)       — lower learning rate
Cycle 3: Train NN on LGBM soft labels again              — medium learning rate
Cycle 4: Final fine-tune on hard targets                  — lowest learning rate
```

Each cycle uses a **cosine annealing learning rate schedule** (lr decays from max to near-zero over the cycle).

```python
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Cycle 1: distillation
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs_cycle1, eta_min=1e-6)
for epoch in range(n_epochs_cycle1):
    train_epoch(model, loader_soft_labels, loss_fn=nn.BCELoss())  # soft labels
    scheduler.step()

# Cycle 2: fine-tune
optimizer.param_groups[0]['lr'] = 5e-4
scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs_cycle2, eta_min=1e-6)
for epoch in range(n_epochs_cycle2):
    train_epoch(model, loader_hard_labels, loss_fn=nn.BCELoss())  # hard labels
    scheduler.step()

# Cycles 3 and 4: repeat with decreasing max lr
```

**Why cycles help**: Each alternation allows the model to re-optimize the balance between LGBM knowledge (soft) and ground truth (hard). The cosine schedule prevents sharp jumps in weights between cycles.

## LGBM Soft Labels: OOF + Test Predictions

The soft labels used for distillation come from:
- **Training data**: LGBM out-of-fold predictions (avoids leakage into training)
- **Test data**: LGBM test predictions (average of models from all folds)

```python
# Generate LGBM OOF soft labels
lgbm_oof = np.zeros(len(X_train))
lgbm_test = np.zeros(len(X_test))

for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train[tr_idx], y_train[tr_idx])
    lgbm_oof[val_idx] = lgbm.predict_proba(X_train[val_idx])[:, 1]
    lgbm_test += lgbm.predict_proba(X_test)[:, 1] / n_folds

# Use lgbm_oof as soft labels for training data
# Use lgbm_test as soft labels for test data (in distillation, treated as pseudo-labels)
```

## Nested 10-in-10 K-Fold for Leak-Free CV

Standard 5-fold CV was insufficient. Chris used a **nested 10-in-10 K-fold**:
- Outer: 10-fold split (provides 10 hold-out sets)
- Inner: For each outer fold's training data, run 10-fold for hyperparameter tuning

This gives leak-free estimates because:
- Outer fold is never seen during inner fitting or hyperparameter selection
- 10x10 = 100 combinations provides very stable CV estimates
- Eliminates the "double-dip" where the same hold-out is used for both model selection and evaluation

```python
from sklearn.model_selection import KFold

outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)
for outer_fold, (outer_tr, outer_val) in enumerate(outer_kf.split(X)):
    # Inner CV only on outer_tr data
    inner_kf = KFold(n_splits=10, shuffle=True, random_state=42)
    inner_scores = []
    for inner_tr, inner_val in inner_kf.split(X[outer_tr]):
        model.fit(X[outer_tr[inner_tr]], y[outer_tr[inner_tr]])
        score = eval(model, X[outer_tr[inner_val]], y[outer_tr[inner_val]])
        inner_scores.append(score)
    # Select hyperparams based on inner_scores
    # Final evaluation on outer_val (never touched during inner)
    final_score = eval(best_model, X[outer_val], y[outer_val])
```

## RAPIDS cuDF: 10–100x Faster Feature Engineering

Feature engineering on 5M+ rows of statement data is slow in pandas. RAPIDS cuDF runs the same DataFrame API on GPU:

```python
import cudf  # GPU DataFrame (same API as pandas)
import cupy as cp  # GPU numpy

# Replace: df = pd.read_parquet('data.parquet')
df = cudf.read_parquet('data.parquet')

# All pandas operations run on GPU
df_grouped = df.groupby('customer_id').agg({'balance': ['mean', 'std', 'last']})

# Convert back to CPU for sklearn/LightGBM
df_cpu = df_grouped.to_pandas()
```

**Speedup**: 10–100x faster depending on operation. Most valuable for:
- Large groupby aggregations (millions of rows × hundreds of features)
- Rolling window computations
- Join operations across large tables

**Requirements**: NVIDIA GPU with CUDA. On little-brother (RTX 2070 Super) or big-brother.

## NN Architecture: LGBM+Transformer Hybrid

```
Input: Customer statement sequence (T × n_features)
→ Feature embedding: Linear(n_features → d_model=256)
→ Positional encoding (learned)
→ TransformerEncoder(nhead=8, num_layers=4, dropout=0.1)
→ [CLS] token or mean pooling
→ Tabular head: Linear(256 → 128) → GELU → Dropout(0.3) → Linear(128 → 1)
```

The Transformer attends to all statement positions simultaneously — different from GRU which processes sequentially. This can capture long-range dependencies (e.g., statement 1 anomaly predicting statement 13 default).

## Key Takeaways
1. LGBM → NN knowledge distillation: train NN on LGBM soft labels first, fine-tune on hard targets
2. 4-cycle cosine schedule alternating distillation/fine-tuning outperforms single-pass distillation
3. Nested 10-in-10 K-fold gives truly leak-free CV estimates
4. RAPIDS cuDF enables GPU-accelerated pandas operations — 10–100x speedup for large feature engineering
5. OOF soft labels for training + test soft labels for distillation (not pseudo-labeling — ground truth still used in fine-tuning)
