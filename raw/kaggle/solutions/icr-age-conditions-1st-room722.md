# ICR — Identifying Age-Related Conditions — 1st Place Solution
**Author**: room722 | **Year**: 2023 | **Votes**: 318

---

## Competition
Binary classification: predict whether a patient has any of three age-related medical conditions, from anonymized health features. Log-loss metric. ~617 training rows (extremely small dataset), 56 features. Key challenge: tiny dataset makes GBMs prone to overfitting; standard NN practices also fail.

## Core Insight: DNN Beat GBMs on a Tiny Dataset

Counter-intuitive result: a deep neural network outperformed LightGBM, XGBoost, and CatBoost on 617 rows. This challenges the conventional wisdom that "trees always beat NNs on tabular data."

**Why it worked here**:
- Dataset is extremely small — GBMs overfit more severely than expected
- The anonymized features had very specific structure that the Variable Selection Network (VSN) could exploit
- Careful regularization (extreme dropout) prevented NN overfitting
- Repeated training with best-of-N selection (cherry-picking) compensated for NN instability

## Variable Selection Network (VSN) Architecture

VSN is a component from the Temporal Fusion Transformer paper (Lim et al., 2020), adapted here for tabular data. It learns per-feature importance weights.

```
For each feature i:
    Linear projection: x_i → h_i (8 neurons)  [learned per-feature transformation]
    GRN (Gated Residual Network):
        h_i → Dense(8, ELU) → Dense(8) → gate (sigmoid-gated residual)
    
Attention over projected features:
    Softmax weights V_i for each feature → v = Σ V_i * h_i  [weighted sum]
    
Final: v → prediction head
```

The key component: each feature gets its own 8-neuron learned linear projection. This is richer than simple normalization — the model learns what transformation of each feature is most informative.

```python
import torch
import torch.nn as nn

class GRN(nn.Module):
    """Gated Residual Network unit"""
    def __init__(self, input_size, hidden_size, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.gate = nn.Linear(input_size, input_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_size)
    
    def forward(self, x):
        h = torch.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        g = torch.sigmoid(self.gate(x))
        return self.norm(g * h + (1 - g) * x)

class VSN(nn.Module):
    def __init__(self, n_features, feature_dim=8):
        super().__init__()
        # Per-feature linear projections
        self.projections = nn.ModuleList([
            nn.Linear(1, feature_dim) for _ in range(n_features)
        ])
        self.grns = nn.ModuleList([
            GRN(feature_dim, feature_dim) for _ in range(n_features)
        ])
        self.attention = nn.Linear(feature_dim, 1)  # learned feature importance
        
    def forward(self, x):
        # x: (batch, n_features)
        feature_outputs = []
        for i, (proj, grn) in enumerate(zip(self.projections, self.grns)):
            h = proj(x[:, i:i+1])  # (batch, feature_dim)
            h = grn(h)
            feature_outputs.append(h)
        
        # Stack: (batch, n_features, feature_dim)
        stacked = torch.stack(feature_outputs, dim=1)
        
        # Attention weights over features
        attn_weights = torch.softmax(self.attention(stacked), dim=1)  # (batch, n_features, 1)
        
        # Weighted sum across features
        output = (attn_weights * stacked).sum(dim=1)  # (batch, feature_dim)
        return output
```

## Learned Per-Feature Projections vs. Standard Scaling

**Standard approach**: Scale each feature with StandardScaler or RankGauss → feed raw scaled values into NN.

**VSN approach**: Each feature gets its own 8-neuron linear projection learned jointly with the rest of the network. This allows the model to learn optimal transformations (e.g., for some features, the square might matter more than the raw value) without explicit feature engineering.

**Key difference**: Standard scaling is a fixed, data-derived transformation. VSN projections are learned, task-specific transformations. For small datasets with anonymized features, this distinction matters.

## Extreme Dropout Rates

Unusually high dropout to combat overfitting on tiny dataset:

```
Layer 1: 617 samples → Dense(512) → Dropout(0.75)  ← drop 75% of neurons!
Layer 2:              → Dense(256) → Dropout(0.5)
Layer 3:              → Dense(128) → Dropout(0.25)
Output:              → Dense(1, Sigmoid)
```

Standard dropout is 0.1–0.3. Using 0.75 on a 617-sample dataset is extreme. This works because:
- With so few samples, the model must generalize to survive 75% dropout
- Each training step uses a different random 25% of the first layer — massive implicit ensemble
- The effective model capacity is dramatically reduced without shrinking the architecture

## Repeated Training: 10–30× per Fold, Cherry-Pick 2 Best

NNs on tiny datasets have high variance — different random seeds produce very different results. room722's solution:

1. Train the model 10–30 times per fold with different seeds
2. Evaluate each training run on the validation fold
3. Keep the 2 runs with the lowest validation loss
4. Average their predictions (2-seed ensemble per fold)

```python
fold_models = []
for seed in range(30):
    set_seed(seed)
    model = VSNModel()
    model.fit(X_train_fold, y_train_fold)
    val_loss = evaluate(model, X_val_fold, y_val_fold)
    fold_models.append((val_loss, model))

# Keep 2 best
fold_models.sort(key=lambda x: x[0])
best_2 = [m for _, m in fold_models[:2]]
fold_preds = np.mean([m.predict(X_val_fold) for m in best_2], axis=0)
```

**Why this works**: On tiny datasets, weight initialization dominates outcome. Running 30 seeds and picking the 2 best reduces initialization variance by selecting the "lucky" initializations that found better basins. This is a form of model selection / cherry-picking that's risky on larger datasets but necessary here.

## "Hardness to Predict" Multi-Label CV Trick

The competition secretly had 3 sub-conditions (A, B, C) that were revealed later. room722 discovered a CV trick:

For each sample, compute a "hardness score" — how inconsistently models predict this sample across different folds. Hard-to-predict samples (high variance across folds) are used to create a stratified split that ensures each fold has roughly the same distribution of easy and hard samples.

```python
# Train models on each fold, collect OOF predictions
oof_preds = cross_val_predict(model, X, y, cv=5, method='predict_proba')

# Hardness = std of OOF prediction across folds (requires fold-by-fold preds)
# Or: hardness = entropy of prediction histogram
sample_hardness = np.std(fold_predictions_per_sample, axis=0)

# Bin hardness into categories and stratify on joint (label, hardness_bin)
hardness_bins = pd.qcut(sample_hardness, q=3, labels=['easy','medium','hard'])
stratify_key = y.astype(str) + '_' + hardness_bins.astype(str)
```

This ensures validation folds aren't accidentally all "easy" samples — a common problem with very small, imbalanced datasets.

## Key Takeaways
1. DNNs can beat GBMs on tiny tabular datasets with proper regularization
2. Variable Selection Networks learn per-feature transformations — more expressive than static scaling
3. Extreme dropout (0.75→0.5→0.25) is the right regularization for <1000 row datasets
4. Repeat training 10–30x per fold; cherry-pick 2 best per fold — compensates for NN initialization variance
5. "Hardness to predict" stratification ensures balanced CV splits on difficult small datasets
6. 8-neuron per-feature projections outperform standard normalization on anonymized features
