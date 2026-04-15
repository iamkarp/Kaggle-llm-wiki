# Kaggle Playground Series S5 & S6 Winning Solutions (2025-2026)

Compiled from competition writeups, forum discussions, and notebook analyses. April 2026.

---

## S5E4 — Multi-Target Prediction + 3-Level Stack

**Task:** Predict 4 correlated targets simultaneously.
**Key techniques:**
- Train separate models per target AND joint multi-output models, then ensemble both approaches
- 3-level stacking: Level 1 = GBDT per target, Level 2 = NN on OOF, Level 3 = Ridge blend
- Feature interactions engineered per target (not shared global FE)
- Correlation between targets exploited via target-as-feature with nested CV

---

## S5E5 — GPU Hill Climbing

**Task:** Tabular regression.
**Key insight:** GPU-accelerated hill climbing for ensemble weight optimization.

```python
# GPU hill climbing for ensemble weights
import cupy as cp

def gpu_hill_climb(oof_preds_list, y_true, n_iter=10000):
    """
    oof_preds_list: list of numpy arrays (one per model)
    """
    n_models = len(oof_preds_list)
    preds_gpu = [cp.array(p) for p in oof_preds_list]
    y_gpu = cp.array(y_true)
    
    weights = cp.ones(n_models) / n_models
    best_score = evaluate(cp.average(preds_gpu, axis=0, weights=weights), y_gpu)
    
    for _ in range(n_iter):
        i = cp.random.randint(0, n_models)
        delta = cp.random.uniform(-0.05, 0.05)
        new_weights = weights.copy()
        new_weights[i] += delta
        new_weights = cp.maximum(new_weights, 0)
        new_weights /= new_weights.sum()
        
        score = evaluate(cp.average(preds_gpu, axis=0, weights=new_weights), y_gpu)
        if score > best_score:
            weights = new_weights
            best_score = score
    
    return cp.asnumpy(weights), float(best_score)
```

**Measured gain:** GPU hill climbing over 10K iterations in <30 seconds vs CPU taking minutes.

---

## S5E6 — 100-Seed Averaging (+0.003 on metric)

**Task:** Binary classification.
**Key technique:** Seed averaging at scale.

```python
# 100-seed averaging pattern
import numpy as np

all_oof_preds = []
all_test_preds = []

for seed in range(100):
    model = LGBMClassifier(random_state=seed, **best_params)
    oof, test_preds = run_cv(model, X, y, X_test, seed=seed)
    all_oof_preds.append(oof)
    all_test_preds.append(test_preds)

# Average gives +0.003 on AUC vs single seed
oof_avg = np.mean(all_oof_preds, axis=0)
test_avg = np.mean(all_test_preds, axis=0)
```

**Why it works:** Each seed produces slightly different tree structures due to feature subsampling. Averaging 100 predictions dramatically reduces variance without bias.

**Practical note:** cuML GPU-accelerated LightGBM allows 100 seeds in ~10 minutes vs ~3 hours on CPU.

---

## S5E11 — SMOTE Leakage Case Study

**Task:** Binary classification (imbalanced).
**Warning documented:** SMOTE applied BEFORE cross-validation inflated CV by +0.16 AUC.

```python
# WRONG — inflates CV by ~0.16 AUC
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
cv = cross_val_score(clf, X_res, y_res, cv=5)  # Looks great, crashes on LB

# CORRECT
pipe = Pipeline([('smote', SMOTE()), ('clf', clf)])
cv = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5))
```

**Reference:** ArXiv 2412.07437 confirmed +0.16 AUC inflation from pre-split SMOTE.

---

## S5E12 — Hill Climbing + Ridge Blend

**Post-cutoff CV trick:**
```python
# Use data after public LB cutoff as independent validation
lb_cutoff_date = '2024-11-15'
post_cutoff_mask = df['date'] > lb_cutoff_date
X_post = X[post_cutoff_mask]
y_post = y[post_cutoff_mask]

# This data was NOT used for public LB scoring — truly independent validation
```

**Findings:**
- CV improvements on post-cutoff data → reliable private LB signal
- Hill climbing on OOF + Ridge on post-cutoff: 2-stage validation avoids overfitting ensemble weights

---

## S6E1 — NNs Beat GBMs

**Anomalous result:** Neural networks outperformed GBDTs on this competition.
**Why:** Dense interaction patterns in the feature space suited deep learning better.

**Winning architecture:**
```python
# TabNet or ResNet-style NN was dominant
import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(x + self.layers(x))

class TabularResNet(nn.Module):
    def __init__(self, n_features, n_blocks=6, dim=512):
        super().__init__()
        self.embedding = nn.Linear(n_features, dim)
        self.blocks = nn.ModuleList([ResBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks: x = block(x)
        return self.head(x)
```

---

## S6E2 — Multi-Seed + Trust CV

**Competition lesson:** Public LB was misleading (only 30% of test data).

**CV-LB breakdown threshold:**
```
Observation: after optimizing 15 parameters, CV improvements stopped correlating with LB
Finding: further optimization past this point was harmful (overfit to 30% public sample)
Rule: Track CV improvement vs LB correlation — when 2+ consecutive CV gains don't move LB, STOP
```

**Multi-seed ensemble:** 20 seeds × 5-fold CV = 100 models averaged; private LB gain confirmed.

---

## S6E3 — LLM-Augmented Tabular

**Task:** Tabular prediction with text/categorical features.
**Novel approach:** Use GPT-4/Gemini/Claude to generate semantic features from categorical values.

```python
import anthropic

client = anthropic.Anthropic()

def generate_semantic_features(row_text: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{
            "role": "user",
            "content": f"Given this data: {row_text}\nExtract: sentiment (1-5), formality (1-5), risk_level (low/medium/high), domain (finance/tech/health/other)"
        }]
    )
    # Parse structured output
    return parse_response(response.content[0].text)

# Cache results (expensive to generate)
semantic_df = df['text_col'].apply(lambda x: generate_semantic_features(x))
```

**Measured gains:** LLM-derived features gave +0.002-0.005 on public LB. Worth it when text columns are present.

**Cost tip:** Generate once, cache to parquet, reuse across all models.

---

## Cross-Competition Patterns (S5/S6)

| Competition | Key Technique | Measured Gain |
|---|---|---|
| S5E5 | GPU hill climbing | ~0.001-0.002 |
| S5E6 | 100-seed averaging | +0.003 |
| S5E11 | SMOTE correctness | Avoid -0.16 |
| S5E12 | Post-cutoff CV | Better model selection |
| S6E1 | ResNet over GBDT | Best on dense interactions |
| S6E2 | Trust CV over LB | Avoided private LB disaster |
| S6E3 | LLM feature augmentation | +0.002-0.005 |

---

Sources:
- Kaggle Playground S5 writeups: https://www.kaggle.com/competitions/playground-series-s5e6/writeups/
- S5E12 post-cutoff validation technique
- SMOTE leakage arXiv: https://arxiv.org/html/2412.07437v1
- S6E3 LLM augmentation discussion threads
