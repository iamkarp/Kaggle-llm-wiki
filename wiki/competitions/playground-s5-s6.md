---
id: competition:playground-s5-s6
type: competition
title: Kaggle Playground Series S5 & S6 Winning Patterns (2025-2026)
slug: playground-s5-s6
aliases: []
tags:
- kaggle
- playground
- seed-averaging
- gpu-hill-climbing
- smote-leakage
- llm-augmentation
- tabular
status: active
date: 2026-04-15
source_count: 2
---

## Competition Metadata

- **Platform:** Kaggle Playground Series (Season 5, Season 6)
- **Period:** 2025-2026
- **Format:** Monthly tabular competitions, free to enter

## Summary

GPU hill climbing (S5E5), 100-seed averaging (+0.003, S5E6), and LLM-augmented tabular features (S5E6 S6E3) are the defining patterns of 2025-2026 Playground competitions. SMOTE before CV inflated a competitor's AUC by +0.16 in S5E11 — the most documented recent leakage incident. S6E2 demonstrated the CV-LB breakdown threshold in practice.

## Key Winning Patterns

### S5E5 — GPU Hill Climbing

GPU-accelerated hill climbing enables searching 10,000+ weight combinations in under 30 seconds:

```python
import cupy as cp

def gpu_hill_climb(oof_preds_list, y_true, n_iter=10000):
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

### S5E6 — 100-Seed Averaging (+0.003 AUC)

```python
all_preds = []
for seed in range(100):
    model = LGBMClassifier(random_state=seed, **best_params)
    oof, test_preds = run_cv(model, X, y, X_test, seed=seed)
    all_preds.append(test_preds)

final_preds = np.mean(all_preds, axis=0)  # +0.003 vs single seed
```

cuML GPU-accelerated LightGBM completes 100 seeds in ~10 minutes vs ~3 CPU hours.

### S5E11 — SMOTE Leakage Case Study

Most documented instance of SMOTE-before-CV leakage in 2025:

```python
# WRONG: inflated CV by +0.16 AUC
X_res, y_res = SMOTE().fit_resample(X_train, y_train)
cv = cross_val_score(clf, X_res, y_res, cv=5)

# CORRECT: SMOTE inside pipeline
pipe = Pipeline([('smote', SMOTE()), ('clf', clf)])
cv = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5))
```

### S5E12 — Post-Cutoff CV Validation

```python
lb_cutoff_date = '2024-11-15'
post_cutoff_mask = df['date'] > lb_cutoff_date

# Post-cutoff data = truly independent validation (not used for public LB)
X_post = X[post_cutoff_mask]
y_post = y[post_cutoff_mask]
```

Hill climbing on OOF + Ridge on post-cutoff: 2-stage validation prevents ensemble weight overfitting.

### S6E1 — NNs Beat GBMs

ResNet architecture outperformed all GBDT models — unusual for Playground series.

```python
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim)
        )
    def forward(self, x):
        return F.relu(x + self.layers(x))

class TabularResNet(nn.Module):
    def __init__(self, n_features, n_blocks=6, dim=512):
        super().__init__()
        self.embedding = nn.Linear(n_features, dim)
        self.blocks = nn.ModuleList([ResBlock(dim) for _ in range(n_blocks)])
        self.head = nn.Linear(dim, 1)
```

### S6E2 — CV-LB Breakdown in Practice

After optimizing 15+ parameters, CV improvements stopped correlating with LB gains. Key lesson:
- Track CV vs LB correlation throughout optimization
- When 2+ consecutive CV gains don't move LB → stop optimizing
- Multi-seed (20 seeds × 5-fold = 100 models) averaging confirmed private LB gain

### S6E3 — LLM-Augmented Tabular Features

```python
import anthropic

client = anthropic.Anthropic()

def generate_semantic_features(row_text: str) -> dict:
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=500,
        messages=[{"role": "user", "content": 
                   f"Given: {row_text}\nExtract: sentiment(1-5), formality(1-5), risk_level(low/medium/high), domain"}]
    )
    return parse_response(response.content[0].text)

# Cache results (expensive)
semantic_df = df['text_col'].apply(generate_semantic_features)
```

**Measured gain:** +0.002-0.005 on public LB when text columns present.

## What Worked / What Didn't

**Worked:**
- GPU hill climbing for ensemble weight optimization
- 100-seed averaging with cuML GPU acceleration
- Post-cutoff CV as independent validation
- LLM-augmented features for competitions with text columns
- ResNet architecture when feature interactions are dense

**Didn't Work:**
- SMOTE before split (documented +0.16 CV inflation that crashed on private LB)
- Optimizing past the CV-LB breakdown threshold (S6E2)
- Trusting public LB when it represents only 30% of test data

## Submission History / Key Results

| Competition | Key Technique | Measured Gain |
|---|---|---|
| S5E5 | GPU hill climbing | +0.001-0.002 AUC |
| S5E6 | 100-seed averaging | +0.003 AUC |
| S5E11 | SMOTE correct usage | Avoid -0.16 AUC |
| S5E12 | Post-cutoff CV | Better model selection |
| S6E1 | ResNet over GBDT | Best on dense interactions |
| S6E2 | Trust CV over LB | Avoided private LB disaster |
| S6E3 | LLM feature augmentation | +0.002-0.005 AUC |

## Sources

- `raw/kaggle/playground-s5-s6-winning-solutions.md` *(not yet ingested)* — full solution details
- [Kaggle Playground Series S5](https://www.kaggle.com/competitions/playground-series-s5e6/writeups/)
- [SMOTE leakage arXiv](https://arxiv.org/html/2412.07437v1)

## Related

- [[../concepts/ensembling-strategies]] — hill climbing implementation
- [[../concepts/validation-strategy]] — CV-LB breakdown, post-cutoff CV
- [[../concepts/imbalanced-data]] — SMOTE correct usage
- [[../concepts/gradient-boosting-advanced]] — cuML GPU acceleration for seed averaging
- [[../strategies/kaggle-meta-strategy]] — trust CV, CV-LB breakdown threshold

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _uses_ → [[entities/lightgbm-catboost|LightGBM & CatBoost — Gradient Boosting Alternatives]]
- _hosted_by_ → `organization:kaggle` (Kaggle)

**Incoming:**
- [[concepts/ensembling-strategies|Ensembling Strategies — Fourth-Root Blend, Stacking, Diversity]] _applied_in_ → here
- [[concepts/gradient-boosting-advanced|Gradient Boosting — Advanced Configuration Tricks]] _applied_in_ → here
- [[concepts/imbalanced-data|Imbalanced Data Techniques for Kaggle]] _applied_in_ → here
- [[concepts/validation-strategy|Validation Strategy — CV Design, Gap Tracking, Anti-Patterns]] _applied_in_ → here
- [[strategies/kaggle-meta-strategy|Kaggle Meta-Strategy — Grandmaster Principles for Any Competition]] _applied_in_ → here
- [[index|Wiki Index]] _related_to_ → here

<!-- kg:end -->
