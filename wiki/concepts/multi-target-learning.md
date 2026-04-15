---
title: "Multi-Target & Auxiliary Learning"
tags: [multi-target, auxiliary-learning, distillation, ordinal, moa, tabular, nlp]
date: 2026-04-15
source_count: 6
status: active
---

## Summary

Using non-scored targets as auxiliary training signal consistently improves performance on scored targets (+0.002-0.010 gain). The mechanism: auxiliary targets force shared encoder to learn richer features. MoA (+0.003 from 402 non-scored targets), Ventilator (physics-derived derivative/integral auxiliaries), and Jigsaw (toxicity subtypes) all demonstrated significant improvements.

## What It Is

A training paradigm where one model simultaneously predicts multiple targets — some scored, some auxiliary. The auxiliary signal regularizes the shared encoder and forces it to generalize better.

## Key Facts / Details

### 1. MoA Pattern — Non-Scored Columns as Auxiliary Targets

**Competition:** MoA Prediction 2020. 206 scored + 402 non-scored targets.

```python
class MoAMultiTargetModel(nn.Module):
    def __init__(self, n_features, n_scored=206, n_nonscored=402):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 1024), nn.BatchNorm1d(1024), nn.Dropout(0.3), nn.ReLU(),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.Dropout(0.2), nn.ReLU(),
        )
        self.scored_head = nn.Linear(512, n_scored)
        self.nonscored_head = nn.Linear(512, n_nonscored)
    
    def forward(self, x):
        features = self.encoder(x)
        return self.scored_head(features), self.nonscored_head(features)

# Training
scored_pred, nonscored_pred = model(batch_x)
loss = (F.binary_cross_entropy_with_logits(scored_pred, batch_y_scored) +
        0.3 * F.binary_cross_entropy_with_logits(nonscored_pred, batch_y_nonscored))
```

**Auxiliary weight schedule:** Many top solutions anneal auxiliary weight 0.5 → 0.0 during training — heavy signal early, pure scored-target focus late.

**Check your data:** Before training, scan for non-scored columns. Even if not evaluated, include them as auxiliary targets.

### 2. Ventilator — Physics-Derived Auxiliary Targets

```python
def create_physics_auxiliary_targets(pressures):
    dp_dt = np.gradient(pressures, axis=1)        # derivative
    integral_p = np.cumsum(pressures, axis=1)     # integral
    return {'pressure': pressures, 'dp_dt': dp_dt, 'integral_p': integral_p}

# Training loss
loss = (F.l1_loss(pred_pressure, target_pressure) * 1.0 +
        F.l1_loss(pred_dp_dt, target_dp_dt) * 0.2 +
        F.l1_loss(pred_integral, target_integral) * 0.1)
```

**Principle:** Derivatives and integrals of the primary target are "free" auxiliary signals that teach the model about temporal dynamics.

### 3. Teacher-Student Soft Labels (Knowledge Distillation)

```python
def distillation_loss(student_logits, soft_labels, hard_labels, temperature=4.0, alpha=0.3):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        soft_labels,
        reduction='batchmean'
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

**Pattern:** Train large ensemble → generate soft OOF labels → use as training signal for fast single model. Gets 90% of ensemble performance at 1/10 inference cost.

```python
# Generate soft labels from ensemble
soft_labels = np.mean([m.predict_proba(X) for m in ensemble_models], axis=0)
```

### 4. Google QUEST — Engineering Auxiliary Targets

```python
def engineer_auxiliary_targets(df):
    # Composite quality scores
    q_cols = [c for c in df.columns if c.startswith('question_')]
    a_cols = [c for c in df.columns if c.startswith('answer_')]
    
    df['aux_question_quality'] = df[q_cols].mean(axis=1)
    df['aux_answer_quality'] = df[a_cols].mean(axis=1)
    df['aux_qa_interaction'] = df['aux_question_quality'] * df['aux_answer_quality']
    return df
```

When auxiliary labels aren't provided, create them via combinations of existing labels.

### 5. Ordinal Decomposition (K-1 Binary Thresholds)

For ordinal classification (1-5 stars, severity levels):

```python
class OrdinalDecomposition:
    def __init__(self, n_classes=5, base_model_class=None):
        self.models = [base_model_class() for _ in range(n_classes - 1)]
    
    def fit(self, X, y):
        for k, model in enumerate(self.models, start=2):
            model.fit(X, (y >= k).astype(int))  # P(Y >= k)
        return self
    
    def predict_proba(self, X):
        threshold_probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=1)
        n = len(X)
        class_probs = np.zeros((n, len(self.models) + 1))
        class_probs[:, 0] = 1 - threshold_probs[:, 0]
        for k in range(1, len(self.models)):
            class_probs[:, k] = threshold_probs[:, k-1] - threshold_probs[:, k]
        class_probs[:, -1] = threshold_probs[:, -1]
        return np.clip(class_probs, 0, 1)
```

**Why better than standard multiclass:** Respects ordinal structure, each classifier specializes in one boundary.

### Summary: Expected Gains by Technique

| Scenario | Technique | Expected Gain |
|---|---|---|
| Competition provides non-scored columns | Multi-head aux loss | +0.002-0.005 |
| Related labels available (subtypes) | Multi-task shared encoder | +0.003-0.010 |
| Physics/domain-derived targets | Engineer + add aux head | +0.001-0.003 |
| Have a large ensemble trained | Teacher-student distillation | 90% ensemble perf |
| Ordinal classification problem | Threshold decomposition | +0.005-0.015 |
| Large unlabeled data available | Soft pseudo-labels | Significant |

## When To Use It

- Always check if competition data contains non-scored/auxiliary columns
- When you have a clear physical relationship between targets (Ventilator: pressure, derivative, integral)
- When ordinal targets exist (ratings, severity scores)
- After training an ensemble and needing a single fast model for deployment

## Gotchas

- Set auxiliary weight too high → the model focuses on auxiliary at expense of primary target
- Ordinal decomposition requires K-1 models — can be slow; calibrate the threshold probabilities
- Teacher-student distillation: temperature must be tuned — too high = too soft labels, too low = overconfident hard labels
- Multi-task learning can hurt if auxiliary targets are anti-correlated with primary

## Sources

- [[../raw/kaggle/multi-target-auxiliary-learning.md]] — full technique reference with code
- [MoA 1st place writeup](https://www.kaggle.com/competitions/lish-moa/discussion/201510)
- [Ventilator 1st place solution](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285256)
- [Knowledge distillation (Hinton 2015)](https://arxiv.org/abs/1503.02531)

## Related

- [[concepts/knowledge-distillation]] — LGBM→NN distillation in detail
- [[concepts/stacking-deep]] — MoA auxiliary targets in stacking context
- [[concepts/pseudo-labeling]] — pseudo-labeling as another form of auxiliary signal
- [[concepts/deep-learning-tabular]] — neural network architectures for multi-task learning
