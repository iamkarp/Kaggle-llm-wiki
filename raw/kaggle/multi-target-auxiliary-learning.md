# Multi-Target & Auxiliary Learning in Kaggle Competitions

Compiled from MoA, Jigsaw, Ventilator, Google QUEST, Kaggle Playground competition writeups. April 2026.

---

## Core Insight

Using **non-scored targets as auxiliary training signal** consistently improves performance on the scored targets. The mechanism: auxiliary targets force the model to learn richer feature representations that generalize better.

---

## 1. MoA (Mechanisms of Action) — Non-Scored Targets Pattern

**Competition:** Mechanisms of Action 2020 (NeurIPS track)

**Setup:** 206 scored binary targets (drug MoA labels) + 402 non-scored targets.

**Key finding:** Including the 402 non-scored targets during training improved final CV by ~0.003.

```python
import torch
import torch.nn as nn

class MoAMultiTargetModel(nn.Module):
    def __init__(self, n_features, n_scored=206, n_nonscored=402):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        # Shared encoder → two heads
        self.scored_head = nn.Linear(512, n_scored)       # Main task
        self.nonscored_head = nn.Linear(512, n_nonscored) # Auxiliary task
    
    def forward(self, x):
        features = self.encoder(x)
        return self.scored_head(features), self.nonscored_head(features)

# Training with auxiliary loss
def train_step(model, batch_x, batch_y_scored, batch_y_nonscored):
    scored_pred, nonscored_pred = model(batch_x)
    
    loss_scored = F.binary_cross_entropy_with_logits(scored_pred, batch_y_scored)
    loss_nonscored = F.binary_cross_entropy_with_logits(nonscored_pred, batch_y_nonscored)
    
    # Auxiliary weight: start at 0.5, can decay during training
    aux_weight = 0.3
    total_loss = loss_scored + aux_weight * loss_nonscored
    return total_loss

# At inference: only use scored head
model.eval()
scored_pred, _ = model(X_test)
```

**Auxiliary weight schedule:** Many top solutions anneal auxiliary weight from 0.5 to 0.0 during training — heavy auxiliary signal early, pure scored-target focus late.

---

## 2. Jigsaw Toxicity Subtypes — Multi-Label Auxiliary

**Competition:** Jigsaw Unintended Bias in Toxicity Classification

**Setup:** Primary target = toxicity (binary) + 6 subtype labels (threat, obscene, insult, etc.)

**Key insight:** Subtype labels correlated with specific vocabulary patterns. Multi-task learning forces the model to separate toxicity types, reducing spurious correlations.

```python
class JigsawMultiTask(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.bert = pretrained_model
        hidden = self.bert.config.hidden_size  # 768
        
        self.toxicity_head = nn.Linear(hidden, 1)
        self.subtype_heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(6)])
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # CLS token
        
        toxicity = self.toxicity_head(cls)
        subtypes = [head(cls) for head in self.subtype_heads]
        return toxicity, subtypes
```

---

## 3. Google QUEST Q&A Labeling — Engineered Auxiliary Targets

**Competition:** Google QUEST Q&A Labeling (NeurIPS 2020)

**Setup:** 30 question+answer quality targets.

**Key technique:** Engineer auxiliary targets from existing labels:
- `answer_relevance_consistency` = (`answer_relevance` + `answer_helpful` - `answer_misleading`) / 3
- `question_quality_composite` = average of question-side targets
- `interaction_quality` = product of top question score × top answer score

```python
def engineer_auxiliary_targets(df):
    # Consistency targets
    df['aux_answer_consistency'] = (
        df['answer_relevance'] + df['answer_helpful'] - df['answer_misleading']
    ) / 3
    
    # Composite quality scores
    question_cols = [c for c in df.columns if c.startswith('question_')]
    answer_cols   = [c for c in df.columns if c.startswith('answer_')]
    
    df['aux_question_quality'] = df[question_cols].mean(axis=1)
    df['aux_answer_quality']   = df[answer_cols].mean(axis=1)
    
    # Cross-interaction
    df['aux_qa_interaction'] = df['aux_question_quality'] * df['aux_answer_quality']
    
    return df
```

---

## 4. Ventilator Pressure Prediction — Derivative + Integral Auxiliaries

**Competition:** Google Brain Ventilator Pressure Prediction 2021

**Problem:** Predict pressure at each time step from ventilator breath controls.

**Auxiliary targets:** Physics-derived:
- **Derivative target:** `dp/dt` — rate of pressure change
- **Integral target:** `∫p dt` — cumulative pressure over breath

```python
import numpy as np

def create_physics_auxiliary_targets(pressures):
    """
    pressures: array of shape (n_samples, seq_len)
    Returns: original + derivative + integral targets
    """
    # Derivative (central difference)
    dp_dt = np.gradient(pressures, axis=1)
    
    # Integral (cumulative sum, approximating ∫p dt)
    integral_p = np.cumsum(pressures, axis=1)
    
    return {
        'pressure': pressures,      # primary target
        'dp_dt': dp_dt,             # auxiliary: derivative
        'integral_p': integral_p,   # auxiliary: integral
    }

class VentilatorModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=256):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, 
                           batch_first=True, dropout=0.1, bidirectional=True)
        hidden_total = hidden_size * 2  # bidirectional
        
        self.pressure_head = nn.Linear(hidden_total, 1)   # Primary
        self.dp_dt_head = nn.Linear(hidden_total, 1)      # Auxiliary 1
        self.integral_head = nn.Linear(hidden_total, 1)   # Auxiliary 2
    
    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq, hidden*2)
        return (self.pressure_head(out).squeeze(-1),
                self.dp_dt_head(out).squeeze(-1),
                self.integral_head(out).squeeze(-1))
```

**Loss:**
```python
loss = (
    F.l1_loss(pred_pressure, target_pressure) * 1.0 +
    F.l1_loss(pred_dp_dt, target_dp_dt) * 0.2 +
    F.l1_loss(pred_integral, target_integral) * 0.1
)
```

---

## 5. Teacher-Student Soft Labels

**Concept:** Use a larger/stronger model's probability outputs as soft labels for training a smaller model.

**Why it works:** Soft labels encode the model's uncertainty — "70% class A, 28% class B, 2% class C" — providing much richer training signal than hard labels.

```python
def create_soft_labels(teacher_model, X, temperature=4.0):
    """
    temperature > 1 softens the distribution (more inter-class info)
    """
    teacher_model.eval()
    with torch.no_grad():
        logits = teacher_model(torch.tensor(X, dtype=torch.float32))
        soft_labels = F.softmax(logits / temperature, dim=-1)
    return soft_labels.numpy()

def distillation_loss(student_logits, soft_labels, hard_labels, 
                      temperature=4.0, alpha=0.3):
    """
    alpha: weight for hard label loss (vs soft)
    """
    # Soft loss (knowledge distillation)
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        soft_labels,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard loss (standard cross-entropy)
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss
```

**Kaggle usage:** Train large ensemble → generate soft OOF labels → use as training signal for single fast model. Gets 90% of ensemble performance at 1/10 inference cost.

---

## 6. Ordinal Decomposition (Threshold Decomposition)

**Use case:** Ordinal classification (1-5 star ratings, severity levels).

**Method:** Decompose K-class ordinal problem into K-1 binary threshold problems:
- `P(Y >= 2)`, `P(Y >= 3)`, `P(Y >= 4)`, `P(Y >= 5)` → 4 binary classifiers

```python
class OrdinalDecomposition:
    def __init__(self, n_classes=5, base_model_class=None):
        self.n_classes = n_classes
        self.models = [base_model_class() for _ in range(n_classes - 1)]
    
    def fit(self, X, y):
        for k, model in enumerate(self.models, start=2):
            # Binary target: P(Y >= k)
            binary_y = (y >= k).astype(int)
            model.fit(X, binary_y)
        return self
    
    def predict_proba(self, X):
        # P(Y >= k) for k = 2, ..., K
        threshold_probs = np.stack([m.predict_proba(X)[:, 1] for m in self.models], axis=1)
        
        # Convert to class probabilities
        n_samples = len(X)
        class_probs = np.zeros((n_samples, self.n_classes))
        
        # P(Y=1) = 1 - P(Y>=2)
        class_probs[:, 0] = 1 - threshold_probs[:, 0]
        # P(Y=k) = P(Y>=k) - P(Y>=k+1) for k = 2,...,K-1
        for k in range(1, self.n_classes - 1):
            class_probs[:, k] = threshold_probs[:, k-1] - threshold_probs[:, k]
        # P(Y=K) = P(Y>=K)
        class_probs[:, -1] = threshold_probs[:, -1]
        
        return np.clip(class_probs, 0, 1)
```

**Why ordinal decomposition beats standard multiclass:**
1. Respects ordinal structure (class 3 is between 2 and 4)
2. Each binary classifier specializes in one threshold boundary
3. Ensemble of K-1 specialized classifiers > one generalist

---

## Summary: When to Use Auxiliary Learning

| Scenario | Technique | Expected Gain |
|---|---|---|
| Competition provides non-scored columns | Multi-head with aux loss | +0.002-0.005 |
| Related labels available (subtypes) | Multi-task shared encoder | +0.003-0.010 |
| Physics/domain-derived targets | Engineer + add aux head | +0.001-0.003 |
| Have a large ensemble trained | Teacher-student distillation | 90% ensemble perf |
| Ordinal classification problem | Threshold decomposition | +0.005-0.015 |
| Large unlabeled data available | Teacher-student soft pseudo-labels | Significant |

---

Sources:
- MoA competition 1st place: https://www.kaggle.com/competitions/lish-moa/discussion/201510
- Jigsaw Unintended Bias top solutions: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/discussion
- Google QUEST labeling: https://www.kaggle.com/competitions/google-quest-challenge/discussion
- Ventilator Pressure Prediction 1st place: https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285256
- Knowledge distillation original paper (Hinton 2015): https://arxiv.org/abs/1503.02531
- Ordinal regression decomposition: https://www.cs.uic.edu/~eyal/papers/ordinal.pdf
