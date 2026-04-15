---
title: "Pseudo-Labeling — Semi-Supervised Learning with High-Confidence Test Predictions"
tags: [pseudo-labeling, semi-supervised, test-augmentation, imbalanced, confidence-threshold]
date: 2026-04-15
source_count: 2
status: active
---

## What It Is
Pseudo-labeling is a semi-supervised technique: train an initial model on labeled training data, assign predicted labels to unlabeled test data with high confidence, then retrain on training + pseudo-labeled test data. It effectively expands the training set using the test distribution.

Pioneered in NLP (self-training), adopted widely in Kaggle competitions with heavily imbalanced or distribution-shifted test data.

## When It Works

**Good conditions**:
- Initial model has high confidence on a significant fraction of test rows (prob > 0.95 or < 0.05)
- Test distribution is different from training (pseudo-labeling adapts to test distribution)
- Large test set relative to training (more pseudo-labels = more gain)
- Imbalanced classes (pseudo-labels can help balance or reinforce minority class signal)
- The data generation process produces clusters of clearly positive/negative examples

**Bad conditions**:
- Initial model is noisy (error propagates into pseudo-labels → error amplification)
- Test and train come from the same distribution (no distribution shift to adapt to)
- Small test set (few pseudo-labels, not worth the complexity)
- Decision boundary is ambiguous — most test predictions will be near 0.5

## Standard Implementation

```python
import numpy as np

def pseudo_label_round(model, X_train, y_train, X_test,
                       pos_threshold=0.95, neg_threshold=0.05, 
                       max_pseudo_ratio=0.5):
    """
    One round of pseudo-labeling.
    max_pseudo_ratio: don't let pseudo-labels exceed this fraction of training set size.
    """
    # Step 1: get initial predictions
    test_probs = model.predict_proba(X_test)[:, 1]
    
    # Step 2: select high-confidence test rows
    pos_mask = test_probs >= pos_threshold
    neg_mask = test_probs <= neg_threshold
    pseudo_mask = pos_mask | neg_mask
    
    n_pseudo = pseudo_mask.sum()
    max_pseudo = int(len(X_train) * max_pseudo_ratio)
    
    if n_pseudo > max_pseudo:
        # If too many, take the most confident ones
        confidence = np.abs(test_probs - 0.5)
        top_idx = np.argsort(confidence)[::-1][:max_pseudo]
        pseudo_idx = top_idx[pseudo_mask[top_idx]]
    else:
        pseudo_idx = np.where(pseudo_mask)[0]
    
    pseudo_labels = (test_probs[pseudo_idx] >= 0.5).astype(int)
    
    print(f"Pseudo-labels: {pseudo_mask.sum()} total | "
          f"{pos_mask.sum()} positive, {neg_mask.sum()} negative | "
          f"Using {len(pseudo_idx)}")
    
    # Step 3: augment training set
    X_aug = np.vstack([X_train, X_test[pseudo_idx]])
    y_aug = np.concatenate([y_train, pseudo_labels])
    
    return X_aug, y_aug

# Usage: iterate 1-3 rounds
for round_i in range(2):
    model.fit(X_aug, y_aug)
    X_aug, y_aug = pseudo_label_round(model, X_train, y_train, X_test)
```

## Santander Competition Specifics

In Santander Transaction 1st place (fl2o), pseudo-labeling was particularly effective because:
1. The uniqueness features (5 categories) produced very high-confidence predictions on clearly positive/negative test rows
2. Target == 1 examples are rare (10%) — pseudo-labels for positives were especially valuable
3. The attention NN treated all features symmetrically, so pseudo-labeled rows were reliable

## Confidence Threshold Selection

The threshold (0.95/0.05) is a hyperparameter. Guidelines:
- Start with 0.90/0.10 as a conservative default
- If initial model AUC is < 0.85: tighten to 0.95/0.05 (model is less reliable)
- If initial model AUC is > 0.95: can use 0.85/0.15 (model is reliable enough)
- Monitor: plot the distribution of pseudo-label confidences to see if thresholds make sense

## Multiple Pseudo-Label Rounds

After round 1, retrain on augmented data → get new predictions → new pseudo-labels. Typically run 2–3 rounds. Each round may find different high-confidence examples. Stop when:
- No new high-confidence examples are being found
- CV score plateaus or degrades
- Pseudo-label count grows beyond ~30% of original training set

## Pseudo-Labeling vs. Full Test Inclusion

Some competitors include ALL test rows with soft labels (predicted probabilities) rather than just high-confidence rows. This is more aggressive and risks error propagation.

| Approach | Risk | Gain |
|----------|------|------|
| Hard pseudo-labels (high-conf only) | Low | Moderate |
| Soft labels (all test) | High | Potentially large |
| Full test with distillation loss | Medium | Large |

For most competitions: start with hard pseudo-labels on high-confidence rows.

## Validation Challenge

Pseudo-labeling creates a subtle CV problem: your validation fold doesn't contain pseudo-labeled rows, but your training fold does. This can cause CV to be optimistic (model sees test distribution in training but not val).

**Mitigation**: When measuring effect of pseudo-labeling, compare models trained with/without pseudo-labels using the same held-out validation set — not K-fold CV.

## Sources
- [[../../raw/kaggle/solutions/santander-transaction-1st-fl2o.md]] — pseudo-labeling combined with uniqueness features and shuffle augmentation
- [[../../raw/kaggle/kaggle-competition-playbook.md]] — semi-supervised learning context

## Related
- [[../concepts/validation-strategy]] — CV complications introduced by pseudo-labeling
- [[../concepts/ensembling-strategies]] — pseudo-labeling is often combined with ensemble models
- [[../concepts/denoising-autoencoders]] — DAE can be used on pseudo-labeled data for representation learning
