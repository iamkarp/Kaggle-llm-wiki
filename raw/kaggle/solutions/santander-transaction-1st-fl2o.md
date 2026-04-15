# Santander Customer Transaction Prediction — 1st Place Solution
**Author**: fl2o | **Votes**: 384

---

## Competition
Predict whether a Santander customer will make a specific transaction. Binary classification, AUC metric. 200 anonymized features, 200K train rows. Key challenge: severe class imbalance (~10% positive), all features look identical in raw form (same range, no obvious structure).

## Core Insight: Value Uniqueness Features

**The breakthrough**: Each of the 200 features had a bimodal distribution in the test set but not the training set. Some values were "unique" to test, some appeared multiple times. fl2o discovered that this uniqueness encodes information about the test data generation process.

**Five categories of uniqueness** were engineered:

### Category 1: Is Value Unique in Test?
```python
# For each feature, count how many times each value appears in test
test_value_counts = test[col].value_counts()
train[f'{col}_unique_in_test'] = train[col].map(test_value_counts) == 1
```
Values that appear exactly once in test are "unique" — they behave differently from values that appear multiple times.

### Category 2: Is Value Unique in Train?
```python
train_value_counts = train[col].value_counts()
train[f'{col}_unique_in_train'] = train[col].map(train_value_counts) == 1
```

### Category 3: Count of Occurrences in Test
The raw count (not just binary unique/not):
```python
train[f'{col}_test_count'] = train[col].map(test_value_counts).fillna(0)
```

### Category 4: Count of Occurrences in Train
```python
train[f'{col}_train_count'] = train[col].map(train_value_counts)
```

### Category 5: Cross-Set Uniqueness
Whether a train value appears in test and vice versa:
```python
test_values = set(test[col].values)
train[f'{col}_in_test'] = train[col].isin(test_values).astype(int)
```

**Why it works**: The test data contained duplicated rows with flipped target-relevant features. Unique values indicate "synthetic" test rows; multi-occurrence values indicate real customers. The uniqueness pattern leaked the data generation mechanism.

## Attention-Style Neural Network

Rather than using feature importance to select/weight features, fl2o built an attention-style NN where all 200 features are treated identically with **learned weighted averaging**.

### Architecture
```
Input: 200 features (each as a scalar)
↓
Feature-wise embedding: each feature × learnable weight
↓
Attention layer: softmax over 200 weights → weighted average
↓
Dense(256, ReLU) → Dropout(0.3)
→ Dense(128, ReLU) → Dropout(0.3)
→ Dense(64, ReLU)
→ Dense(1, Sigmoid)
```

**Key property**: The attention weights are learned jointly with the rest of the network. This is parameter-efficient for 200 symmetric features — the model discovers which features matter most without hand-tuning.

**Why symmetric treatment**: All 200 features had similar distributions; no domain knowledge distinguished them. Treating them identically with learned attention was more principled than arbitrary ordering.

## Pseudo-Labeling

After training initial models with high confidence, use pseudo-labels on test data to augment training:

1. Train model on labeled training data → get test predictions
2. Filter test rows with very high confidence (e.g., predicted prob > 0.95 or < 0.05)
3. Assign pseudo-labels to these high-confidence test rows
4. Retrain model on original training + pseudo-labeled test rows
5. Repeat 1–2 times

```python
# Step 1: get initial predictions
initial_preds = model.predict_proba(X_test)[:, 1]

# Step 2: select high-confidence
high_conf_mask = (initial_preds > 0.95) | (initial_preds < 0.05)
pseudo_labels = (initial_preds[high_conf_mask] > 0.5).astype(int)

# Step 3: augment training set
X_train_aug = np.vstack([X_train, X_test[high_conf_mask]])
y_train_aug = np.concatenate([y_train, pseudo_labels])

# Step 4: retrain
model_v2 = train_model(X_train_aug, y_train_aug)
```

**When it helps**: When the test distribution has clear structure (as here — uniqueness features made high-confidence predictions reliable).
**When it hurts**: When initial model is noisy → pseudo-labels propagate errors.

## Shuffle Augmentation

For imbalanced classification, fl2o augmented the training set by shuffling feature values within rows:

- **For target=1 rows**: 16x augmentation (shuffle the 200 feature values within each row 16 times)
- **For target=0 rows**: 4x augmentation

**Rationale**: Since all 200 features are treated symmetrically by the attention model, a permuted version of a row is a valid augmentation. The model sees more examples of the minority class.

```python
def shuffle_augment(X, y, n_aug_pos=16, n_aug_neg=4):
    X_aug, y_aug = [X], [y]
    for _ in range(n_aug_pos):
        X_pos = X[y == 1].copy()
        for i in range(len(X_pos)):
            np.random.shuffle(X_pos[i])  # shuffle feature order within row
        X_aug.append(X_pos)
        y_aug.append(np.ones(len(X_pos)))
    for _ in range(n_aug_neg):
        X_neg = X[y == 0].copy()
        for i in range(len(X_neg)):
            np.random.shuffle(X_neg[i])
        X_aug.append(X_neg)
        y_aug.append(np.zeros(len(X_neg)))
    return np.vstack(X_aug), np.concatenate(y_aug)
```

## Key Takeaways
1. Uniqueness features (5 categories) exploited the data generation mechanism — test-vs-train value counts encode signal
2. Attention-style NNs work well when many features are symmetric and equally likely to be relevant
3. Pseudo-labeling is powerful when combined with high-confidence thresholding
4. Shuffle augmentation is valid when the model treats features permutation-invariantly
5. Always inspect test set distributions — they often contain information about the generation process
