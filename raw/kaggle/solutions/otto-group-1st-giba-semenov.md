# Otto Group Product Classification — 1st Place Solution
**Authors**: Giba & Semenov | **Votes**: 499

---

## Competition
Classify products into 9 categories based on anonymized numerical features. Multi-class classification, log-loss metric. ~62K training samples, 93 features, 9 classes. This 2015 competition became the canonical reference for multi-level stacking.

## The Classic 3-Level Stacking Blueprint

This solution defined the stacking template that Kaggle practitioners still follow. Three levels, progressively fewer but more powerful models, culminating in a geometric mean blend.

### Level 1: 33 Diverse Base Models

The breadth of L1 diversity was unprecedented at the time. Every major model family represented:

| Model Type | Variants | Notes |
|-----------|----------|-------|
| Random Forest | Multiple `n_estimators`, `max_features` | scikit-learn |
| Logistic Regression | L1/L2 regularization, different C | One-vs-rest |
| Extra Trees | Multiple depths | Faster RF variant |
| KNN | K=1, 3, 5, 10, 25, 50, 100, 250, 500, 1000 | Multiple K values! |
| libFM | Factorization machines | Captures interactions |
| H2O Deep Learning | H2O AutoML NN | Java-based distributed NN |
| Lasagne NN bags | Multiple seeds, architectures | Theano-based |
| XGBoost | Multiple depth/lr combinations | |
| Sofia | Stochastic gradient SVM | Large-scale linear |
| T-SNE features → any classifier | 3D t-SNE + classifiers | (see below) |
| KMeans clusters → any classifier | 50 cluster features | (see below) |

**Key rule from this solution**: Don't discard low-performance L1 models. A model that underperforms on its own may still add diversity to L2 and improve the ensemble. The contribution to L2 is through orthogonality of errors, not raw performance.

### Level 2: Three High-Capacity Stackers

L2 receives the 33 OOF prediction columns (9 classes × 33 models = 297 features) as input:

```
XGBoost L2:   250 bags (different seeds, slight hyperparameter variation)
NN L2:        600 bags (Lasagne/Theano, multiple architectures and seeds)
AdaBoost L2:  250 bags with Extra Trees as base estimator
```

The massive bagging (250–600 bags per L2 model) is distinctive. With 297 input features, NNs have high variance — averaging hundreds of bags produces a stable estimate.

### Level 3: Geometric Mean Blend

The final combination is not a simple weighted average but a **geometric mean blend**:

```
Final = 0.85 × (XGB_L2^0.65 × NN_L2^0.35) + 0.15 × ET_L2
```

Breaking this down:
- Inner geometric mean: `XGB_L2^0.65 × NN_L2^0.35`
  - XGBoost gets 65% geometric weight, NN gets 35%
  - Geometric mean of probabilities is better calibrated than arithmetic mean for multi-class log-loss
- Outer linear blend: 85% of the geometric blend + 15% ET
  - ET provides a diversifying "anchor" from a different model family

**Why geometric mean for probabilities**:
- For multi-class log-loss, probabilities must sum to 1 per sample
- Arithmetic averaging preserves this; geometric averaging followed by renormalization tends to produce sharper predictions
- Geometric mean of two well-calibrated models is often better calibrated than arithmetic mean

```python
import numpy as np

def geometric_blend(preds_list, weights):
    """
    Geometric mean blend of probability matrices.
    preds_list: list of (n_samples, n_classes) arrays
    weights: list of exponent weights (should sum to 1)
    """
    log_blend = sum(w * np.log(p + 1e-10) for w, p in zip(weights, preds_list))
    blend = np.exp(log_blend)
    # Renormalize to sum to 1 per sample
    return blend / blend.sum(axis=1, keepdims=True)

# Inner geometric blend
inner = geometric_blend([xgb_l2, nn_l2], weights=[0.65, 0.35])
# Outer linear combination with ET
final = 0.85 * inner + 0.15 * et_l2
```

## t-SNE 3D Features

Rather than using t-SNE purely for visualization, Giba & Semenov used it to generate numeric features:

1. Run t-SNE with `n_components=3` on the training feature matrix (jointly with test)
2. The 3 t-SNE dimensions capture the global cluster structure of the data
3. Use these 3 dimensions as additional input features for L1 models

```python
from sklearn.manifold import TSNE

# Run on train + test jointly (t-SNE is transductive — must run on all data together)
X_all = np.vstack([X_train, X_test])
tsne = TSNE(n_components=3, perplexity=30, random_state=42, n_jobs=4)
X_tsne = tsne.fit_transform(X_all)

X_train_tsne = X_tsne[:len(X_train)]
X_test_tsne = X_tsne[len(X_train):]

# Add as features
X_train_aug = np.hstack([X_train, X_train_tsne])
X_test_aug = np.hstack([X_test, X_test_tsne])
```

**Warning**: t-SNE is not easily applicable to new test data in a production setting (it's transductive). For competitions where the full test set is available upfront, this is fine. For online test scenarios, use UMAP (has `transform()` method) instead.

## KMeans Cluster Features

Fit KMeans (50 clusters) on training data; use cluster assignments and distances as features:

```python
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=50, random_state=42, n_init=10)
kmeans.fit(X_train)

# Cluster assignment (categorical) → one-hot or label encode
train_cluster = kmeans.predict(X_train)
test_cluster = kmeans.predict(X_test)

# Distance to each cluster center (50 features)
train_dists = kmeans.transform(X_train)   # (n_train, 50)
test_dists = kmeans.transform(X_test)     # (n_test, 50)
```

The cluster distance features capture "how far is this sample from each cluster center" — a non-parametric density estimate that's useful when cluster membership is soft.

## The Diversity Principle (Codified Here First)

> "Don't discard low-performance L1 models."

Giba & Semenov explicitly documented this principle. A model with L1 solo log-loss of 0.60 vs. the best model's 0.55 is still worth including in L2 because:
- It makes different errors than the top model
- Those error differences are exactly what the L2 meta-learner exploits
- The L2 contribution is correlation-based, not performance-based

This principle was later independently rediscovered and quantified in many solutions (cf. Home Credit: 90+ models, including simple Ridge variants).

## Key Takeaways
1. 33 diverse L1 models is the template — breadth across model families matters more than individual quality
2. KNN at multiple K values (K=1,3,5,10,...1000) is a cheap way to add smooth-interpolation diversity
3. Keep ALL L1 models regardless of individual performance — diversity, not quality, is the selection criterion
4. Geometric mean blend (with renormalization) outperforms arithmetic mean for multi-class log-loss
5. t-SNE 3D features add global cluster structure as numeric inputs to L1 models
6. 250–600 L2 bags per stacker stabilizes NN/XGB predictions on the 297-feature L2 input
7. This blueprint (diverse L1 → bagged L2 → geometric L3) remains canonical 10 years later
