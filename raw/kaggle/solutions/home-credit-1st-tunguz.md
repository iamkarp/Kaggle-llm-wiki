# Home Credit Default Risk — 1st Place Solution
**Author**: Tunguz team | **Votes**: 465

---

## Competition
Predict probability of client default on a Home Credit loan. Binary classification, ROC-AUC metric. Multiple relational tables: main application table + bureau history, previous applications, credit card balance, POS cash, installment payments.

## Scale of Feature Engineering: 1800–2000 Features

The team engineered 1800–2000 features across all tables. This is exceptional in scale. Key principle: **join everything, aggregate everything**.

### Feature Sources
| Table | Feature Count | Key Aggregations |
|-------|--------------|-----------------|
| application_train | ~100 raw + ~200 engineered | ratios, products, differences |
| bureau | ~300 | history stats per client |
| bureau_balance | ~200 | monthly balance patterns |
| previous_application | ~400 | past loan behavior |
| credit_card_balance | ~200 | revolving credit patterns |
| pos_cash_balance | ~200 | point-of-sale payment history |
| installments_payments | ~200 | payment timing/amount deviation |

### Example Aggregations
For each table joined to the main application:
- mean, std, min, max, sum of every numeric column → grouped by `SK_ID_CURR`
- Count, count distinct, ratio of counts for categoricals
- Time-based aggregations: last 6 months, last 12 months, ever

### KNN Target Mean — Top Feature
**The single most important feature**: K-nearest neighbors target mean.

For each training sample, find its K nearest neighbors in feature space and compute the mean of their target values (default rate). This is a form of non-parametric target encoding that captures local default rates in similar applicant profiles.

```python
from sklearn.neighbors import KNeighborsClassifier

# Fit KNN on training data
knn = KNeighborsClassifier(n_neighbors=500, metric='euclidean')
knn.fit(X_train_normalized, y_train)

# For each sample, get neighbors and compute mean target
distances, indices = knn.kneighbors(X_train_normalized)
knn_target_mean = y_train[indices].mean(axis=1)

# This becomes a feature — do OOF to avoid leakage!
```

**Why it's the top feature**: It aggregates hundreds of features into a single "how risky is this type of applicant" signal. Essentially a non-parametric local risk estimate.

## Forward Feature Selection: 1600 → 240 Features

With 1800+ features, the team needed to reduce to avoid overfitting and slow training. They used **forward feature selection with Ridge regression**:

1. Start with empty feature set
2. Train Ridge regression with cross-validation on candidate features
3. Add the feature that improves CV AUC the most
4. Repeat until adding features stops helping (or budget is reached)
5. Final set: ~240 features

**Why Ridge for selection**: Fast, stable, doesn't overfit badly, gives reliable signal on feature importance. The final models (LightGBM, etc.) are then trained on this reduced set.

**Alternative**: Use LightGBM importance + permutation importance to eliminate bottom features, then use Ridge for forward selection on the survivors.

## 3-Level Stacking Architecture

```
Level 0 (Base Features)
    ↓
Level 1: 90+ base models
    LightGBM variants × 30
    XGBoost variants × 20
    CatBoost variants × 10
    LogisticRegression × 5
    NN (MLP) × 5
    DAE+NN × 5
    Ridge variants × 10
    Other × 5+
    ↓ (OOF predictions → L1 meta-features)
Level 2: L1 Stackers
    Ridge on OOF from Level 1
    LightGBM on OOF from Level 1
    DAE+NN on OOF from Level 1
    ↓ (OOF predictions → L2 meta-features)
Level 3 / Final Blend
    Weighted average of L2 stackers
```

**90+ base models**: Same algorithm family with different hyperparameters, random seeds, feature subsets, and data preprocessing variations. Generates maximum diversity.

**DAE+NN as stacker**: The denoising autoencoder (cf. Porto Seguro approach) is used at level 2 to learn non-linear combinations of L1 predictions — more powerful than Ridge but controlled enough to not overfit.

## Validation
- 5-fold stratified CV on main training set
- Always OOF for stacking — no exceptions
- Target leakage check on all joined features

## Key Takeaways
1. For relational/multi-table competitions: aggregate everything, join everything
2. KNN target mean is a powerful non-parametric feature (does OOF)
3. Forward feature selection with Ridge is reliable for 1000+ feature reduction
4. Deep stacking (3+ levels) with 90+ base models can squeeze out the last performance
5. DAE+NN is an effective stacker at higher levels (better than Ridge alone)
6. Diversity in base models (different algorithms, seeds, feature sets) is critical for stacking gains
