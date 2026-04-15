# Advanced Target Encoding Techniques for Kaggle

Compiled from Pargent 2022 benchmark, competition notebooks, category_encoders docs. April 2026.

---

## Benchmark: Which Target Encoding Wins?

**Pargent et al. 2022** ([arXiv:2104.01621](https://arxiv.org/abs/2104.01621)) — Benchmarked 9 target encoding methods on 20 datasets.

**Rankings:**
1. **GLMM Encoding** — Best overall (wins on 12/20 datasets)
2. **CatBoost Ordered TS** — Runner-up (best for low-cardinality with noise)
3. **James-Stein Encoding** — Strong shrinkage; outperforms plain TE on small groups
4. **Leave-One-Out + Noise** — Good baseline; simple to implement
5. **Quantile Encoder** — Best for skewed targets
6. **Plain Target Encoding** — Worst (without regularization) — leaks training labels

---

## 1. GLMM Encoding (Generalized Linear Mixed Model)

**Winner of Pargent 2022 benchmark.** Treats category as a random effect, shrinks estimates toward grand mean.

```python
# pip install category-encoders
from category_encoders.glmm import GLMMEncoder

encoder = GLMMEncoder(
    cols=['city', 'product_type', 'device'],
    binomial_target=True  # True for binary classification
)

# Uses nested CV internally to prevent leakage
encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
```

**What it does:** Fits a GLMM where the category is a random effect with variance component σ². The estimate for each category is a weighted blend of the category mean and grand mean, where the weight is determined by the variance component.

**Why it wins:** GLMM naturally handles small groups (extreme shrinkage) and large groups (minimal shrinkage) in a statistically principled way.

---

## 2. CatBoost Ordered Target Statistics — Gold Standard for Leakage Prevention

CatBoost's built-in target encoding uses a random permutation of the training data to prevent leakage:

**The algorithm:**
1. Randomly permute training rows (different random seed per tree)
2. For row i: compute category mean using only rows BEFORE i in the permutation
3. This prevents row i from "seeing" its own label during encoding

```python
from catboost import CatBoostClassifier, Pool

# CatBoost handles it natively
cat_features = ['city', 'product_type', 'device']

model = CatBoostClassifier(
    cat_features=cat_features,  # Specify which are categorical
    iterations=1000,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)

model.fit(X_train, y_train)  # CatBoost encodes internally with ordered TS
```

**Manual implementation (for understanding):**
```python
def catboost_ordered_ts(X, y, col, a=1):
    """
    a: smoothing parameter (higher = more shrinkage)
    Processes rows in random order; each row uses only preceding rows.
    """
    n = len(X)
    perm = np.random.permutation(n)
    
    # Initialize with prior (grand mean)
    cumsum = np.zeros(X[col].nunique())
    cumcount = np.zeros(X[col].nunique())
    
    result = np.zeros(n)
    cat_to_idx = {cat: i for i, cat in enumerate(X[col].unique())}
    
    for i, idx in enumerate(perm):
        cat = X[col].iloc[idx]
        cat_id = cat_to_idx[cat]
        
        # Encode using past samples only
        prior = y.mean()
        numer = cumsum[cat_id] + a * prior
        denom = cumcount[cat_id] + a
        result[idx] = numer / denom
        
        # Update running stats
        cumsum[cat_id] += y.iloc[idx]
        cumcount[cat_id] += 1
    
    return result
```

---

## 3. James-Stein Encoding

**Principle:** Shrinks each category estimate toward the global mean using variance-based shrinkage factor B.

```python
from category_encoders.james_stein import JamesSteinEncoder

encoder = JamesSteinEncoder(
    cols=['city', 'product_type'],
    model='independent',  # 'independent' or 'pooled'
    handle_unknown='value',
    handle_missing='value'
)
encoder.fit(X_train, y_train)
```

**The shrinkage formula:**
```
B = σ²_within / (σ²_between + σ²_within)

For each category c:
JS_encoding(c) = (1 - B) * mean(c) + B * grand_mean
```

Where:
- `B` → 1 (shrinks to grand mean) when within-category variance is high vs between-category variance
- `B` → 0 (uses raw mean) when the category signal is strong

**When it's best:** High-cardinality categoricals where many groups have few examples.

---

## 4. Leave-One-Out Encoding with Noise

```python
from category_encoders.leave_one_out import LeaveOneOutEncoder

encoder = LeaveOneOutEncoder(
    cols=['city', 'product_type'],
    sigma=0.05  # Gaussian noise standard deviation
)
encoder.fit(X_train, y_train)
X_train_enc = encoder.transform(X_train, y=y_train)  # LOO for train
X_test_enc  = encoder.transform(X_test)               # Mean encoding for test
```

**How LOO prevents leakage:**
- Training: exclude the current row's label when computing the category mean
- Test: use full training set mean (no exclusion needed)

**Noise injection:** Adding `sigma * N(0,1)` prevents overfitting when the same category appears many times.

---

## 5. Quantile Encoding (for Skewed Targets)

For regression with skewed targets (house prices, sales revenue):

```python
from category_encoders.quantile_encoder import QuantileEncoder, SummaryEncoder

# Single quantile
q25_encoder = QuantileEncoder(cols=['city'], quantile=0.25, m=1.0)
q75_encoder = QuantileEncoder(cols=['city'], quantile=0.75, m=1.0)

# Summary encoder: mean + std + quantiles (most information)
summary_encoder = SummaryEncoder(
    cols=['city'],
    quantiles=[0.25, 0.5, 0.75],
    m=1.0,
    handle_unknown='value'
)
summary_encoder.fit(X_train, y_train)

# Produces: city_mean, city_std, city_q25, city_q50, city_q75
X_encoded = summary_encoder.transform(X_train)
```

**When to use:** Log-normal targets (prices, counts) where median is more representative than mean. SummaryEncoder captures full target distribution per category.

---

## 6. Weight of Evidence (WoE) Encoding

**Best for:** Binary classification with logistic regression.

```python
from category_encoders.woe import WOEEncoder

encoder = WOEEncoder(
    cols=['city', 'product_type'],
    regularization=1.0  # Laplace smoothing
)
encoder.fit(X_train, y_train)
```

**Formula:**
```
WoE(c) = ln(P(X=c | Y=1) / P(X=c | Y=0))
```

**Properties:**
- WoE values are additive in log-odds space → natural fit for logistic regression
- Extreme positive WoE = strong positive predictor; extreme negative = strong negative
- Automatically handles rare categories (regularization clips extreme WoE values)

**Information Value (IV) for feature selection:**
```python
def compute_iv(X, y, col, bins=10):
    """Information Value: overall predictive power of the feature."""
    data = pd.DataFrame({'x': X[col], 'y': y})
    
    # Discretize numeric columns
    if X[col].dtype in [float, int]:
        data['x'] = pd.qcut(data['x'], q=bins, duplicates='drop')
    
    # Compute WoE and IV for each bin
    iv = 0
    for bin_val in data['x'].unique():
        mask = data['x'] == bin_val
        p1 = data.loc[mask, 'y'].mean()  # P(positive | bin)
        p0 = 1 - p1                       # P(negative | bin)
        if p1 == 0 or p0 == 0: continue   # Skip pure bins
        
        dist_y1 = data.loc[mask, 'y'].sum() / data['y'].sum()
        dist_y0 = (mask.sum() - data.loc[mask, 'y'].sum()) / (len(data) - data['y'].sum())
        
        iv += (dist_y1 - dist_y0) * np.log(dist_y1 / dist_y0)
    
    return iv
# IV < 0.02: not useful; 0.02-0.1: weak; 0.1-0.3: medium; >0.3: strong
```

---

## 7. Entity Embeddings (Neural Network Approach)

**Paper:** Guo & Berkhahn (2016) — "Entity Embeddings of Categorical Variables"
**Competition use:** 3rd place in Rossmann Store Sales (2015), widely used since.

```python
import torch
import torch.nn as nn

class TabularModelWithEmbeddings(nn.Module):
    def __init__(self, cat_dims, num_features, embed_dim=8):
        """
        cat_dims: dict of {col_name: n_unique_values}
        """
        super().__init__()
        
        # Embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n_cats + 1, min(embed_dim, (n_cats + 1) // 2 + 1))
            for col, n_cats in cat_dims.items()
        })
        
        # Embedding output size
        embed_total = sum(
            min(embed_dim, (n + 1) // 2 + 1) 
            for n in cat_dims.values()
        )
        
        self.network = nn.Sequential(
            nn.Linear(embed_total + num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x_cat, x_num):
        # Embed each categorical feature
        embedded = [self.embeddings[col](x_cat[:, i]) 
                   for i, col in enumerate(self.embeddings.keys())]
        cat_features = torch.cat(embedded, dim=1)
        
        # Combine with numerical features
        combined = torch.cat([cat_features, x_num], dim=1)
        return self.network(combined)
```

**Recommended embedding dimension:** `min(50, (n_categories + 1) // 2)` — the fastai rule of thumb.

**Key advantage:** Embeddings capture semantic similarity between categories (e.g., Monday and Tuesday will have similar embeddings if they have similar target values).

---

## 8. Temporal Target Encoding (Expanding Window)

**For time-series tabular data:** Standard TE uses all data → leaks future. Use expanding window.

```python
def temporal_target_encode(df, date_col, cat_col, target_col, min_periods=30):
    """
    Encode based on past data only (expanding window).
    """
    df = df.sort_values(date_col)
    encoded = []
    
    # Expanding window: for each row, use all past rows
    cumsum = {}
    cumcount = {}
    
    for _, row in df.iterrows():
        cat = row[cat_col]
        
        # Encode using PAST data only
        if cat in cumcount and cumcount[cat] >= min_periods:
            encoding = cumsum[cat] / cumcount[cat]
        else:
            encoding = df[target_col].mean()  # prior when insufficient history
        
        encoded.append(encoding)
        
        # Update running stats
        cumsum[cat] = cumsum.get(cat, 0) + row[target_col]
        cumcount[cat] = cumcount.get(cat, 0) + 1
    
    df[f'{cat_col}_te_temporal'] = encoded
    return df
```

---

## Complete category_encoders API Reference

```python
from category_encoders import (
    TargetEncoder,           # Plain mean TE (use only as baseline)
    LeaveOneOutEncoder,      # LOO with noise
    WOEEncoder,              # Weight of Evidence (logistic regression)
    JamesSteinEncoder,       # Variance-based shrinkage
    GLMMEncoder,             # GLMM (benchmark winner)
    QuantileEncoder,         # Percentile encoding (regression)
    SummaryEncoder,          # Mean+Std+Quantiles
    MEstimateEncoder,        # M-estimate smoothing (similar to James-Stein)
    CatBoostEncoder,         # CatBoost ordered TS (manual implementation)
    OrdinalEncoder,          # Simple ordinal encoding
    BinaryEncoder,           # Binary representation
    BaseNEncoder,            # Base-N encoding
    HashingEncoder,          # Feature hashing
    HelmertEncoder,          # Contrast coding
)
```

---

## Decision Guide

| Scenario | Best Encoder | Why |
|---|---|---|
| Binary classification, large dataset | GLMMEncoder | Benchmark winner |
| With CatBoost model | CatBoost native | Built-in ordered TS |
| High cardinality (>500 categories) | JamesSteinEncoder | Strong shrinkage |
| Logistic regression pipeline | WOEEncoder | Log-odds additivity |
| Skewed regression target | SummaryEncoder | Captures distribution |
| Time-series data | Temporal expanding window | Prevents future leakage |
| Small dataset, NN model | Entity Embeddings | Captures semantic similarity |
| Competition baseline | LeaveOneOutEncoder | Simple, defensible |

---

Sources:
- Pargent et al. 2022 benchmark: https://arxiv.org/abs/2104.01621
- category_encoders package: https://contrib.scikit-learn.org/category_encoders/
- CatBoost ordered TS paper: https://arxiv.org/abs/1706.09516
- Entity embeddings (Guo & Berkhahn 2016): https://arxiv.org/abs/1604.06737
- Rossmann 3rd place solution: https://github.com/entron/entity-embedding-rossmann
- James-Stein encoder: https://contrib.scikit-learn.org/category_encoders/jamesstein.html
