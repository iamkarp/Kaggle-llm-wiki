---
id: concept:target-encoding-advanced
type: concept
title: Advanced Target Encoding Techniques
slug: target-encoding-advanced
aliases: []
tags:
- target-encoding
- categorical
- glmm
- james-stein
- catboost
- entity-embeddings
- woe
- tabular
status: active
date: 2026-04-15
source_count: 6
---

## Summary

GLMM encoding wins the Pargent 2022 benchmark (12/20 datasets). CatBoost's ordered target statistics is the gold standard for leakage prevention. James-Stein shrinkage outperforms plain target encoding for high-cardinality features with small groups. Entity embeddings (Guo & Berkhahn 2016, Rossmann 3rd place) capture semantic category similarity and are the best option for neural network tabular models.

## What It Is

A collection of advanced categorical encoding techniques that go beyond plain mean target encoding. Critical for tabular competitions with high-cardinality categorical features.

## Key Facts / Details

### Benchmark Rankings (Pargent et al. 2022)

1. **GLMM Encoding** — Best overall (wins on 12/20 datasets)
2. **CatBoost Ordered TS** — Runner-up
3. **James-Stein Encoding** — Strong shrinkage; best for small groups
4. **Leave-One-Out + Noise** — Good baseline; simple to implement
5. **Quantile Encoder** — Best for skewed targets
6. **Plain Target Encoding** — Worst without regularization (leaks labels)

### 1. GLMM Encoding (Benchmark Winner)

```python
from category_encoders.glmm import GLMMEncoder

encoder = GLMMEncoder(
    cols=['city', 'product_type', 'device'],
    binomial_target=True  # True for binary classification
)
encoder.fit(X_train, y_train)
X_train_encoded = encoder.transform(X_train)
X_test_encoded = encoder.transform(X_test)
```

Treats category as a random effect; estimates are a weighted blend of category mean and grand mean.

### 2. CatBoost Ordered Target Statistics (Gold Standard for Leakage Prevention)

CatBoost's built-in encoding uses random permutation — each row uses only rows before it in the permutation:

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(
    cat_features=['city', 'product_type'],  # Specify categoricals
    iterations=1000,
    learning_rate=0.05,
    random_seed=42,
    verbose=0
)
model.fit(X_train, y_train)  # Encodes internally — no leakage
```

### 3. James-Stein Encoding

```python
from category_encoders.james_stein import JamesSteinEncoder

encoder = JamesSteinEncoder(cols=['city', 'product_type'])
encoder.fit(X_train, y_train)

# Formula: JS(c) = (1-B)*mean(c) + B*grand_mean
# where B = σ²_within / (σ²_between + σ²_within)
# → small groups (high within-variance) → shrink toward grand mean
```

Best for: high-cardinality categoricals where many groups have few examples.

### 4. Leave-One-Out with Noise

```python
from category_encoders.leave_one_out import LeaveOneOutEncoder

encoder = LeaveOneOutEncoder(cols=['city'], sigma=0.05)
encoder.fit(X_train, y_train)
X_train_enc = encoder.transform(X_train, y=y_train)  # LOO for train
X_test_enc  = encoder.transform(X_test)               # Mean for test
```

### 5. Quantile/Summary Encoding (Skewed Targets)

```python
from category_encoders.quantile_encoder import SummaryEncoder

# Produces: city_mean, city_std, city_q25, city_q50, city_q75
summary_encoder = SummaryEncoder(
    cols=['city'],
    quantiles=[0.25, 0.5, 0.75],
    m=1.0
)
summary_encoder.fit(X_train, y_train)
```

Best for: log-normal targets (prices, counts) where median is more representative than mean.

### 6. Weight of Evidence Encoding (Logistic Regression)

```python
from category_encoders.woe import WOEEncoder

encoder = WOEEncoder(cols=['city'], regularization=1.0)
encoder.fit(X_train, y_train)
# WoE(c) = ln(P(X=c|Y=1) / P(X=c|Y=0))
# Additive in log-odds space → natural for logistic regression
```

Information Value (IV) thresholds for feature selection:
- IV < 0.02: not useful
- 0.02-0.1: weak
- 0.1-0.3: medium
- >0.3: strong

### 7. Entity Embeddings (Neural Network Models)

Paper: Guo & Berkhahn (2016). Used by Rossmann Store Sales 3rd place (2015).

```python
class TabularModelWithEmbeddings(nn.Module):
    def __init__(self, cat_dims, num_features, embed_dim=8):
        super().__init__()
        # cat_dims: {col_name: n_unique_values}
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(n_cats + 1, min(embed_dim, (n_cats + 1) // 2 + 1))
            for col, n_cats in cat_dims.items()
        })
        embed_total = sum(min(embed_dim, (n+1)//2+1) for n in cat_dims.values())
        self.network = nn.Sequential(
            nn.Linear(embed_total + num_features, 512),
            nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1)
        )
    
    def forward(self, x_cat, x_num):
        embedded = [self.embeddings[col](x_cat[:, i]) for i, col in enumerate(self.embeddings)]
        combined = torch.cat([torch.cat(embedded, dim=1), x_num], dim=1)
        return self.network(combined)
```

**Embedding dimension rule:** `min(50, (n_categories + 1) // 2)` — the fastai rule.

### 8. Temporal Target Encoding (Time Series)

```python
def temporal_target_encode(df, date_col, cat_col, target_col, min_periods=30):
    """Use expanding window — encode based on past data only."""
    df = df.sort_values(date_col)
    cumsum, cumcount, encoded = {}, {}, []
    
    for _, row in df.iterrows():
        cat = row[cat_col]
        if cat in cumcount and cumcount[cat] >= min_periods:
            encoding = cumsum[cat] / cumcount[cat]
        else:
            encoding = df[target_col].mean()  # prior
        encoded.append(encoding)
        cumsum[cat] = cumsum.get(cat, 0) + row[target_col]
        cumcount[cat] = cumcount.get(cat, 0) + 1
    
    df[f'{cat_col}_te'] = encoded
    return df
```

### Decision Guide

| Scenario | Best Encoder |
|---|---|
| Binary classification, large dataset | GLMMEncoder |
| With CatBoost model | CatBoost native |
| High cardinality (>500 categories) | JamesSteinEncoder |
| Logistic regression pipeline | WOEEncoder |
| Skewed regression target | SummaryEncoder |
| Time-series data | Temporal expanding window |
| NN model | Entity Embeddings |
| Competition baseline | LeaveOneOutEncoder |

## Gotchas

- Plain target encoding without regularization is the single most common source of CV inflation
- Never fit encoder on both train and test together — always fit on train, transform test separately
- For temporal data, NEVER use future information — always expanding window or fold-based
- GLMM is slow for very high cardinality (>10K categories) — consider James-Stein instead
- Entity embeddings need enough training data to learn meaningful representations (>1K examples per category)

## In Jason's Work

Standard approach for tabular competitions: use `CatBoostEncoder` or `GLMMEncoder` from category_encoders. For neural network tabular models, use entity embeddings for high-cardinality categoricals.

## Sources

- `raw/kaggle/target-encoding-advanced.md` *(not yet ingested)* — full reference with all code
- [Pargent et al. 2022 benchmark](https://arxiv.org/abs/2104.01621)
- [category_encoders package](https://contrib.scikit-learn.org/category_encoders/)
- [CatBoost ordered TS paper](https://arxiv.org/abs/1706.09516)
- [Entity embeddings (Guo & Berkhahn 2016)](https://arxiv.org/abs/1604.06737)

## Related

- [[target-encoding]] — basic target encoding (weighted blend formula)
- [[feature-engineering-tabular]] — feature engineering context
- [[deep-learning-tabular]] — neural network tabular models
- [[gradient-boosting-advanced]] — CatBoost native encoding

<!-- kg:begin -->
<!-- This block is auto-generated by tools/inject_kg_blocks.py — do not hand-edit -->
## Knowledge Graph

**Outgoing:**
- _improves_on_ → [[concepts/deep-learning-tabular|Deep Learning on Tabular Data — When DNNs Beat GBMs]]
- _improves_on_ → [[concepts/feature-engineering-tabular|Feature Engineering — Tabular Data Patterns]]
- _improves_on_ → [[concepts/target-encoding|Target Encoding — Weighted Blend with OOF]]
- _improves_on_ → [[concepts/target-encoding|Target Encoding — Weighted Blend with OOF]]
- _works_with_ → [[concepts/gradient-boosting-advanced|Gradient Boosting — Advanced Configuration Tricks]]
- _works_with_ → [[entities/lightgbm-catboost|LightGBM & CatBoost — Gradient Boosting Alternatives]]

<!-- kg:end -->
