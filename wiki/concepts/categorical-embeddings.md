---
title: "Categorical Embeddings — Topic Models on Interaction Sequences"
tags: [categorical, embeddings, lda, nmf, pca, topic-models, interaction-sequences, fraud, high-cardinality]
date: 2026-04-15
source_count: 1
status: active
---

## What It Is
Standard handling of high-cardinality categoricals (target encoding, one-hot, label encoding) treats each category value as atomic. Categorical embedding via topic models treats **sequences of interactions** between entities as documents, then runs LDA/NMF/SVD to extract dense embeddings that capture behavioral patterns.

The core insight from TalkingData 1st place: an IP address making clicks on apps `[A, B, A, C, D, B]` is analogous to a document containing words `A B A C D B`. Topic models find the latent "click behavior types" (topics) that characterize different IPs, just as document topic models find thematic clusters in text corpora.

## When to Use This Approach

| Signal | Use Topic Embeddings |
|--------|---------------------|
| High-cardinality categorical (>10K unique) | Yes — label encoding is meaningless |
| Entity has a sequence of interactions with another entity | Yes — sequence structure is exploitable |
| Same entity appears in both train and test (transductive) | Yes — fit on all data together |
| Entity is unseen in test (inductive problem) | No — can't embed new entities |
| Low cardinality (<500 unique values) | No — one-hot or target encoding is simpler |

## The Three Decompositions

For each (entity, attribute) pair, apply all three and concatenate:

### LDA — Latent Dirichlet Allocation
Probabilistic generative model. Finds topics as distributions over attributes; each entity gets a soft membership vector over topics. Best when topics are interpretable (e.g., "bot type A clicks mostly on gaming apps").

```python
from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(
    n_components=100,     # number of topics
    max_iter=20,
    learning_method='online',   # for large matrices
    random_state=42,
    n_jobs=-1,
)
lda_embeddings = lda.fit_transform(X_counts)  # (n_entities, 100)
```

### NMF — Non-Negative Matrix Factorization
Deterministic. Factors the count matrix into non-negative components. Often produces sparser, more interpretable topics than LDA. Better when the count matrix is sparse.

```python
from sklearn.decomposition import NMF

nmf = NMF(
    n_components=100,
    init='nndsvda',      # better initialization for sparse matrices
    random_state=42,
    max_iter=200,
)
nmf_embeddings = nmf.fit_transform(X_counts)  # (n_entities, 100)
```

### SVD/PCA — Truncated Singular Value Decomposition
Linear decomposition (equivalent to PCA on sparse matrices). No non-negativity constraint; captures variance most efficiently. Fastest of the three.

```python
from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=100, random_state=42)
svd_embeddings = svd.fit_transform(X_counts)  # (n_entities, 100)
```

## Full Implementation

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD

def topic_embed_pair(df_all, entity_col, attribute_col,
                     n_topics=100, max_vocab=10000):
    """
    Embed each entity based on its sequence of attribute interactions.
    df_all: combined train+test (fit on all available data for transductive setting)
    Returns: DataFrame with entity_col + embedding columns
    """
    # Build "document" for each entity: sequence of attribute values as space-separated string
    entity_docs = (df_all.groupby(entity_col)[attribute_col]
                         .apply(lambda x: ' '.join(x.astype(str)))
                         .reset_index()
                         .rename(columns={attribute_col: 'doc'}))
    
    # Build count matrix (entity × attribute co-occurrence)
    vectorizer = CountVectorizer(max_features=max_vocab, min_df=2)
    X_counts = vectorizer.fit_transform(entity_docs['doc'])
    # X_counts: (n_entities, n_vocab) sparse matrix
    
    # Three decompositions
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    nmf = NMF(n_components=n_topics, random_state=42)
    svd = TruncatedSVD(n_components=n_topics, random_state=42)
    
    lda_emb = lda.fit_transform(X_counts)
    nmf_emb = nmf.fit_transform(X_counts)
    svd_emb = svd.fit_transform(X_counts)
    
    # Concatenate: (n_entities, 300)
    all_emb = np.hstack([lda_emb, nmf_emb, svd_emb])
    
    prefix = f'{entity_col}_{attribute_col}'
    embed_cols = [f'{prefix}_lda_{i}' for i in range(n_topics)] + \
                 [f'{prefix}_nmf_{i}' for i in range(n_topics)] + \
                 [f'{prefix}_svd_{i}' for i in range(n_topics)]
    
    embed_df = pd.DataFrame(all_emb, columns=embed_cols)
    embed_df[entity_col] = entity_docs[entity_col].values
    return embed_df

# Apply to multiple pairs
pairs = [('ip', 'app'), ('ip', 'channel'), ('app', 'channel'), ('ip', 'device')]
for entity, attr in pairs:
    embeds = topic_embed_pair(df_all, entity, attr)
    df = df.merge(embeds, on=entity, how='left')
```

## The Critical Step: Drop Raw Categoricals After Embedding

Once embeddings are added, **remove the original label-encoded categorical columns**. TalkingData: this jumped LB from 0.9821 → 0.9828 — as large as adding the embeddings themselves.

```python
# After adding all embedding features:
raw_cat_cols = ['ip', 'app', 'device', 'os', 'channel']  # high-cardinality raw IDs
df.drop(columns=raw_cat_cols, inplace=True)
```

**Why dropping helps**:
- Raw label-encoded IDs are arbitrary integers — the model wastes splits trying to learn from them
- High-cardinality IDs cause trees to memorize specific values seen in training (overfitting)
- Embeddings capture the same information (and more) in a generalizable form
- Unseen IDs in test get zero embeddings vs. meaningless integer splits on raw IDs

**When NOT to drop**: If the entity has low cardinality (<200 unique values), the raw ID may still be informative via frequency encoding or target encoding. Only drop high-cardinality IDs where tree splits are unreliable.

## Frequency and Count Features (Complement to Embeddings)

Topic embeddings capture behavioral patterns. Frequency features capture volume:

```python
# How often does this entity appear? Rarity is often a fraud signal.
for col in ['ip', 'app', 'device', 'channel']:
    freq = df[col].value_counts()
    df[f'{col}_freq'] = df[col].map(freq)
    df[f'{col}_is_rare'] = (df[f'{col}_freq'] < 10).astype(int)

# Cross-entity frequency: how often does (ip, app) pair appear?
df['ip_app_count'] = df.groupby(['ip', 'app'])['click_time'].transform('count')
```

These frequency features are fast to compute and highly predictive for fraud (bots make many identical requests).

## Choosing n_topics

- Start with 50–100 per decomposition (150–300 total embedding features)
- More topics → richer representation but more risk of overfitting
- Validate using cross-validated CV score, not raw embedding quality
- For fraud/click data: 100 topics per method works well empirically

## Sources
- [[../../raw/kaggle/solutions/talkingdata-fraud-1st-komaki.md]] — LDA+NMF+SVD on ip→app sequences; dropping raw cats 0.9821→0.9828

## Related
- [[../concepts/target-encoding]] — alternative for moderate-cardinality categoricals
- [[../concepts/text-feature-engineering]] — TF-IDF + SVD applies same pattern to text vocabulary
- [[../concepts/feature-engineering-tabular]] — where embeddings fit in the 5-stage process
- [[../concepts/negative-downsampling]] — TalkingData uses both techniques together
