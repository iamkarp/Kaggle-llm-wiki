# TalkingData Ad Fraud Detection — 1st Place Solution
**Authors**: Komaki / flowlight | **Votes**: 269

---

## Competition
Predict whether a mobile ad click will be followed by an app download (i.e., detect fraudulent clicks from bots). Binary classification, AUC metric. ~184M training rows, ~19M test rows. Highly imbalanced: ~99.83% negative (no download), ~0.17% positive. Key features: ip, app, device, os, channel, click_time.

## Negative Downsampling: Discard 99.8% of Negatives

Training on 184M rows is computationally infeasible. The solution uses **negative downsampling**: keep all positives, randomly discard most negatives.

```python
pos_df = train[train['is_attributed'] == 1]          # ~456K rows
neg_df = train[train['is_attributed'] == 0]           # ~184M rows

# Keep only 0.2% of negatives (1-in-500 sampling)
neg_sample = neg_df.sample(frac=0.002, random_state=42)  # ~368K rows

train_sampled = pd.concat([pos_df, neg_sample])
# Result: ~824K rows, ~55% positive (artificially balanced)
```

**Critical**: After downsampling, the model's predicted probabilities are calibrated for the sampled distribution, not the true distribution. Must recalibrate before final submission:

```python
# Prior correction (Bayes' theorem)
# True positive rate: p = 0.0017
# Sampling rate of negatives: s = 0.002
# Correction: p_true = p_model / (p_model + (1-p_model) * s / (1-s_pos))

def calibrate_downsampled(p_model, neg_sampling_rate=0.002, true_pos_rate=0.0017):
    """Correct predicted probability after negative downsampling."""
    # From the Facebook downsampling paper
    odds_model = p_model / (1 - p_model)
    odds_correction = neg_sampling_rate  # sampling rate of negatives
    odds_corrected = odds_model * odds_correction
    return odds_corrected / (1 + odds_corrected)
```

**5-bag averaging**: Train 5 models on 5 different random negative samples. Average predictions. This:
- Reduces variance from the particular sample chosen
- Each bag sees different negatives → different decision boundaries
- Aggregated prediction is more stable than any single model

```python
bag_preds = []
for seed in range(5):
    neg_sample = neg_df.sample(frac=0.002, random_state=seed)
    train_bag = pd.concat([pos_df, neg_sample])
    model = train_lgbm(train_bag, params)
    bag_preds.append(model.predict(X_test))

final_preds = np.mean(bag_preds, axis=0)
# Apply prior correction to final_preds
```

## LDA/NMF/PCA Categorical Embedding: Topic Models on Interaction Sequences

**The breakthrough**: treat categorical interaction sequences as documents and run topic models to generate dense embedding features.

### Intuition
Each IP address makes a sequence of app clicks: `ip_123 → [app_A, app_B, app_A, app_C, ...]`. This is analogous to a document where words are app IDs. LDA/NMF topic models find latent "topics" (click patterns) that characterize different types of IPs (real users, bot type A, bot type B, etc.).

### Implementation
```python
from sklearn.decomposition import LatentDirichletAllocation, NMF, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer

# For each (entity, attribute) pair: ip→app, ip→channel, app→channel, etc.
def topic_embed(df, entity_col, attribute_col, n_components=100):
    """
    Treat each entity's sequence of attributes as a document.
    Run topic model; get entity embeddings.
    """
    # Build co-occurrence matrix: entity × attribute
    # CountVectorizer treats attribute sequences as "words"
    entity_docs = (df.groupby(entity_col)[attribute_col]
                     .apply(lambda x: ' '.join(x.astype(str)))
                     .reset_index())
    
    vectorizer = CountVectorizer(max_features=10000)
    X_counts = vectorizer.fit_transform(entity_docs[attribute_col])
    
    # LDA for probabilistic topic assignment
    lda = LatentDirichletAllocation(n_components=n_components, random_state=42,
                                     n_jobs=-1)
    lda_embeddings = lda.fit_transform(X_counts)
    
    # NMF for non-negative factorization
    nmf = NMF(n_components=n_components, random_state=42)
    nmf_embeddings = nmf.fit_transform(X_counts)
    
    # SVD/PCA for dense linear embedding
    svd = TruncatedSVD(n_components=100, random_state=42)
    svd_embeddings = svd.fit_transform(X_counts)
    
    # Concatenate: 300 embedding features per entity
    embeddings = np.hstack([lda_embeddings, nmf_embeddings, svd_embeddings])
    
    embed_df = pd.DataFrame(embeddings, 
                             columns=[f'{entity_col}_{attribute_col}_embed_{i}' 
                                      for i in range(embeddings.shape[1])])
    embed_df[entity_col] = entity_docs[entity_col].values
    return embed_df

# Apply to multiple (entity, attribute) pairs
for entity, attribute in [('ip', 'app'), ('ip', 'channel'), 
                           ('app', 'channel'), ('ip', 'device')]:
    embeds = topic_embed(train, entity, attribute, n_components=100)
    train = train.merge(embeds, on=entity, how='left')
    # 4 pairs × 300 features = 1200 embedding features total
    # Final solution used a subset: 646 features total
```

### The Critical Discovery: Drop Raw Categoricals After Embedding

Raw categorical features (ip, app, device, os, channel as label-encoded integers) were **removed** from the model after adding the topic model embeddings. This improved LB from **0.9821 → 0.9828**.

**Why removing raw categoricals helps**:
- Raw label-encoded IDs are meaningless to GBMs (arbitrary integer ordering)
- The tree model uses high-cardinality IDs as split points → overfits to specific IPs seen in training
- Embedding features capture the semantic structure (click patterns) without memorizing specific IDs
- Embeddings generalize to unseen IPs/apps in test; raw IDs don't

**The pattern**: Generate embeddings → verify CV improvement → drop raw source columns → verify further CV improvement.

## 646 Final Features

| Feature Group | Count | Source |
|--------------|-------|--------|
| Frequency/count features | ~100 | How often each (ip, app, channel, device) appears |
| Time delta features | ~100 | Time between clicks from same ip/device |
| Topic model embeddings (LDA+NMF+SVD) | ~400 | ip→app, ip→channel, app→channel sequences |
| Temporal aggregations | ~46 | Hourly/daily click counts per entity |
| **Total** | **646** | |

## Key Takeaways
1. Negative downsampling (0.2% of negatives) makes 184M-row training feasible
2. 5-bag averaging over different negative samples dramatically reduces variance
3. Always apply prior correction after downsampling — naive predictions are mis-calibrated
4. LDA+NMF+SVD on interaction sequences = 300 dense embedding features per (entity, attribute) pair
5. Drop raw label-encoded categoricals AFTER adding embeddings — embeddings generalize, IDs overfit
6. The LB jump from dropping raw IDs (0.9821→0.9828) was as large as adding the embeddings
