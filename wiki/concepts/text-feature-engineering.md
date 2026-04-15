---
title: "Text Feature Engineering — Embeddings vs TF-IDF vs Regex Extraction"
tags: [nlp, text, embeddings, tfidf, regex, feature-engineering, tabular, llm]
date: 2026-04-14
source_count: 1
status: active
---

## What It Is
When a tabular dataset contains free-text columns, you need to convert them to numeric features that tree models or linear models can consume. Three strategies, roughly in order of increasing precision but also increasing effort.

## Strategy A: Embeddings (Fastest Path)

Use a pretrained sentence encoder to get dense vector representations.

**Options** (2026):
- `sentence-transformers` (SBERT variants) — fast, good quality, local
- OpenAI `text-embedding-3-small` — strong, API-based
- Qwen/Mistral embedding variants — strong local alternative

**Critical step: dimensionality reduction before feeding to trees**.
Raw embeddings (768d, 1536d) are too high-dimensional for gradient boosting — each dimension looks like noise, and split capacity gets wasted. Reduce to 16–64 dimensions:

```python
from sklearn.decomposition import TruncatedSVD
import numpy as np

# embeddings: (n_samples, 768) numpy array
svd = TruncatedSVD(n_components=32, random_state=42)
X_reduced = svd.fit_transform(train_embeddings)         # fit on train
X_test_reduced = svd.transform(test_embeddings)         # apply to test

# Then treat the 32 columns as regular numeric features
```

**When to use**: Text is semantically rich, categories are well-represented in pretrained model vocabulary, no strong domain-specific terminology.

**When to skip**: Domain vocabulary is too specialized (clinical notes, legal jargon) — pretrained embeddings won't cluster meaningfully.

## Strategy B: TF-IDF (Fast + Interpretable)

Bag-of-words with term frequency weighting. Works well when vocabulary is shared between train and test.

### Check train/test term overlap first
```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=50000)
vectorizer.fit(train_text)
train_vocab = set(vectorizer.vocabulary_.keys())

# Get all unique terms in test
test_terms = set(t for doc in test_text for t in doc.lower().split())

overlap = len(train_vocab & test_terms) / len(train_vocab | test_terms)
print(f"Jaccard overlap: {overlap:.2%}")
```

**Rule**: If overlap < 40% → use embeddings instead. TF-IDF will produce mostly zero vectors for test, providing no signal.

### TF-IDF best practices
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

tfidf_svd = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),       # unigrams + bigrams
        sublinear_tf=True,        # log(1+tf) instead of raw tf
        min_df=2,                 # drop terms appearing in only 1 doc
        strip_accents='unicode',
        analyzer='word',
    )),
    ('svd', TruncatedSVD(n_components=64, random_state=42)),
])

X_train_text = tfidf_svd.fit_transform(train_text)
X_test_text = tfidf_svd.transform(test_text)
```

**When to use**: Train and test text come from same distribution, vocabulary overlap is high, interpretability of top terms is useful for debugging.

## Strategy C: LLM-Guided Regex Extraction (Highest Precision)

Use an LLM (Claude, GPT) to inspect a sample of the text and identify extractable structured patterns. Then implement those patterns as regex or rule-based extractors that run on all rows efficiently.

**Workflow**:
1. Feed 20–50 random text samples to Claude with: "What structured information can reliably be extracted from this text? List fields and their patterns."
2. Review Claude's suggestions — validate on a held-out sample.
3. Implement as regex/string operations in Python (fast, scalable).
4. The resulting binary/categorical/numeric features are highly interpretable for trees.

**Example** (from playbook):
```
Prompt to Claude: "Identify: (1) whether text mentions a dollar amount,
(2) overall sentiment (pos/neg/neutral), (3) presence of urgency words
(e.g., 'urgent', 'immediately', 'ASAP')"
```

Resulting features:
```python
df['has_dollar'] = df['text'].str.contains(r'\$[\d,]+', regex=True).astype(int)
df['has_urgency'] = df['text'].str.contains(r'\b(urgent|immediately|asap|right away)\b',
                                             case=False, regex=True).astype(int)
```

**When to use**: Text has consistent semi-structured patterns (medical notes, product descriptions, support tickets, legal filings). Pattern inspection reveals extractable signal. High competition where precision matters more than speed.

**When to skip**: Text is genuinely unstructured and variable — embeddings will generalize better.

## Combining Strategies
In practice: **use all three and ensemble the feature sets**. Embeddings capture semantic similarity; TF-IDF captures lexical overlap; regex captures structured signals. They're complementary.

Typical feature table for a text column:
| Feature Set | Count | Notes |
|-------------|-------|-------|
| Embedding SVD components | 32 | Dense semantic representation |
| TF-IDF SVD components | 64 | Lexical signals |
| Regex / rule-based | 5–20 | Hand-crafted domain signals |
| Meta-features | 3–5 | char_count, word_count, unique_word_ratio |

## In Jason's Work

### AUTOPILOT VQA
This is an extreme case of unstructured → structured: video captions generated by Qwen2.5-VL-32B are then parsed by Claude Sonnet via prompt-based extraction. The final output is 25 structured integer columns. This is essentially Strategy C at scale, mediated by two LLMs.

### General Kaggle Use
Embeddings (Strategy A) are the fastest baseline for any competition with text. Always try embeddings + PCA-32 first before investing in TF-IDF or regex.

## Sources
- [[../../raw/kaggle/kaggle-competition-playbook.md]] — §6 text/unstructured data strategies

## Related
- [[../concepts/feature-engineering-tabular]] — how text features plug into the broader 5-stage pipeline
- [[../concepts/ensembling-strategies]] — combining models trained on different feature sets
- [[../competitions/autopilot-vqa-2026]] — applied example: VLM-to-structured classification
