---
title: "Memory Optimization & Large Dataset Handling"
tags: [memory-optimization, polars, cudf, parquet, pandas, rapids, performance]
date: 2026-04-15
source_count: 7
status: active
---

## Summary

The `reduce_mem_usage` function achieves 65-80% memory reduction with zero effort. Polars is 10-15x faster than pandas for large datasets. cuDF gives 50-150x GPU speedup with zero code changes. Use parquet with zstd compression as the standard intermediate format for feature caching.

## What It Is

A toolkit of techniques for handling large Kaggle datasets that would otherwise cause OOM errors or slow notebooks. Enables Chris Deotte's approach of generating and testing 10,000+ features by making each iteration fast.

## Key Facts / Details

### Quick Decision Matrix

| Dataset Size | RAM | Recommended Stack |
|---|---|---|
| <2 GB | Any | Pandas + dtype reduction |
| 2–8 GB | 16 GB | Pandas + reduce_mem_usage + Parquet |
| 8–30 GB | 16 GB | Polars lazy + streaming, or Dask |
| >30 GB | 16 GB | Polars streaming or XGBoost external memory |
| Any | GPU available | cuDF-pandas (zero code change) |
| NLP/sparse | Any | scipy.sparse + linear models |
| Large embeddings | Any | numpy memmap + PyTorch DataLoader |

### 1. reduce_mem_usage — 65-80% Reduction

Originated by Arjan Groen (Zillow competition). Most widely forked memory function in Kaggle history.

```python
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory {start_mem:.1f} MB → {end_mem:.1f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df
```

**Caveats:**
- Avoid float16 for model training — LightGBM/XGBoost upcast anyway; float16 causes precision artifacts
- Use float32 as minimum for model inputs
- Run AFTER joining/merging to avoid type overflow bugs

### 2. Polars — 10-15x Faster Than Pandas

```python
import polars as pl

# Lazy + streaming (larger than RAM)
df = (
    pl.scan_csv("large.csv")            # no data loaded
    .filter(pl.col("value") > 0)        # predicate pushdown
    .select(["id", "value", "date"])    # projection pushdown
    .with_columns([pl.col("value").cast(pl.Float32)])
    .collect(streaming=True)            # process in chunks
)

# Multiple files
df = pl.scan_csv("data/part_*.csv").collect(streaming=True)
```

**When to switch:** GroupBy/join on >1M rows, pandas notebook >5 minutes, OOM on EDA.
**When to stay on pandas:** sklearn Pipeline, libraries not yet supporting Polars.

### 3. cuDF — 50-150x GPU Speedup

```python
# Zero-code-change (recommended)
%load_ext cudf.pandas   # Jupyter magic — all pandas calls run on GPU
import pandas as pd     # GPU-backed transparently

# Explicit cuDF
import cudf
df = cudf.read_csv('large.csv')
result = df.groupby('user_id').agg({'amount': ['mean', 'std', 'count']})
```

Benchmarks: 150x on 5GB dataset (NVIDIA GTC 2024), 50x on Colab T4 GPU, 30x with unified memory (spills to host RAM).

**Enable on Kaggle:** Accelerator → GPU → `%load_ext cudf.pandas`.

Chris Deotte (2025 1st place) used cuDF to generate and test 10,000+ engineered features.

### 4. Parquet — 2-5x Smaller, 5-20x Faster Read

```python
# Save with zstd compression (beats snappy, matches speed)
df.to_parquet('features.parquet', engine='pyarrow',
              compression='zstd', compression_level=3)

# Read only needed columns (columnar advantage)
df = pd.read_parquet('features.parquet', columns=['id', 'feature_1', 'target'])
```

Compression guide:
- `snappy`: default, best speed/ratio balance
- `zstd` level 3: beats snappy ratio, matches decompression — increasingly standard
- `gzip`: only when storage cost >> compute cost

### 5. Feature Store Pattern

```python
def cached_feature(name, compute_fn, *args, cache_dir='./feature_cache', **kwargs):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f'{cache_dir}/{name}.parquet'
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    result = compute_fn(*args, **kwargs)
    result.to_parquet(cache_path, compression='zstd', index=False)
    return result

lag_features = cached_feature('lag_7d', compute_lag_features, df, window=7)
```

GM pattern: Stage 1 — EDA + FE → save parquet. Stage 2 — Model training → load only needed columns. Avoids recomputing 2-hour feature pipelines on every HPO run.

### 6. Sparse Matrices

```python
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore', max_categories=50)
X_cat = enc.fit_transform(df[['city', 'product_type', 'device']])
X_num = csr_matrix(df[numeric_cols].values.astype('float32'))
X_combined = hstack([X_num, X_cat])  # stays sparse
```

Memory math: 1K categories × 1M rows = 1GB dense → 4MB sparse (99.6% reduction).

**When NOT to use:** GBDT models — they prefer label/frequency encoding over OHE.

### 7. Chunked CSV Processing

```python
chunk_size = 500_000
chunks = []
dtypes = {'col_a': 'float32', 'col_b': 'int32', 'col_c': 'category'}

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size, dtype=dtypes):
    chunk = chunk[chunk['col_a'] > 0]  # filter early
    chunk = reduce_mem_usage(chunk, verbose=False)
    chunks.append(chunk)
    gc.collect()
```

Key: pass `dtype=` explicitly — auto-detection is slow. Apply filters inside the loop.

## Gotchas

- Don't use float16 for model training features — LightGBM internally upcasts, but precision artifacts in edge cases
- Polars streaming mode silently falls back to standard engine for sorts and complex joins
- cuDF requires NVIDIA GPU — not available on CPU-only Kaggle notebooks
- XGBoost external memory needs batch sizes of ~10GB for accurate split finding — never use tiny batches

## Sources

- [[../raw/kaggle/memory-optimization-large-datasets.md]] — full reference with all code
- [reduce_mem_usage origin (Arjan Groen)](https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65)
- [Polars PDS-H benchmark](https://pola.rs/posts/benchmarks/)
- [RAPIDS cuDF 150x](https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/)

## Related

- [[concepts/gradient-boosting-advanced]] — cuDF integration for XGB/LGB/CAT
- [[concepts/knowledge-distillation]] — RAPIDS cuDF + nested 10-in-10 K-fold from Chris Deotte
- [[concepts/feature-engineering-tabular]] — the features this infrastructure enables
