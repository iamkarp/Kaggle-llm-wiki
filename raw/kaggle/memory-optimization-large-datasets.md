# Memory Optimization and Large Dataset Handling in Kaggle

Compiled from NVIDIA blogs, Kaggle high-vote notebooks, Polars benchmarks, RAPIDS docs. April 2026.

---

## 1. dtype Reduction (reduce_mem_usage) — 65–80% Memory Reduction

Originated by Arjan Groen (Zillow competition). Most widely forked memory function in Kaggle history.

```python
import numpy as np
import pandas as pd

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
        print(f'Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df
```

**Typical result:** 65–80% reduction. Example: 286 MB → 60 MB (79%).

**Caveats:**
- Avoid float16 for model training features — LightGBM/XGBoost upcasts anyway; float16 can introduce precision artifacts.
- Use float32 as minimum for model inputs; float16 only for storing raw data.
- Run AFTER joining/merging to avoid silent type overflow bugs.

---

## 2. Chunked CSV Processing

```python
import pandas as pd
import gc

chunk_size = 500_000
chunks = []
dtypes = {'col_a': 'float32', 'col_b': 'int32', 'col_c': 'category'}

for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size, dtype=dtypes):
    chunk = chunk[chunk['col_a'] > 0]  # filter early
    chunk = reduce_mem_usage(chunk, verbose=False)
    chunks.append(chunk)
    gc.collect()

df = pd.concat(chunks, ignore_index=True)
del chunks
gc.collect()
```

**Key:** Pass `dtype=` explicitly — auto-detection is slow and wastes memory. Apply filters inside the loop.

---

## 3. Polars — 10–15x Faster Than Pandas

Benchmark (Polars PDS-H, May 2025): Polars 10–15x faster than pandas on most operations. DuckDB 3–5x faster for SQL-style analytics. Pandas OOM-fails at SF-100 scale; Polars handles via streaming engine.

```python
import polars as pl

# Lazy + streaming (larger than RAM)
df = (
    pl.scan_csv("large.csv")           # no data loaded
    .filter(pl.col("value") > 0)       # predicate pushdown
    .select(["id", "value", "date"])   # projection pushdown
    .with_columns([pl.col("value").cast(pl.Float32)])
    .collect(streaming=True)           # process in chunks
)

# Glob multiple files into one LazyFrame
df = pl.scan_csv("data/part_*.csv").collect(streaming=True)
```

**When to switch:** Any groupby/join on >1M rows, pandas notebook running >5 minutes, OOM on EDA.

**When to stay on pandas:** sklearn Pipeline, legacy code, libraries not yet supporting Polars.

**Streaming caveat:** Sorts and complex joins require full materialization. Polars silently falls back to standard engine.

---

## 4. Parquet Files — 2–5x Smaller, 5–20x Faster Read

```python
# Save
df.to_parquet('features.parquet', engine='pyarrow',
              compression='zstd', compression_level=3)

# Read only columns needed (columnar advantage)
df = pd.read_parquet('features.parquet', columns=['id', 'feature_1', 'target'])

# Polars reads parquet extremely fast
df = pl.scan_parquet('features.parquet').select(['id', 'feature_1']).collect()
```

**Compression guide:**
- `snappy` (default): best speed/ratio balance
- `zstd` level 3: beats snappy compression ratio, matches decompression speed — increasingly standard
- `gzip`: only when storage cost >> compute cost

---

## 5. Feature Store Pattern — Cache Expensive Features

```python
import os
import pandas as pd

def cached_feature(name, compute_fn, *args, cache_dir='./feature_cache', **kwargs):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f'{cache_dir}/{name}.parquet'
    if os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    result = compute_fn(*args, **kwargs)
    result.to_parquet(cache_path, compression='zstd', index=False)
    return result

# Usage
lag_features = cached_feature('lag_7d', compute_lag_features, df, window=7)
```

**GM pattern:** Stage 1: EDA + feature engineering → save parquet. Stage 2: Model training → load only needed columns. Avoids recomputing 2-hour feature pipelines on every HPO run.

---

## 6. RAPIDS cuDF — Up to 150x Speedup

Benchmarks:
- pandas operations on 5 GB dataset: **150x faster** with cuDF (NVIDIA GTC 2024)
- Google Colab T4 GPU: **50x faster** than CPU pandas
- Unified Memory (spills to host RAM): still **30x faster**

```python
# Zero-code-change acceleration (recommended)
%load_ext cudf.pandas   # Jupyter magic — all subsequent pandas calls run on GPU
import pandas as pd     # This is now GPU-backed transparently

# Or explicit cuDF
import cudf
df = cudf.read_csv('large.csv')
result = df.groupby('user_id').agg({'amount': ['mean', 'std', 'count']})
```

**Chris Deotte 2025:** Used cuDF to generate and test 10,000+ engineered features — the scale advantage is the key insight.

**Enable GPU on Kaggle notebooks:** Accelerator → GPU → enables cuDF; run `%load_ext cudf.pandas` to get zero-code-change acceleration.

---

## 7. Sparse Matrices for High-Cardinality One-Hot Features

```python
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

enc = OneHotEncoder(sparse_output=True, handle_unknown='ignore', max_categories=50)
X_cat = enc.fit_transform(df[['city', 'product_type', 'device']])

X_num = csr_matrix(df[numeric_cols].values.astype('float32'))
X_combined = hstack([X_num, X_cat])  # stays sparse

# LightGBM and XGBoost accept sparse matrices directly
import lightgbm as lgb
dtrain = lgb.Dataset(X_combined, label=y_train)
```

**Memory math:** 1,000 categories × 1M rows = 1 billion entries dense (1 GB at float32) vs 1M non-zero entries sparse = 4 MB. ~99.6% reduction.

**When to use:** OHE features with >20 categories, TF-IDF text features, user-item interaction matrices.
**When NOT to use:** GBDT models — they prefer label/frequency encoding over OHE.

---

## 8. NumPy memmap

For large pre-processed feature matrices that don't fit in RAM:

```python
import numpy as np

fp = np.memmap('features.npy', dtype='float32', mode='w+', shape=(n_samples, n_features))
# Fill in chunks; fp.flush() after each

# Load read-only (zero RAM cost, OS page cache handles it)
fp = np.memmap('features.npy', dtype='float32', mode='r', shape=(n_samples, n_features))

# PyTorch Dataset
class MemmapDataset(torch.utils.data.Dataset):
    def __init__(self, path, shape):
        self.data = np.memmap(path, dtype='float32', mode='r', shape=shape)
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx].copy())
```

**Benchmark:** memmap accelerates iteration over large ML datasets up to 20x vs CSV-per-epoch.

---

## 9. XGBoost External Memory (Out-of-Core Training)

```python
import xgboost as xgb

class CustomIter(xgb.DataIter):
    def __init__(self, files):
        self.files = files; self._it = 0
        super().__init__(cache_prefix='/tmp/xgb_cache')
    def next(self, input_data):
        if self._it == len(self.files): return 0
        df = pd.read_parquet(self.files[self._it])
        input_data(data=df.drop('label', axis=1), label=df['label'])
        self._it += 1; return 1
    def reset(self): self._it = 0

it = CustomIter(['part_0.parquet', 'part_1.parquet'])
dtrain = xgb.DMatrix(it)
model = xgb.train({'tree_method': 'hist'}, dtrain, num_boost_round=100)
```

**Batch size:** ~10 GB per batch if you have 64 GB RAM. Never use tiny batches — GBDT needs large batches for accurate split finding.

---

## Quick Decision Matrix

| Dataset Size | RAM | Recommended Stack |
|---|---|---|
| <2 GB | Any | Pandas + dtype reduction |
| 2–8 GB | 16 GB | Pandas + reduce_mem_usage + Parquet |
| 8–30 GB | 16 GB | Polars lazy + streaming, or Dask |
| >30 GB | 16 GB | Polars streaming or XGBoost external memory |
| Any | GPU available | cuDF-pandas (zero code change) |
| NLP/sparse | Any | scipy.sparse + linear models |
| Large embedding matrices | Any | numpy memmap + PyTorch DataLoader |

---

Sources:
- Arjan Groen reduce_mem_usage: https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65
- Polars PDS-H benchmark May 2025: https://pola.rs/posts/benchmarks/
- RAPIDS cuDF 150x: https://developer.nvidia.com/blog/rapids-cudf-accelerates-pandas-nearly-150x-with-zero-code-changes/
- cuDF Unified Memory 30x: https://developer.nvidia.com/blog/rapids-cudf-unified-memory-accelerates-pandas-up-to-30x-on-large-datasets/
- Chris Deotte cuDF 1st place: https://www.kaggle.com/competitions/playground-series-s5e6/writeups/chris-deotte-1st-place-fast-gpu-experimentation-wi
- XGBoost external memory docs: https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html
- Parquet performance: https://dev.to/alexmercedcoder/all-about-parquet-part-10-performance-tuning-and-best-practices-with-parquet-1ib1
