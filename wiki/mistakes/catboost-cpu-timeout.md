---
title: "Mistake: CatBoost CPU Timeout"
category: mistakes
tags: [catboost, runtime, kaggle, timeout]
created: 2026-04-15
updated: 2026-04-15
---

# Mistake: CatBoost on CPU Exceeds Kaggle 12h Limit

**What happened:** CatBoost multi-class classification with ~157 features, 4-fold CV. Each fold took ~3 hours on Kaggle CPU. 4 folds × 3h = 12h just for CatBoost, leaving no time for LightGBM, XGBoost, or any other processing.

**Three versions timed out (V7, V11, V12)** before we identified CatBoost as the bottleneck.

**The fix:** Drop CatBoost from CPU-only pipelines. Use LightGBM + XGBoost only — together they finish in ~2h for the same 4-fold CV.

**General rule:** On Kaggle CPU notebooks:
- LightGBM: fastest (~10 min/fold for moderate data)
- XGBoost: moderate (~15 min/fold)
- CatBoost: slowest (~3h/fold for multi-class) — **avoid on CPU**

CatBoost is fine on GPU or for binary classification, but multi-class on CPU with many features is prohibitively slow.

## Prevention

Add runtime budgeting to your notebook:
```python
import time
start = time.time()
# ... after each fold ...
elapsed_h = (time.time() - start) / 3600
if elapsed_h > 10.5:
    print(f"Approaching limit ({elapsed_h:.1f}h). Stopping.", flush=True)
    break
```

## See Also
- [[techniques/gradient-boosted-trees]]
- [[tools/kaggle-cpu-notebooks]]
