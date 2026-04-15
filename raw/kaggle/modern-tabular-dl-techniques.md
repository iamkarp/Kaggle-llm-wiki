# Modern Deep Learning & Advanced Techniques for Tabular Kaggle (2023–2025)

Compiled from NeurIPS papers, ICLR 2025, Nature 2024, mlcontests.com, NVIDIA GM blogs, Kaggle discussions. April 2026.

---

## Why GBDTs Still Dominate (2024)

From "Why do tree-based models still outperform deep learning on tabular data?" (NeurIPS 2022):
1. **Irregular target functions**: Trees model discontinuous, non-smooth decision boundaries; NNs struggle.
2. **Uninformative features**: GBDTs are inherently feature-selective; NNs degrade under high noise feature ratios.
3. **Rotation non-invariance**: Tabular data is typically not rotationally invariant; linear combinations (NNs) misrepresent structure.

ML Contests 2024: 16 LightGBM wins, 13 CatBoost, 8 XGBoost.

### When Neural Networks Win:
- Large datasets (>100K rows) with dense, informative features
- Heavy feature interactions that GBDTs approximate poorly
- Embedding-rich inputs (high-cardinality categoricals, text, mixed modalities)
- Small datasets (<1K rows) — TabPFN dominates
- Source: "When Do Neural Nets Outperform Boosted Trees on Tabular Data?" (arXiv 2305.02997)

---

## Deep Learning Architectures for Tabular Data

### TabM (ICLR 2025) — Current State of the Art
https://arxiv.org/abs/2410.24210 | Code: https://github.com/yandex-research/tabm

**Mechanism:** Single MLP-like model that efficiently imitates an ensemble of MLPs through weight sharing and parallel predictions. Fits in one forward pass.

**Benchmarks (46 datasets):** Best average rank 1.7 vs 2.9 for nearest competitor. Rivals CatBoost/XGBoost/LightGBM — first MLP-derivative to do so consistently.

**Why it works:** Parameter-efficient ensembling acts as built-in regularization — outperforms naive ensemble of independently-trained MLPs.

**Usage in Kaggle:** Used in CIBMTR HCT survival prediction competition (2024–2025).

### TabPFN v2 / v2.5 — In-Context Learning (Nature 2024)
https://www.nature.com/articles/s41586-024-08328-6

**Mechanism:** Pre-trained Transformer that does in-context learning — NO training loop required.

**Win rate vs XGBoost:**
- ≤10K rows, ≤500 features: **100% win rate**
- Up to 100K rows, 2K features: **87% win rate**
- >100K rows: XGBoost competitive

**TabPFN-2.5 (Nov 2025):** Matches AutoGluon tuned 4 hours in a single forward pass of 2.8 seconds.

**Usage:** `pip install tabpfn`. Sklearn API. Include as diverse predictor in ensembles.

**Limitation:** Performance degrades beyond ~100K samples (v2.5). Use GBDTs for large datasets.

### FT-Transformer
- Embeds each feature as independent token, applies multi-head self-attention for inter-feature relationships.
- Library: `pip install rtdl`
- Implementation tip: Use piecewise-linear or periodic embeddings for numericals. Add `[CLS]` token for classification head.
- Consistently the strongest pure-DL model for medium-to-large datasets before TabM.

### RealMLP (2024)
- MLP with robust preprocessing (quantile normalization, numerical embeddings, better initialization).
- "Nearly matches GBDTs" tier on 2024 survey across 68 datasets.
- Used in Playground S6E2 top ensemble.

---

## Advanced GBDT Hyperparameter Tricks

### XGBoost
- `booster=dart`: Dropout on trees. Better on noisy tabular data. Slower but often +0.1–0.3% LB vs `gbtree`.
- `tree_method=hist` + `device=cuda`: Required for GPU, ~10–20x speedup.
- `monotone_constraints`: Enforce monotonic relationships where domain knowledge exists.
- `interaction_constraints`: Prevent specific feature pairs from co-appearing in a split.
- `grow_policy=lossguide` + `max_leaves`: Leaf-wise growth like LightGBM.
- OOF trick: Use `early_stopping_rounds=50` on validation fold → retrain with found n_estimators on full data.

### LightGBM
- `boosting_type=goss`: Gradient-based One-Side Sampling — fastest for large datasets.
- `boosting_type=dart`: Random tree dropping, can improve generalization.
- `num_leaves`: Most critical parameter. Rule of thumb: `2^max_depth * 0.6`. Typical: 63–255.
- `min_child_samples`: Critical anti-overfit regularizer. For noisy tabular: 20–100.
- `path_smooth`: Smoothing factor for leaf values — underused, helps with noisy targets.
- `feature_fraction_bynode`: Resample columns at each node (more randomness, often better than `feature_fraction` alone).
- `linear_tree=True`: Fits linear models in leaf nodes — very effective for near-linear relationships.

### CatBoost
- Ordered boosting (default): Avoids target leakage from in-sample TE. Best for small datasets.
- `cat_features` list: Pass raw categoricals without encoding — CatBoost handles with ordered target statistics.
- `border_count`: Default 254; increase to 1024 for financial/medical data precision.
- GPU: Up to 15x speedup on V100. `task_type='GPU'`.

### Optuna Tuning Recipe
- Use `TPESampler` (default) or new `GPSampler` (Optuna 4.7.0+, more sample-efficient).
- `MedianPruner` to stop bad trials early.
- Target CV score as objective.
- 50–200 trials typically sufficient.
- Source: https://medium.com/@o.ankeli/how-i-used-optuna-to-win-an-air-quality-forecasting-kaggle-competition-f420fa642bb5

---

## RAPIDS cuML / cuDF

Speedups vs pandas/scikit-learn CPU:
- KNN (MNIST): **600x**
- SVM: **15x train, 28x inference**
- UMAP: **311x**
- cuDF groupby/aggregation: **10–100x**

**Chris Deotte 2025 1st place:** Used RAPIDS to train 72 models (XGBoost, LightGBM, CatBoost, NNs, TabPFN, KNN, SVR, Ridge, RF) in a 3-level stack — infeasible on CPU.

```python
import cudf  # drop-in pandas replacement
df = cudf.read_csv("train.csv")
df["feature_X_mean_by_Y"] = df.groupby("Y")["X"].transform("mean")
```

---

## AutoGluon in Kaggle 2024

In 2024: **7 gold medals and 15 top-3 finishes** in 18 tabular Kaggle competitions.

1st/2nd/3rd in $75K AutoML Grand Prix were all AutoGluon-based.

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label="target").fit(
    train_data, presets="best_quality", time_limit=3600
)
```

Use AutoGluon OOF predictions as a free diverse member in your ensemble stack.

---

## Feature Engineering Automation

### Polars for Rapid Generation (2024)
Polars in winning solutions: grew from 0 to 7 winning solutions in 2024.

```python
import polars as pl
df = pl.read_csv("train.csv")
stats = df.group_by("category").agg([
    pl.col("value").mean().alias("value_mean"),
    pl.col("value").std().alias("value_std"),
    pl.col("value").quantile(0.9).alias("value_q90"),
])
```

### Novel FE Tricks from 2024–2026 Winners
- **Digit extraction:** Extract units/tens/decimal digits from floats — "captures hidden structure" (S6E2)
- **Column concatenation search:** GPU-accelerated brute-force over all 2–6 column combos → apply TE/CE battery. 145,000 combos searched; 170 found useful (S4E12)
- **Row-wise sorted features:** Sort all feature values within each row; captures rank structure (S4E5)
- **Count threshold features:** Count of features exceeding threshold values (S4E5)
- **WoE/Entropy from original dataset:** In synthetic Playground comps, use source dataset statistics as features (S6E2)
- **Genetic Programming:** `gplearn` for nonlinear interaction features, adds diversity not single-model gains (S6E2)
- **DVAE latent representations:** Noisy autoencoder embeddings add diversity even without improving single model CV (S6E2)
- **Frequency encoding:** How often each value appears — complements target encoding, safe at inference (S6E2)

### Genetic Algorithm Feature Selection
- Represent feature set as binary chromosome
- Fitness = CV score
- More systematic than random search for large feature spaces
- Example: https://www.kaggle.com/code/ar89dsl/genetic-algorithms-for-feature-selection-0-13236

---

## Post-Processing Techniques

### RankGauss
Transform predictions to Gaussian via rank. Useful for regression with heavy-tailed targets.
```python
from scipy.stats import rankdata
from scipy.special import erfinv
import numpy as np
ranks = rankdata(predictions)
ranks_norm = (ranks - 0.5) / len(ranks)
gaussian = erfinv(2 * ranks_norm - 1)
```

### Temperature Scaling
```python
# Tune temperature on held-out validation set
calibrated_prob = sigmoid(logit(raw_prob) / temperature)
```
Does not affect AUC/ranking but improves log-loss metrics and downstream stacking.

### Probability Clipping
`np.clip(predictions, 1e-6, 1 - 1e-6)` — often gains 0.001–0.005 on log-loss metrics.

### Rank Blend
```python
rank_pred = predictions.rank() / len(predictions)
final = 0.7 * predictions + 0.3 * rank_pred
```
Reduces sensitivity to outlier predictions.

---

## Multi-Label/Multi-Class Tricks

### Multi-Label:
- Binary classifiers per label (Binary Relevance) as baseline
- ClassifierChain to capture inter-label dependencies
- Threshold tuning per label via F1 maximization on OOF
- Pseudo-labeling + iterative retraining

### Multi-Class:
- Label smoothing: `(1 - eps)` and `eps/K`, eps=0.05–0.1. Reduces overconfidence.
- Focal loss: upweight hard examples for imbalanced multi-class.
- Ordinal encoding for ordered classes.

---

Sources:
- TabM paper: https://arxiv.org/abs/2410.24210
- TabPFN Nature 2024: https://www.nature.com/articles/s41586-024-08328-6
- "Why tree-based models outperform DL on tabular": https://arxiv.org/abs/2207.08815
- "When NNs outperform boosted trees": https://arxiv.org/abs/2305.02997
- NVIDIA Grandmasters Playbook: https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/
- RAPIDS cuML 600x speedup: https://developer.nvidia.com/blog/accelerating-k-nearest-neighbors-600x-using-rapids-cuml/
- cuDF feature engineering 1st place: https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-kaggle-competition-with-feature-engineering-using-nvidia-cudf-pandas/
- ML Contests 2024: https://mlcontests.com/state-of-machine-learning-competitions-2024/
- hillclimbers library: https://github.com/Matt-OP/hillclimbers
