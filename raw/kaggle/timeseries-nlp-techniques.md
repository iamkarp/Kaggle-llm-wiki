# Kaggle Time Series Forecasting & NLP/LLM Techniques (2022–2025)

Compiled from competition writeups, mlcontests.com annual reports, arXiv papers, Kaggle discussions. April 2026.

---

## PART 1: TIME SERIES FORECASTING

### Model Hierarchy: What Actually Wins

**GBDTs dominate tabular time series.** Neural networks win when series are long, numerous, and lack useful exogenous variables.

2024 ML Contests: 16 LightGBM wins, 13 CatBoost, 8 XGBoost. In DrivenData Water Supply Forecast Rodeo 2024 (largest prize pool): 1st place used CatBoost + LightGBM ensemble, no deep learning.

**GBDT key trick:** Train one model per forecast horizon rather than one model for all horizons. Lets each model learn which features are relevant at each look-ahead step.

**M5 (Walmart):** 220 LightGBM models total — 10 store-level, 30 store-category-level, 70 store-department-level. Each series averaged from ~6 models.

**Neural Networks win when:**
- Very large dataset, hierarchical, strong periodicity without good exogenous covariates
- N-BEATS outperformed M4 winner by ~3% in follow-on benchmarks; improved statistical baselines ~11%
- TFT (Temporal Fusion Transformer): Strong when you have known future covariates (promotions, holidays)
- N-HiTS: Hierarchical interpolation for multi-scale patterns, competitive with N-BEATS on long-horizon tasks

**Foundation Models (emerging 2025+):**
- Google TimesFM (200M params, pre-trained on 100B real-world time points): zero-shot performance close to supervised baselines
- Not yet proven in competition wins but worth watching

---

### Feature Engineering for Time Series

#### Lag Features
- Raw lags at domain-relevant intervals: lag-1, lag-7, lag-28, lag-364
- Lags grouped by entity: `lag_7_by_store`, `lag_7_by_item`, `lag_7_by_store_item_combo`
- M5 winner used lags by store, item, store-class, store-department, and all pairwise combinations
- **CRITICAL: All features must use `.shift(1)` before windowing — current time step's value MUST NOT be included**

#### Rolling Window Statistics
- Windows: 7, 14, 28, 56, 90, 180 days
- Statistics: mean, median, std, quantiles (10th, 90th)
- EWMA (exponentially weighted moving avg) — captures recent trends with decay
- Rolling stats were the **distinguishing feature of top-2 solutions** in multiple retail forecasting competitions

#### Fourier / Cyclical Features
Encode day-of-week, month, week-of-year as sine/cosine pairs:
```python
df['sin_dow'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['cos_dow'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
```
Critical for long-period seasonality (yearly, quarterly).

#### Calendar and Exogenous Features
- Holiday indicators (national + regional), event flags (Super Bowl, Black Friday)
- Promotions, price changes, markdown events (critical in retail)
- Weather data (temperature, precipitation) — critical in energy/demand
- Item/store metadata embeddings

#### Target Encoding
- Historical mean sales by entity group
- Always use smoothed/regularized encodings to prevent leakage

---

### Cross-Validation for Time Series

**Standard k-fold is WRONG for time series.**

#### Walk-Forward (Expanding Window) — Gold Standard
- Train on [0, t], validate on [t+1, t+h]. Expand training window each fold.
- `sklearn.model_selection.TimeSeriesSplit` implements this.
- Forces simulation of real production conditions.

#### Sliding Window Validation
- Train on fixed-size N-month window, validate on next M months.
- Useful when older data is less relevant (concept drift, structural breaks).

#### Purged Cross-Validation — Critical for Financial Data
- Removes training observations whose labels overlap in time with test labels.
- Adds an **embargo period** after each test fold to prevent correlated label leakage.
- Libraries: `mlfinlab.PurgedKFold`, `skfolio.CombinatorialPurgedCV`
- CPCV significantly outperforms standard methods on Deflated Sharpe Ratio and Probability of Backtest Overfitting metrics.

#### Key Rule: Match CV to Test Structure
- If test = future 28 days → validate on 28-day windows.
- If test has gap between public and private LB → build gap into CV splits.
- Top competitors validate across 3–5 different time windows and look for consistency.

---

### M6 Competition (Financial Forecasting, 2022–2023)
**100 S&P 500 stocks/ETFs, 12 monthly rounds**

- Winner: **AutoTS** (automated time series Python library), combining preprocessing methods + multiple forecasting models optimized via **genetic algorithm**. Automated ensembling beat hand-crafted deep learning.
- Second approach: "quasi-average" predictions — exploiting efficient market hypothesis rather than trying to beat it.
- Only ~23% of participants beat the benchmark in the forecasting track.
- All 5 forecasting winners updated submissions every round (adaptive/online methods mattered).
- **Lesson:** In financial time series, **regression to the mean beats overconfident extrapolation**.
- Source: https://arxiv.org/pdf/2310.13357

---

### Ensemble Strategies for Time Series
- Simple averaging of multiple GBDTs (different feature sets, seeds) is first-order improvement
- Weighted averaging by OOF validation score
- Two-level stacking: GBDT base → Ridge/GBDT meta on OOF predictions
- **Diversity > Count:** 3 diverse models (LightGBM + LSTM + statistical model) > 10 similar LightGBM models
- Global models (trained on all series) beat local models (one per series) in most modern competitions

---

## PART 2: NLP / LLM COMPETITIONS

### Architecture Transition: DeBERTa → Decoder LLMs

**2022–2023: DeBERTa era**
- DeBERTa-v3-large: near-universal backbone for NLP classification.
- Winning solutions: ensemble of 7–11 DeBERTa models with different:
  - Context lengths (512, 1024, 1536 tokens)
  - Pooling strategies (mean, max, attention-weighted, CLS, concat first+last)
  - Random seeds
  - Training data compositions

**2024: Decoder LLM shift**
- Decoder-only models winning in generation/preference tasks.
- Most common winner backbones 2024: Llama 3, Gemma 2, Mistral, Qwen 2, DeepSeek.
- For classification: DeBERTa still competitive.
- ML Contests 2024: "decoder-only GPT-style models are starting to disrupt the status quo of DeBERTa models."

---

### LLM Usage Patterns in Winning Solutions

**Pattern 1: LoRA/QLoRA Fine-tuning**
- LMSYS Chatbot Arena 2024 ($100K): Fine-tuned Gemma2-9B + Llama3-70B + Qwen2-72B with LoRA on 8× A100-80GB.
- Then **knowledge distillation** from large models into Gemma2-9B.
- Final inference: only 8-bit quantized Gemma2-9B.
- Winner: *"Distillation is a very promising approach, especially where inference constraints are a limiting factor."*

**Pattern 2: Quantization for Inference**
- KDD Cup 2024: Fine-tuned multiple Qwen2-72B with LoRA at train time; **4-bit quantization** at test time.
- 4-bit quantization reduces size ~75% vs 16-bit, enabling 72B models in Kaggle submission environments.

**Pattern 3: Encoder LLMs as Feature Generator**
- Generate embeddings → feed into lightweight head (Ridge, LightGBM, small MLP).
- Effective when LLM can't be fine-tuned within GPU budget.

**Pattern 4: Synthetic Data Generation**
- CommonLit 2023 winners: generated custom training data using ChatGPT + other LLMs.
- Pseudo-labeling on test data using confident predictions, then retraining (semi-supervised).

---

### Fine-Tuning Tricks for Transformers/LLMs

#### Layer-Wise Learning Rate Decay (LLRD)
Apply different LRs per layer: top (classifier head) = highest LR; bottom (early blocks) = lowest LR.
Typical: top_lr = 3.5e-6, decay multiplier 0.9 per layer → bottom ~1e-6.
**Prevents catastrophic forgetting** of pre-trained representations.

#### Adversarial Weight Perturbation (AWP)
Add small perturbations to model weights during training to find flatter loss minima.
- Double perturbation: perturb both input AND weights.
- Near-universal in winning NLP solutions from 2022 onward.
- Implementation: https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp
- Note: Adds ~2× training time.

#### Multisample Dropout
Create multiple dropout masks per forward pass, average the losses.
- Accelerates training convergence (fewer epochs needed), improves generalization.
- Low overhead: most compute is in attention layers.
- Paper: "Multi-Sample Dropout for Accelerated Training" (Inoue, 2019, arXiv 1905.09788)

#### Gradient Checkpointing
Trades compute for memory: recompute activations during backward pass rather than storing them.
- Saves 10× memory, adds ~10–20% compute.
- Enables longer sequences or larger batch sizes on fixed GPU budget.

#### Two-Stage Training
1. Pre-train on large related dataset (domain adaptation)
2. Fine-tune on competition dataset only
- Automated Essay Scoring 2.0 (2024) winner: jumped from rank 619 (public) to 1st (private) using this.

#### Custom Pooling Strategies
Combine multiple methods:
- Mean pooling over all tokens
- Max pooling
- Weighted attention pooling
- Concatenation of first + last token
- "Gem pooling" (generalized mean)
Models differing only in pooling strategy give genuine ensemble diversity at low cost.

#### MLM Pre-training on Competition Data
Before fine-tuning, run additional Masked Language Model pre-training on competition's unlabeled text.
- 15–20% further MLM pre-training epochs on competition text.
- Adapts model vocabulary distribution to domain.

---

### NLP Cross-Validation
- **Stratified K-Fold (k=5 or k=4)** standard for classification.
- Stratify on binned target buckets for regression (essay scoring, readability).
- **Multi-label stratification:** `iterstrat` library (iterative stratification).
- **OOF predictions** for: generalization estimation, stacking meta-models, ensemble weight calibration.
- Adversarial validation for distribution shift detection.

### NLP Ensemble Strategies

**Stacking (Two-Level):**
- Level 1: Multiple diverse base models → OOF predictions
- Level 2: Ridge / LightGBM / small NN on OOF predictions

**Blending (Weighted Averaging):**
- Weights via Nelder-Mead optimization against OOF score
- Hill-climbing weight search

**Rank Averaging:**
- Convert predictions to ranks, average ranks.
- More robust than averaging raw probabilities when models poorly calibrated.

**Model Diversity Sources (most to least impactful):**
1. Different backbone architectures
2. Different context lengths / truncation strategies
3. Different pooling methods
4. Different training data subsets / augmentations
5. Different random seeds

---

### Kaggle Inference Efficiency

Kaggle GPU submissions: typically single T4/P100 (16GB), 9-hour time limit.

- **4-bit quantization (GPTQ / bitsandbytes NF4):** Fits 7B in ~4–5GB, 13B in ~8GB.
- **Flash Attention 2:** ~2–4× faster attention, lower memory.
- **Dynamic padding:** Group sequences by length, pad within batch only.
- **torch.compile (PyTorch 2.0+):** ~20–30% inference speedup.
- **FP16/BF16 inference:** Standard; don't use FP32.
- **Mixed precision training (AMP):** Faster training + lower memory.

---

### Competition Results Reference

| Competition | Year | Winner Approach | Key Trick |
|---|---|---|---|
| LMSYS Chatbot Arena | 2024 | Gemma2-9B via distillation from Llama3-70B + Qwen2-72B | Knowledge distillation + 8-bit inference |
| Automated Essay Scoring 2.0 | 2024 | DeBERTa-v3-large ensemble | Two-stage training + distribution shift analysis |
| KDD Cup 2024 | 2024 | Multiple Qwen2-72B with LoRA | 4-bit quantization for inference |
| CommonLit Evaluate | 2023 | DeBERTa-v3-large + synthetic data | ChatGPT data augmentation + pseudo-labeling |
| M5 Walmart Forecasting | 2020 | 220 LightGBM models | Hierarchical model decomposition + lag features |
| M6 Financial Forecasting | 2023 | AutoTS genetic algorithm | Automated ensembling of simple models |

---

Sources:
- ML Contests 2024 report: https://mlcontests.com/state-of-machine-learning-competitions-2024/
- M5 forecasting paper: https://www.sciencedirect.com/science/article/pii/S0169207021001874
- "Learnings from Kaggle's Forecasting Competitions": https://arxiv.org/pdf/2009.07701
- LMSYS 1st place: https://github.com/tascj/kaggle-lmsys-chatbot-arena
- AWP implementation: https://www.kaggle.com/code/itsuki9180/introducing-adversarial-weight-perturbation-awp
- M6 competition paper: https://arxiv.org/pdf/2310.13357
- HuggingFace QLoRA blog: https://huggingface.co/blog/4bit-transformers-bitsandbytes
- Purged CV: https://www.kaggle.com/code/marketneutral/purged-time-series-cv-xgboost-optuna
- neptune.ai text classification tips: https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions
