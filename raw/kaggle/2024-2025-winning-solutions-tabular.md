# 2024–2025 Winning Solutions: Tabular/Financial/Insurance Competitions

Scraped April 2026 from direct Kaggle writeup pages and mlcontests.com annual report.

---

## Home Credit – Credit Risk Model Stability (2024)
**1st Place:** yuuniee | May 2024
**Writeup:** https://www.kaggle.com/competitions/home-credit-credit-risk-model-stability/writeups/yuuniee-1st-place-solution-my-betting-strategy

- **CV strategy:** StratifiedGroupKFold (shuffled and non-shuffled). CRITICAL: CV improvements of 0.001–0.005 had LOW correlation with LB. Improvements >0.01 had HIGH correlation. FE-driven CV improvement correlated much more reliably with LB than hyperparameter-driven improvement.
- **Feature engineering:** Max/Min/Avg/Var/First/Last/Max-Min-Difference aggregations per account group. 644 features for LGBM, 661 for CatBoost (different sets — some categoricals caused LGBM degradation).
- **Models:** CatBoost (strongest, ~117 categorical features), LightGBM (diversity), DenseLight DNN. DenseLight beat FT-Transformer/TabNet.
- **What DIDN'T work:** TabNet, TabTransformer, FT-Transformer, K-means clustering, period-based income/expenditure statistics.
- **Post-processing trick (metric hack):** Score adjustment based on reconstructed temporal proxy (date_decision minus min_refreshdate correlated 0.9+ with WEEK_NUM). Applied `df.loc[condition, 'score'] = (df.loc[condition, 'score'] - 0.03).clip(0)` for rows in first half of WEEK_NUM range. Post-processing adjustment (~0.0X) completely dominated pure ML score differences (~0.00X).
- **Key lesson:** On competitions with complex temporal stability metrics, always analyze whether the metric can be exploited by time-conditional score adjustments.

---

## Playground Series S4E10 – Loan Approval Prediction (October 2024)
**1st Place:** Hardy Xu | Nov 2024
**Writeup:** https://www.kaggle.com/competitions/playground-series-s4e10/writeups/hardy-xu-1st-place-solution-catboost-all-the-way-d

- **CV philosophy:** CV measures 60% of data; public LB measures 8%. Trust CV.
- **Feature preprocessing:** Dual representation — keep both numeric and categorical copy of each feature.
- **Models:** XGBoost, LightGBM, CatBoost, NN. Each trained with Optuna finding 10 "optimal" hyperparameter sets, predictions averaged.
- **Large `max_bin` values** called out as important tip.
- **SECRET: CatBoost as meta-learner using `baseline` parameter.** Pass any model's predictions as `baseline=` to CatBoost; it learns to IMPROVE upon them:

| Model | Base CV | After CatBoost refinement | Delta |
|-------|---------|--------------------------|-------|
| LightGBM | 0.96811 | 0.96856 | +0.00045 |
| XGBoost | 0.96767 | 0.96815 | +0.00048 |
| CatBoost | 0.96972 | 0.96997 | +0.00025 |
| NN | 0.96678 | 0.96732 | +0.00054 |

Final stacking with NN on top of 4 CatBoost-refined models: CV 0.97059.

---

## Playground Series S4E12 – Insurance Premium Prediction (December 2024)
**1st Place:** Chris Deotte (NVIDIA Grandmaster) | Jan 2025
**Writeup:** https://www.kaggle.com/competitions/playground-series-s4e12/writeups/chris-deotte-1st-place-single-model-feature-engine

- **Final model:** Single XGBoost with 611 features. `learning_rate=0.001`, `n_estimators=20000`. Trained 6 hours on A100 GPU.
- **Target encoding:** kfold=10 (reduced leakage vs standard 5-fold TE).
- **Multi-representation framework per categorical column:**
  1. Label encoding
  2. Target Encoding mean
  3. Target Encoding median
  4. Target Encoding min
  5. Target Encoding max
  6. Target Encoding nunique
  7. Count Encoding
  = 7 representations per column simultaneously. GBDTs pick the most useful per split.
- **Column combination search:** Concatenate 2–6 columns as strings → apply all 7 encodings. Creates interaction features. Using RAPIDS cuDF-Pandas (10–100x faster), brute-force searched 145,000 possible column combinations. Found 170 useful combinations; top 20 published in code.
- **Treat numerical columns as categorical:** Convert floats to strings, apply same TE/CE battery.
- **Key insight:** In Sept/Nov 2024 playground comps, FE did not help. In this insurance dataset with rich categoricals, FE completely dominated.

---

## Playground Series S4E5 – Flood Probability Prediction (May 2024)
**1st Place:** aldparis | June 2024
**Writeup:** https://www.kaggle.com/competitions/playground-series-s4e5/writeups/aldparis-1st-place-solution

- **EDA discovery:** Data was a sum of Poisson distributions. Sum/std/max per row were powerful features. Sorted original features (within each row) added signal.
- **"Count threshold" features:** nb_sup6, nb_sup7, nb_sup8 — count of features exceeding each threshold.
- **Target transformation:** `target_transf = FloodProbability - mean(original_features)*0.1` to separate signal from noise.
- **Permutation feature importance + backward elimination** to drop kurtosis/skewness features.
- **30+ GBM models** with varied feature sets; Optuna HPO.
- **Grow_policy diversity:** XGBoost/CatBoost/LightGBM have different default grow policies — natural diversity.
- **Ridge with `positive=False, fit_intercept=False`** (allows negative ensemble weights — better than constrained).
- **AutoGluon integration:** Adding AutoGluon OOF predictions to ensemble: 0.96934 → 0.86939. Running 6 parallel AutoGluon variants (with/without features), fitting final Ridge ensemble.
- **Validation:** Only 2 submissions in final 10 days. CV reliable from day 1.

---

## Playground Series S4E11 – Mental Health Data (November 2024)
**1st Place:** Mahdi Ravaghi | Dec 2024
**Writeup:** https://www.kaggle.com/competitions/playground-series-s4e11/writeups/mahdi-ravaghi-1st-place-solution

- 69 models trained; final ensemble used only 24 (fewer outperformed more).
- No feature engineering, no feature dropping (including `Name` text column — GBDTs handled it).
- Four data pipelines (with/without original dataset, variant preprocessings).
- **AutoGluon as ensemble manager:** Let AutoGluon handle OOF ensembling. Significantly outperformed hill climbing, Ridge, logistic regression for this task.
- Higher CV model won over higher public LB model on private LB — trust CV.

---

## Playground Series S5E12 – Insurance Regression (December 2025)
**1st Place:** wind1234it | Jan 2026
**Writeup:** https://www.kaggle.com/competitions/playground-series-s5e12/writeups/1st-place-solution-hill-climbing-ridge-ensembl

- **Two-stage ensemble:** Hill Climbing (HC) first → Ridge Ensemble Stacking when HC plateaued. HC stuck at 0.7088X; Ridge stacking broke through.
- **Rank transformation** before Ridge stacking (alpha=10).
- **OOF alignment:** All base model OOF files MUST use same fold splits. Mismatched folds cause leakage.
- **Post-cutoff CV:** CV using data after public LB cutoff was more reliable than standard CV.

---

## Playground Series S6E2 – Binary Classification (February 2026)
**1st Place:** Masaya Kawamata | Mar 2026 (comprehensive writeup)
**Writeup:** https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t

### 7 Feature Engineering types:
1. Quantile binning + equal-width binning + simple rounding
2. **Digit features:** Extract units digit, tens digit, hundreds digit, decimal digits from numerical variables — "captures hidden structure"
3. All features as categorical (string format)
4. Frequency encoding
5. Genetic Programming features using `gplearn` — nonlinear interactions for diversity
6. Original dataset statistics (WoE, Entropy, smoothed target mean)
7. **Denoising Variational Autoencoder (DVAE)** latent representations for diversity

### Best single models:
- AutoGluon (BASE+BIN+DIGIT+ALL_CATS): CV 0.955747
- RealMLP same features: 0.955739
- CatBoost same: 0.955686
- XGBoost + target encoding: 0.955663
- RGF (lower CV but high ensemble contribution): 0.954980

### Ensemble: ~150 OOF files → Optuna with 2500 trials → ~15 selected. Ridge (alpha=10) for final stacking.

### Full data retraining: `n_estimators = 1.25 × average best CV iteration`, 20 random seeds averaged.

### CRITICAL INSIGHT: CV–LB relation breakdown threshold
Winner had CV=0.955865 (best) but chose ~0.955780. At CV>0.95578, improvements no longer translated to LB gains — "split overfitting" regime detected by tracking multiple actual submissions. **Identify the CV–LB relation breakdown threshold and don't go above it.**

### What DIDN'T work: pseudo-labeling (soft and hard), knowledge distillation, very deep GBDT, high-order interaction expansion, non-DVAE autoencoders, nonlinear stacking, averaging all OOFs without selection.

---

## State of ML Competitions 2024 (mlcontests.com)

- GBDTs dominated: 16 LightGBM wins, 13 CatBoost, 8 XGBoost
- Multi-library ensembles are the norm
- Deep learning remained secondary for structured tabular data
- CatBoost excels with high-cardinality categoricals; LightGBM for diversity
- Source: https://mlcontests.com/state-of-machine-learning-competitions-2024/
