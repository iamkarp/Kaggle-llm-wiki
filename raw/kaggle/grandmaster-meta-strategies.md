# Kaggle Grandmaster Meta-Strategies

Compiled from Chris Deotte (NVIDIA, 4x GM), KazAnova/Marios Michailidis (former #2 globally), 
Anthony Goldbloom (Kaggle CEO), ML Contests annual reports, winner interviews. April 2026.

---

## The Two Winning Approaches (Anthony Goldbloom)

1. **Handcrafted feature engineering**: Test hundreds of hypotheses; most fail. The one that works wins the competition.
2. **Deep learning / ensemble neural nets**: For unstructured or unusual data. For structured tabular, GBDTs have dominated since Kaggle's founding.

---

## What Separates Top 1% from Top 10%

1. **Multi-level stacking**, not just blending. Top 10% average predictions. Top 1% build 2–3 layer stacks (StackNet pattern).
2. **Validation scheme mastery.** Top 1% spend 20–30% of time on CV setup before touching model parameters.
3. **Pseudo-labeling done right.** Compute K separate pseudo-label sets for K folds — no validation row ever sees test-derived labels from models that trained on it.
4. **Adversarial validation.** Always check train/test distribution shift (AUC ~0.5 = same; AUC >0.7 = serious shift). Filter training rows to resemble test if shift exists.
5. **Speed of experimentation.** GPU-accelerated frameworks enable 100+ model variants in the time others run 10.

---

## CV vs LB: The Core Rule

**Trust Your CV** — repeated by virtually every Grandmaster as the most important principle.

Mathematical reason: Public LB typically uses only 20–35% of the test set. With small public test, a single noisy prediction cluster can swing 50+ ranks. Private LB (65–80% of test) is far more stable.

### Conditions for Trusting CV:
- CV must mirror the test split (time-based test → time-split CV; new entities → GroupKFold)
- Monitor CV-LB correlation: log both scores for every submission. Divergence = warning sign.
- Stable CV at lower value > bouncing CV at higher value.

### When to Trust LB Over CV:
- Very small training data (CV has high variance)
- Known public test leakage you can exploit
- Very few possible LB submissions (near-optimal baselines)

### Shake-Up Survival Playbook:
1. Every submission must move your CV score. LB-only improvement = treat as noise.
2. Simulate public/private split via CV folds; check ranking stability between them.
3. Before deadline: Is best CV model near top of LB? If yes → safe. If not → pick CV model.

---

## Validating Feature/Model Improvement: Real vs Noise

- Run ablation over ALL folds, compute mean + stddev of CV delta.
- If improvement < 1 stddev of CV variance across folds → noise.
- Heuristic: if improvement only visible in 2/5 folds → likely noise.
- Use permutation importance (shuffle feature, measure score drop on validation set) vs default feature importance. Permutation = ground truth.
- For changes passing CV bar: do one LB submission as sanity check only.

---

## Seeds and Folds

### Folds:
- 5-fold: standard for most competitions
- 10-fold: small datasets
- StratifiedKFold: classification with class imbalance
- GroupKFold: entity-level correlation (patient data, time series by entity)
- TimeSeriesSplit: forecasting — NOT using time-aware splits is a fatal mistake
- Leave-one-out: essentially never used in competition settings

### Seeds:
- Early in competition: 1–2 seeds (exploring, not optimizing)
- Late game (last 1–2 days): ensemble across 5–20 seeds — known score booster
- Same model pipeline × 5 different seeds, averaged: +0.001–0.003 CV on typical competition
- Seeds matter most for: neural networks, GBMs with small learning rates, anything with randomized subsampling

### Stacking rule: Every model MUST use exact same fold splits. Fix splits at competition start.

---

## Final Submission Selection

**Standard: Two diverse submissions:**
1. Highest CV model/ensemble
2. A very different approach (different algorithm family, different feature set)

Rationale: If CV is well-calibrated, #1 should win. #2 is insurance for subtly wrong CV.

**If you've been LB chasing (oversubmitting):** Pick highest CV model for BOTH slots.

**Practical checklist:**
1. Log CV vs LB for all submissions; identify correlation
2. Top 3 CV models identified
3. Check for CV-good/LB-suspiciously-high models (overfit suspects)
4. Pick top CV + most diverse runner-up
5. NLP/CV: consider one single-model + one ensemble

---

## Data Leakage Detection

**Row-level:**
- Check for duplicate/near-duplicate rows between train and test
- `pandas.DataFrame.duplicated()` across train+test on feature subsets

**Feature-level:**
- Add `random_noise = np.random.random(len(train))` — if model gives it high importance, your validation has leakage
- Features with near-zero train variance / high test variance are suspect

**Time-based leakage (most common, most subtle):**
- Any feature computed using "future" data (running means, cumulative stats)
- If CV suspiciously good on time-series problem, examine all lag features

**Adversarial Validation:**
- Combine train+test, label train=0/test=1. Train classifier.
- AUC > 0.6: distribution shift exists. Top importance features are "shift features."
- Fix: drop shift features, or weight training rows by classifier's test-likelihood score.

**Pseudo-label leakage:**
- WRONG: one set of pseudo-labels for all folds
- RIGHT: generate pseudo-labels for fold K using model trained on all OTHER folds only

---

## 2-Week Time Allocation Framework

**Days 1–3: Deep Understanding**
- Read every discussion post with 10+ votes (Grandmasters drop gold here)
- Read all public notebooks (especially EDA)
- Understand evaluation metric behavior
- Build simplest possible baseline + submit
- Establish CV framework — if this is wrong, all downstream work is wasted

**Days 4–7: Experimentation**
- Diverse baselines: GBDTs (LGB/XGB/CAT), simple NN, linear model
- Feature engineering: groupby aggregations, interactions, target encoding, date features
- Systematic experiment log (spreadsheet: description, CV, LB, notes)

**Days 8–11: Optimization**
- HPO (1–2% of final score comes from HPO — don't over-invest)
- Feature selection (drop features that consistently hurt CV)
- Model zoo building for ensemble diversity
- Pseudo-labeling if applicable

**Days 12–14: Ensembling and Hardening**
- Hill-climbing greedy ensemble: start best model, greedily add models improving OOF
- Multi-seed averaging on best models
- Final submission selection
- Leakage sanity check on full pipeline

**Realistic time investment for medal contention:** 15–30 hours/week on month-long competition.

---

## Hill-Climbing Ensemble Selection

Start with single best OOF model. Iteratively perturb each weight by small delta; keep change if it improves metric. Repeat until convergence. More robust than weight optimization because it implicitly regularizes ensemble composition.

```python
# pip install hillclimbers
from hillclimbers import climb
best_weights = climb(predictions_list, y_true, metric="roc_auc", maximize=True, n_iterations=1000)
```

---

## Rank Averaging

When ensembling models with different score scales, rank-transform predictions before averaging:
```python
rank_pred = (predictions.rank() / len(predictions))
```
Prevents scale-dominant models from swamping the ensemble.

---

## Adversarial Validation for Distribution Shift Correction

After identifying shift features, weight training examples by how "test-like" the adversarial classifier judges each row to be. Shifts your model toward the test distribution without discarding data.

---

## Key Quotes

**KazAnova:** "Understanding the problem and evaluation metric is key. Creating a reliable cross-validation process that resembles the leaderboard or test set will allow you to explore many different algorithms knowing their true impact."

**Anthony Goldbloom:** "The way winners find successful features is to test lots and lots of hypotheses, with the vast majority not working out, but the one that does winning them the competition."

**ML Contests 2024:** "In tabular competitions: 16 wins with LightGBM, 13 CatBoost, 8 XGBoost. 65% of winning teams were solo competitors."

---

Sources:
- Chris Deotte NVIDIA blog: https://developer.nvidia.com/blog/author/cdeotte/
- KazAnova winning tips: https://www.hackerearth.com/practice/machine-learning/advanced-techniques/winning-tips-machine-learning-competitions-kazanova-current-kaggle-3/tutorial/
- Anthony Goldbloom W&B interview: https://wandb.ai/wandb_fc/gradient-dissent/reports/Anthony-Goldbloom-How-to-Win-Kaggle-Competitions--Vmlldzo2MzE3MDI
- Kaggle Handbook shake-up survival: https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8
- ML Contests 2024 report: https://mlcontests.com/state-of-machine-learning-competitions-2024/
