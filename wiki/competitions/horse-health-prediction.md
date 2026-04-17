---
title: "Horse Health Prediction (Kaggle Playground)"
tags: [kaggle, tabular, multiclass, f1, lightgbm, xgboost, catboost, ensemble, small-data]
date: 2026-04-15
source_count: 2
status: active
---

## Summary
Kaggle Playground tabular 3-class classification: predict final horse colic case outcome (`lived` / `died` / `euthanized`). Small dataset (988 train / 247 test) mirroring the classic UCI horse-colic problem. Metric: **micro-averaged F1**.

## Competition Metadata
- **Platform**: Kaggle
- **Prize**: — (Playground)
- **Metric**: micro-F1 (equivalent to accuracy here since every row gets one label)
- **Deadline**: 2026-04-16 (≈16h from ingest)
- **Team**: Jason (solo)
- **Best OOF Score**: 0.71862 (10-fold stratified CV, seed-averaged blend)
- **Best LB Score**: TBD
- **Submission Count**: 1 primary + 3 diagnostic so far

## Data Shape
- train: 988 × 29 (incl. id + outcome)
- test: 247 × 28
- target distribution: lived 46.5%, died 33.2%, euthanized 20.3% (mild imbalance)
- missingness concentrated in categorical medical assessments: `abdomen` (156), `rectal_exam_feces` (150), `nasogastric_tube` (60), `peripheral_pulse` (48)
- numeric columns: no missing in test, small missing in train
- `hospital_number`: 239 unique train / ~180 unique test with **124 overlap** → useful for target / frequency encoding

## Strategy Summary
Straight play from [[../strategies/kaggle-competition-playbook]]:

1. **CV**: StratifiedKFold(10) — small + imbalanced → more folds for lower variance on OOF.
2. **Feature engineering**:
   - Decompose `lesion_1` into `site / type / subtype / specific` (horse-colic code convention).
   - `has_lesion_2`, `has_lesion_3`, `n_lesions`.
   - Clinical ratios: `pulse/resp`, `pcv/tp`, `|rectal_temp - 37.8|` (normal equine temp), `tachy_severity = max(pulse-40, 0)`.
   - OOF target encoding on `hospital_number` (per-class, smoothing=5); plus frequency encoding.
3. **Missing handling**: categoricals → `"__NA__"` class; numeric → median + `_was_missing` flag (only added when any row had it missing).
4. **Models**: LightGBM + XGBoost + CatBoost, all **seed-averaged over 3 seeds** (42 / 1337 / 2024) within each fold — critical on 988 rows where single-seed noise is ~0.01 F1.
5. **Blend**: fourth-root weighting on OOF scores with baseline-shifted by 0.5 (scores are in a narrow 0.70–0.72 band). Arithmetic vs geometric blend are both computed; arithmetic won here.
6. **No OOF-grid-search on blend weights** — the first run did that and picked "CatBoost alone" which disappeared after seed-averaging. Classic OOF overfit with 988 rows.

## Submission Discipline Applied
- Kept `id` order aligned with `sample_submission.csv` (verified).
- OOS score in filename: `submission_oof_microF1_0.71862.csv`.
- Per-model diagnostic submissions also saved for diversity analysis.

## What Worked
- **Seed averaging** within each model — individual scores stabilized around 0.706–0.712 (previously 0.708–0.723 single-seed spread).
- **Blend > best individual** by ~0.007 F1 — the three trees disagree enough to add signal.
- **Lesion decomposition** — trees pick up `lesion_1_site` as informative (the raw 4–5 digit code is effectively a high-cardinality categorical that trees struggle to split meaningfully).
- **Hospital target encoding** — real overlap (124 shared hospitals) makes this a legitimate feature, not leakage.

## What Didn't Work
- Grid-search over blend weights on OOF — overfit. Fourth-root weighting is the right default on a dataset this small.
- Geometric-mean blend of probabilities was 0.003 F1 below arithmetic — unusual, but the models are already close enough in calibration that logs add noise.

## Submission History
| Version | Config | OOF micro-F1 | Notes |
|---------|--------|--------------|-------|
| v1 blend | LGB+XGB+CAT, single seed, grid-searched weights | 0.72267 (OOF overfit) | Picked CatBoost alone; rejected |
| v2 blend | Seed-avg ×3, fourth-root weighting, arithmetic | **0.71862** | Current best, submitted |

## Open Questions
- Logistic regression + one-hot as a 4th ensemble component for additional diversity? Small dataset makes this appealing.
- Additional lesion decomposition: the 5-digit codes (e.g., `11300`) encode `site=11` (all intestinal) — does mapping site=11 to a separate "multi-site" flag help?
- Would stacking with a ridge meta-learner beat the fourth-root blend? On 988 rows, probably noisy.

## Sources
- [[../../raw/kaggle/horse-health-readme.txt]] — official competition readme
- [[../../raw/kaggle/horse-health-pipeline.py]] — full training pipeline

## Related
- [[../strategies/kaggle-competition-playbook]] — source workflow this followed
- [[../concepts/validation-strategy]] — CV choice rationale for imbalanced multiclass
- [[../concepts/target-encoding-advanced]] — OOF target encoding for `hospital_number`
- [[../concepts/ensembling-strategies]] — ensemble weighting scheme (fourth-root blending)
