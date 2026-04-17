---
title: "March Mania v6 Ensemble — Weighted 3-Pipeline Architecture"
tags: [ensemble, xgboost, lightgbm, logistic-regression, calibration, ncaa, kaggle]
date: 2026-04-14
source_count: 2
status: active
---

## Summary
The v6 ensemble is Jason's best March Mania submission (0.02210 Stage 1 LB). It combines three independently trained pipelines with fixed weights. The key design principle: balance high-variance deep trees with lower-variance linear/hybrid models to reduce overfitting across tournament years.

## Architecture

```
Final = 0.35 × v2.9_Elite + 0.35 × v2.8_Advanced + 0.30 × v5_Hybrid
```

| Component | Weight | Type | Description |
|-----------|--------|------|-------------|
| v2.9 "Elite" | 35% | XGBoost deep | Deepest trees, most expressive |
| v2.8 "Advanced" | 35% | XGBoost moderate | Balanced depth/regularization |
| v5 "Hybrid" | 30% | Meta-ensemble | Diversity buffer |

## Component Models

### v2.9 "Elite" (35%)
- **Model**: XGBoost
- **n_estimators**: 600
- **max_depth**: 9
- **learning_rate**: 0.01
- **Features**: Four Factors, Elo Ratings, Momentum (30-day), SOS, Tournament Experience, Massey Ordinals, Pace, Pythagorean Wins

### v2.8 "Advanced" (35%)
- **Model**: XGBoost
- **n_estimators**: 500
- **max_depth**: 8
- **learning_rate**: 0.015
- **Features**: Similar to v2.9 but slightly shallower; different regularization

### v5 "Hybrid" (30%)
A meta-ensemble of diverse model types:
- XGBoost with depths 5, 6, 7, 8
- LightGBM
- Logistic Regression (Platt-calibrated)
- Elo-only baseline

The hybrid component provides regularization against the two XGBoost-heavy components. The Elo-only baseline is a useful anchor — if Elo disagrees strongly with the tree models, it's worth investigating.

## Rationale
- Two deep XGBoost components capture non-linear basketball dynamics (matchup-specific features, conference effects)
- v5 hybrid provides ensemble diversity — particularly the calibrated LogReg and Elo baseline prevent extreme probabilities
- Fixed 35/35/30 weights chosen empirically; slight downweight on hybrid reflects its lower peak performance but value for variance reduction
- Calibration via Platt scaling keeps probabilities well-behaved in log-loss / MSE space

## Feature Importance (General)
Based on domain knowledge and observed model behavior:
1. **Elo Rating** — strongest single predictor of win probability
2. **Pythagorean Win %** — true strength signal beyond raw W/L
3. **Four Factors** (eFG%, TOV%, ORB%, FT rate) — modern basketball efficiency metrics
4. **Massey Ordinals** — consensus strength-of-schedule ranking
5. **Momentum (30-day)** — recent form heading into tournament
6. **Tournament Experience** — teams with deeper tournament history tend to perform
7. **SOS (Strength of Schedule)** — adjusts for conference quality
8. **Pace** — tempo normalization

## Calibration
- v5 LogReg component uses Platt scaling (sklearn `CalibratedClassifierCV`)
- XGBoost components output raw probabilities — generally well-calibrated by default on balanced tournament data
- No post-hoc calibration applied to final ensemble output

## Reproducibility
```bash
cd /Users/admin/Documents/Openclaw/workspace/
bash reproduce_v6.sh
# Verified on big-brother; produces submission matching 0.02210 LB score
```
Output: `v6_optimized_stage1_repro.csv`

## Results
| Stage | Score | Notes |
|-------|-------|-------|
| Stage 1 LB | **0.02210** | Verified honest score |
| Stage 2 | Submitted | Awaiting final tournament results |

## What v11 and v12 Taught Us
- **v11 (Adjusted Efficiency)**: Adding adjusted efficiency features (accounting for opponent quality) didn't improve. Possible cause: leakage, noise, or the base features already implicitly capture this.
- **v12 (Recency Weighting)**: Weighting recent games more heavily hurt performance. Tournament outcomes seem to require full-season signals, not just late-season form.
- **Conclusion**: v6 strikes the right balance. Further improvement likely requires new data sources or fundamentally different architecture (e.g., bracket-aware simulation).

## Sources
- [[../../raw/kaggle/v6-ensemble-documentation.md]] — primary documentation
- [[../../raw/kaggle/memory-2026-02-22.md]] — v11/v12 experimental notes

## Related
- [[../competitions/march-mania-2026]] — competition context
- [[../concepts/xgboost-ensembles]] — XGBoost technique details
- [[../concepts/calibration]] — probability calibration
- [[../concepts/feature-engineering-tabular]] — feature construction patterns
- [[../entities/xgboost]] — XGBoost framework
- [[../entities/lightgbm-catboost]] — LightGBM & CatBoost frameworks
