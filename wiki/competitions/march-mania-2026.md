---
title: "March Mania 2026 — NCAA Basketball Prediction"
tags: [kaggle, ncaa, basketball, ensemble, xgboost, lightgbm, logistic-regression]
date: 2026-04-14
source_count: 3
status: active
---

## Summary
March Mania 2026 is Kaggle's annual NCAA men's basketball tournament prediction competition (prize: $50K). Jason's best submission is **v6 weighted ensemble at 0.02210 Stage 1 LB score**. Stage 2 (actual 2026 bracket) was also submitted using v6 pipeline. Later attempts (v11, v12) did not improve on v6.

## Competition Metadata
- **Platform**: Kaggle
- **Prize**: $50,000
- **Metric**: Mean Squared Error (MSE) on win probabilities
- **Stage 1 Deadline**: ~March 2026 (historical game predictions)
- **Stage 2**: Actual 2026 tournament bracket (drops Selection Sunday, ~March 16–17)
- **Team**: Jason (solo)
- **Best LB Score**: 0.02210 (Stage 1, v6 ensemble)
- **Stage 2 Strategy**: Exact v6 pipeline applied to 2026 bracket/seeds

## Strategy Summary
Three-pipeline weighted ensemble. See [[../strategies/march-mania-v6-ensemble]] for full details.

Core insight: balance high-variance deep XGBoost trees with lower-variance hybrid/linear models to prevent overfitting to training years while capturing complex basketball dynamics.

## Submission History
| Version | Score | Notes |
|---------|-------|-------|
| v6 ensemble | **0.02210** | Best; 0.35×v2.9 + 0.35×v2.8 + 0.30×v5 |
| v11 (Adjusted Efficiency) | Worse than v6 | Adjusted efficiency metrics didn't help |
| v12 (Recency Weighting) | Worse than v6 | Recency weighting hurt generalization |
| v13 | Unknown | Alternate submission |

## What Worked
- Weighted ensemble of diverse model depths (5–9) and types (XGBoost, LightGBM, LogReg, Elo baseline)
- Four Factors features (eFG%, TOV%, ORB%, FT rate)
- Elo ratings as a baseline signal
- Tournament Experience as a feature (prior tournament appearances)
- Massey Ordinals for strength-of-schedule
- Pythagorean Win % for true strength estimation
- Calibrated probabilities (Platt scaling on LogReg component)

## What Didn't Work
- v11: Adjusted efficiency metrics — may have introduced noise or leakage
- v12: Recency weighting — tournament performance may not follow recent-season trends
- Going deeper than v6's architecture

## Open Questions
- What was the Stage 2 final score?
- Would team-specific tempo/pace features help?
- Does conference strength need explicit encoding?

## Key Files
- `raw/kaggle/v6-ensemble-documentation.md` — primary doc
- `raw/kaggle/memory-2026-02-22.md` — session notes with LB scores
- Scripts in `/Users/admin/Documents/Openclaw/workspace/`: `reproduce_v6.sh`, `generate_stage2_2026.py`

## Sources
- [[../../raw/kaggle/v6-ensemble-documentation.md]] — v6 ensemble architecture and rationale
- [[../../raw/kaggle/memory-2026-02-22.md]] — session notes with v11/v12 test results

## Related
- [[../strategies/march-mania-v6-ensemble]] — full ensemble architecture page
- [[../concepts/xgboost-ensembles]] — XGBoost technique used throughout
- [[../concepts/calibration]] — probability calibration used in v5 hybrid
- [[../entities/xgboost]] — primary model framework
