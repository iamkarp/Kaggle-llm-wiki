---
title: OOF vs LB Divergence
category: patterns
tags: [validation, leakage, generalization, diagnostic]
created: 2026-04-15
updated: 2026-04-15
---

# OOF vs LB Divergence

When out-of-fold (OOF) cross-validation scores don't predict public leaderboard scores, something is fundamentally wrong with either the validation strategy or the model's generalization.

## Types of Divergence

### OOF >> LB (OOF overestimates)
**The model overfits to the training distribution.** Common causes:
- Train/test distribution shift (different subjects, time periods, geographies)
- Data leakage in validation (same group in train and val)
- Features that encode training-distribution-specific patterns

**Example from WEAR 2026:** GBM with handcrafted features had OOF 0.59, LB 0.24. The features captured training-subject movement patterns that didn't transfer.

### OOF << LB (OOF underestimates)
**The model generalizes better than expected.** This can happen when:
- Test distribution is "easier" than the validation split
- The model learns truly invariant representations
- Random favorable alignment between model biases and test distribution

**Example from WEAR 2026:** PatchTST had OOF 0.52, LB 0.59. The learned representations transferred well, possibly because test subjects had more typical movement patterns.

## Diagnostic Table

| OOF F1 | LB F1 | Delta | Interpretation |
|--------|-------|-------|---------------|
| 0.62 | 0.22 | -0.40 | Severe overfitting to training distribution |
| 0.59 | 0.24 | -0.35 | Severe overfitting |
| 0.52 | 0.59 | +0.07 | Good generalization |
| 0.63 | 0.50 | -0.13 | Moderate overfitting (blend contamination) |

## What to Do

1. **First**: Ensure grouped CV matches the actual train/test split structure
2. **If OOF >> LB**: Your features/model are distribution-dependent. Switch to learned representations.
3. **If OOF << LB**: Your model generalizes well. Trust it and focus on improving OOF.
4. **Track the ratio**: If OOF improvement doesn't translate to LB improvement, you're optimizing the wrong thing.

## Key Insight

**High OOF does not mean good model.** In WEAR 2026, the submission with the highest OOF (0.6348) scored only 0.50 on LB, while the submission with the lowest OOF (0.5204) scored 0.59 on LB. Always submit to verify.

## See Also
- [[patterns/cross-subject-generalization]]
- [[mistakes/gbm-for-cross-subject-har]]
